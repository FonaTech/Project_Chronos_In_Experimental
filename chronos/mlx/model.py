"""
chronos/mlx/model.py — Full MLX Chronos model.

Architecture mirrors the PyTorch version (ChronosForCausalLM) but uses:
- mlx.nn.Module throughout
- MLAAttentionMLX (even layers) / SlidingWindowAttentionMLX (odd layers)
- ChronosMLXMOE for feed-forward
- LookaheadRouter as a plain Linear projection
- mx.compile()-friendly: no Python control flow on runtime shapes

On Apple Silicon, mx.eval() materialises arrays into Metal shared memory.
Prefetch = eagerly eval expert weights during the prefetch window.
"""
import math
import mlx.core as mx
import mlx.nn as nn

from chronos.mlx.attention import MLAAttentionMLX, SlidingWindowAttentionMLX
from chronos.mlx.moe import ChronosMLXMOE, FeedForwardMLX


class RMSNormMLX(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class LookaheadRouterMLX(nn.Module):
    """Predicts expert probabilities for t, t+1, ..., t+K steps."""

    def __init__(self, config):
        super().__init__()
        K = getattr(config, 'lookahead_steps', 2)
        self.lookahead_steps = K
        hidden = max(1, config.hidden_size // 4)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, config.num_experts * (K + 1), bias=False),
        )
        self.num_experts = config.num_experts

    def __call__(self, x: mx.array) -> mx.array:
        B, S, H = x.shape
        out = self.proj(x).astype(mx.float32)                          # [B, S, E*(K+1)]
        out = out.reshape(B, S, self.lookahead_steps + 1, self.num_experts)
        return mx.softmax(out, axis=-1)                                # [B, S, K+1, E]


class ChronosMLXBlock(nn.Module):
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.input_norm  = RMSNormMLX(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm   = RMSNormMLX(config.hidden_size, eps=config.rms_norm_eps)

        use_hybrid = getattr(config, 'use_hybrid_attention', True)
        if use_hybrid:
            if layer_id % 2 == 0:
                self.self_attn = MLAAttentionMLX(config)
            else:
                self.self_attn = SlidingWindowAttentionMLX(config)
        else:
            self.self_attn = SlidingWindowAttentionMLX(config)

        if getattr(config, 'use_moe', True):
            self.mlp = ChronosMLXMOE(config)
        else:
            self.mlp = FeedForwardMLX(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,
        available_expert_mask: mx.array = None,
    ):
        h, new_cache = self.self_attn(self.input_norm(x), mask=mask, cache=cache)
        x = x + h
        mlp_in = self.post_norm(x)
        if isinstance(self.mlp, ChronosMLXMOE):
            x = x + self.mlp(mlp_in, available_expert_mask=available_expert_mask)
        else:
            x = x + self.mlp(mlp_in)
        return x, new_cache


class ChronosMLXModel(nn.Module):
    """
    Full Chronos model in MLX.

    forward() returns (logits [B, S, V], lookahead_probs [B, S, K+1, E])
    Cache: list of per-layer attention caches (passed back in, None at start).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [ChronosMLXBlock(i, config)
                       for i in range(config.num_hidden_layers)]
        self.norm = RMSNormMLX(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # LookaheadRouter inserted after block 0
        self.lookahead_router = LookaheadRouterMLX(config)

    def __call__(
        self,
        input_ids: mx.array,               # [B, S]
        cache=None,                         # list of per-layer caches, or None
        available_expert_masks=None,        # list of [E] float masks, or None
    ):
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)   # [B, S, H]

        # Causal mask — only needed for prefill (S > 1); decode (S==1) needs no mask
        mask = None
        if S > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(S)
            mask = mask.astype(x.dtype)

        new_caches = []
        lookahead_probs = None

        if cache is None:
            cache = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            avail = (available_expert_masks[i]
                     if available_expert_masks is not None else None)
            x, lyr_cache = layer(x, mask=mask, cache=cache[i],
                                 available_expert_mask=avail)
            new_caches.append(lyr_cache)

            # Lookahead routing after block 0
            if i == 0:
                lookahead_probs = self.lookahead_router(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, lookahead_probs, new_caches

    @staticmethod
    def from_chronos_pytorch(pt_model, config, include_experts: bool = True):
        """
        Convert a trained PyTorch ChronosForCausalLM to ChronosMLXModel.
        Copies weights layer by layer via numpy as bridge.
        """
        mlx_model = ChronosMLXModel(config)

        def _to_mx(pt_tensor):
            return mx.array(pt_tensor.detach().float().cpu().numpy())

        def _assign_linear(linear, sd, key, missing):
            if key in sd:
                linear.weight = _to_mx(sd[key])
            else:
                missing.append(key)

        def _assign_norm(norm, sd, key, missing):
            if key in sd:
                norm.weight = _to_mx(sd[key])
            else:
                missing.append(key)

        # Embedding + head
        sd = pt_model.state_dict()
        missing = []
        for key, obj in [
            ("model.embed_tokens.weight", mlx_model.embed_tokens),
            ("lm_head.weight", mlx_model.lm_head),
        ]:
            if key in sd:
                obj.weight = _to_mx(sd[key])
            else:
                missing.append(key)
        _assign_norm(mlx_model.norm, sd, "model.norm.weight", missing)

        for i in range(config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            blk = mlx_model.layers[i]
            _assign_norm(blk.input_norm, sd, f"{prefix}.input_layernorm.weight", missing)
            _assign_norm(blk.post_norm, sd, f"{prefix}.post_attention_layernorm.weight", missing)

            attn = blk.self_attn
            attn_linears = [
                "q_nope_proj", "q_rope_proj", "kv_down_proj",
                "k_nope_proj", "k_rope_proj", "v_proj", "o_proj",
                "q_proj", "k_proj",
            ]
            for name in attn_linears:
                if hasattr(attn, name):
                    _assign_linear(getattr(attn, name), sd, f"{prefix}.self_attn.{name}.weight", missing)
            for name in ["q_nope_norm", "q_rope_norm", "kv_down_norm", "q_norm", "k_norm"]:
                if hasattr(attn, name):
                    _assign_norm(getattr(attn, name), sd, f"{prefix}.self_attn.{name}.weight", missing)

            moe = blk.mlp
            if isinstance(moe, ChronosMLXMOE):
                _assign_linear(moe.gate, sd, f"{prefix}.mlp.gate.weight", missing)
                if include_experts:
                    for ei, expert in enumerate(moe.experts):
                        for proj in ("gate_proj", "up_proj", "down_proj"):
                            _assign_linear(
                                getattr(expert, proj),
                                sd,
                                f"{prefix}.mlp.experts.{ei}.{proj}.weight",
                                missing,
                            )
                for si, expert in enumerate(moe.shared_experts):
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        _assign_linear(
                            getattr(expert, proj),
                            sd,
                            f"{prefix}.mlp.shared_experts.{si}.{proj}.weight",
                            missing,
                        )

        proj_layers = mlx_model.lookahead_router.proj.layers
        _assign_linear(proj_layers[0], sd, "model.lookahead_router.proj.0.weight", missing)
        _assign_linear(proj_layers[2], sd, "model.lookahead_router.proj.2.weight", missing)

        if missing:
            raise RuntimeError(
                "MLX conversion is missing Chronos checkpoint tensors: "
                + ", ".join(missing[:40])
            )
        mx.eval(mlx_model.parameters())
        return mlx_model
