"""
chronos/mlx/moe.py — MLX-native ChronosMOEFeedForward.

Soft gating is expressed as weighted sum (compile-safe):
    out = avail[i] * expert_out + (1 - avail[i]) * shared_out

mx.compile() traces a single graph — no Python branches on mask values.
"""
import mlx.core as mx
import mlx.nn as nn


class FeedForwardMLX(nn.Module):
    """SwiGLU FFN matching minimind's FeedForward structure."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class ChronosMLXMOE(nn.Module):
    """
    MLX-native Mixture-of-Experts with:
    - LookaheadRouter output stored as last_router_probs
    - Compile-safe soft gating (avail mask → float multiply)
    - Shared expert fallback at zero I/O cost (unified memory)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        H = config.hidden_size
        I = config.moe_intermediate_size

        self.gate = nn.Linear(H, config.num_experts, bias=False)
        self.experts = [FeedForwardMLX(H, I) for _ in range(config.num_experts)]
        self.shared_experts = [
            FeedForwardMLX(H, I) for _ in range(config.num_shared_experts)
        ]
        self.last_router_probs = None  # [B, S, E] stored after each forward

    def _shared_out(self, x: mx.array) -> mx.array:
        return sum(e(x) for e in self.shared_experts) / len(self.shared_experts)

    def __call__(
        self,
        x: mx.array,
        available_expert_mask: mx.array = None,
    ) -> mx.array:
        """
        x: [B, S, H]
        available_expert_mask: [num_experts] float (1.0 = in GPU, 0.0 = miss)
                                or None for training (all available).
        """
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)                        # [N, H]

        scores = mx.softmax(self.gate(x_flat), axis=-1)  # [N, E]
        self.last_router_probs = scores.reshape(B, S, self.num_experts)

        # top-k routing
        topk_idx = mx.argpartition(-scores, kth=self.num_experts_per_tok - 1, axis=-1)
        topk_idx = topk_idx[:, :self.num_experts_per_tok]              # [N, K]
        topk_weight = mx.take_along_axis(scores, topk_idx, axis=-1)   # [N, K]
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(axis=-1, keepdims=True) + 1e-20)

        y = mx.zeros_like(x_flat)

        if available_expert_mask is None:
            # Training: standard MoE scatter
            for i, expert in enumerate(self.experts):
                # mask over tokens where expert i is selected
                sel = (topk_idx == i)            # [N, K] bool
                tok_mask = sel.any(axis=-1)       # [N] bool
                w = mx.where(sel, topk_weight, 0.0).sum(axis=-1, keepdims=True)  # [N, 1]
                contrib = expert(x_flat) * w * tok_mask[:, None]
                y = y + contrib
        else:
            # Inference: compile-safe soft gating
            avail = available_expert_mask.astype(x_flat.dtype)  # [E] float
            shared_out = self._shared_out(x_flat)               # [N, H]

            for i, expert in enumerate(self.experts):
                sel = (topk_idx == i)
                tok_mask = sel.any(axis=-1)
                w = mx.where(sel, topk_weight, 0.0).sum(axis=-1, keepdims=True)

                expert_out  = expert(x_flat)
                # avail[i]==1 → expert; avail[i]==0 → shared fallback
                blended = avail[i] * expert_out + (1.0 - avail[i]) * shared_out
                contrib = blended * w * tok_mask[:, None]
                y = y + contrib

        return y.reshape(B, S, H)
