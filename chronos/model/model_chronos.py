import sys
import chronos.deps  # ensure minimind on sys.path

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from model.model_minimind import (
    RMSNorm, precompute_freqs_cis, Attention, FeedForward,
)
from .config import ChronosConfig
from .lookahead_router import LookaheadRouter
from .moe_chronos import ChronosMOEFeedForward
from .hybrid_attention import make_attention


class ChronosBlock(nn.Module):
    """
    Transformer block with hybrid attention:
      even layers → MLAAttention  (compressed KV cache, global context)
      odd  layers → SlidingWindowAttention (bounded KV cache, local context)
    """

    def __init__(self, layer_id: int, config: ChronosConfig):
        super().__init__()
        self.layer_id = layer_id
        if getattr(config, 'use_hybrid_attention', True):
            self.self_attn = make_attention(layer_id, config)
        else:
            self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ChronosMOEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
        available_expert_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states = hidden_states + residual
        mlp_in = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, ChronosMOEFeedForward):
            hidden_states = hidden_states + self.mlp(mlp_in, available_expert_mask)
        else:
            hidden_states = hidden_states + self.mlp(mlp_in)
        return hidden_states, present_key_value


class ChronosModel(nn.Module):
    """
    Backbone: identical to MiniMindModel but:
    - Uses ChronosBlock (with ChronosMOEFeedForward)
    - Inserts LookaheadRouter after block 0
    - Exposes router_probs_seq for temporal loss computation
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            ChronosBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lookahead_router = LookaheadRouter(config)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        available_expert_masks=None,  # list[Tensor|None] per layer
        **kwargs,
    ):
        B, S = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # MLA caches only c_kv tuple; SlidingWindow caches (k, v) tuple
        # Determine start_pos from first non-None past_kv
        start_pos = 0
        for pkv in past_key_values:
            if pkv is not None:
                start_pos = pkv[0].shape[1]
                break

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Pass full freqs so hybrid attention layers can slice as needed
        position_embeddings = (
            self.freqs_cos,
            self.freqs_sin,
        )

        presents = []
        lookahead_probs = None  # [B, S, K, E] from block 0

        for i, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            mask = available_expert_masks[i] if available_expert_masks else None
            hidden_states, present = layer(
                hidden_states, position_embeddings,
                past_key_value=past_kv,
                use_cache=use_cache,
                attention_mask=attention_mask,
                available_expert_mask=mask,
            )
            presents.append(present)
            # Insert LookaheadRouter after block 0
            if i == 0:
                lookahead_probs = self.lookahead_router(hidden_states)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, ChronosMOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss, lookahead_probs


class ChronosForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ChronosConfig
    # Map of tied-weight keys (lm_head.weight ↔ model.embed_tokens.weight).
    # Newer Transformers expects a dict; older versions accept a list. We
    # use a dict to satisfy the newest API.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @property
    def all_tied_weights_keys(self) -> dict:
        """Compat shim: newer Transformers expects this property to exist
        on every PreTrainedModel; older base classes don't define it."""
        v = self._tied_weights_keys
        if isinstance(v, dict):
            return v
        if isinstance(v, (list, tuple, set)):
            return {k: None for k in v}
        return {}

    def __init__(self, config: ChronosConfig = None):
        self.config = config or ChronosConfig()
        super().__init__(self.config)
        self.model = ChronosModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        available_expert_masks=None,
        **kwargs,
    ):
        hidden_states, past_key_values, aux_loss, lookahead_probs = self.model(
            input_ids, attention_mask, past_key_values, use_cache,
            available_expert_masks=available_expert_masks,
        )
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            x = logits[..., :-1, :].contiguous()
            y = labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        ), lookahead_probs

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=512,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        do_sample=True,
        repetition_penalty=1.0,
        available_expert_masks=None,
        **kwargs,
    ):
        input_ids = kwargs.pop("input_ids", inputs)
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs, _ = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values,
                use_cache=use_cache,
                available_expert_masks=available_expert_masks,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
                )
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()
        return input_ids
