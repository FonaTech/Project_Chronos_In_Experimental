"""
chronos/model/hybrid_attention.py

Hybrid attention for Project Chronos:
- MLAAttention: Multi-head Latent Attention (DeepSeek-style)
  Compresses KV cache via low-rank projection: hidden_dim → kv_latent_dim
  Cache stores latent vectors instead of full KV → ~8-16x smaller KV cache

- SlidingWindowAttention: local attention with fixed window
  KV cache capped at window_size tokens → O(1) memory per step

- HybridAttentionBlock: alternates MLA (even layers) / SlidingWindow (odd layers)
  Even layers: MLA for global context with compressed cache
  Odd layers: SlidingWindow for local patterns with bounded cache
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_minimind import RMSNorm, apply_rotary_pos_emb, repeat_kv
from .config import ChronosConfig


# ── MLA: Multi-head Latent Attention ─────────────────────────────

class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek MLA style).

    KV compression:
        c_kv = W_kv_down(x)          [B, S, kv_latent_dim]   ← cached
        k, v  = W_kv_up(c_kv)        [B, S, nH, dH]          ← recomputed from cache

    Query decoupled RoPE:
        q_nope = W_q_nope(x)         [B, S, nH, dH - rope_dim]
        q_rope = W_q_rope(x)         [B, S, nH, rope_dim]     ← RoPE applied here
        k_rope = W_k_rope(c_kv)      [B, S, nH, rope_dim]     ← RoPE applied here

    Cache stores only c_kv (latent) + k_rope → much smaller than full KV.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        # KV latent dimension: much smaller than n_kv_heads * head_dim
        self.kv_latent_dim = getattr(config, 'kv_latent_dim',
                                     max(64, self.n_kv_heads * self.head_dim // 8))
        # RoPE dimension per head (subset of head_dim); must leave room for nope
        self.rope_dim = min(getattr(config, 'rope_dim', min(64, self.head_dim)), self.head_dim - 1)
        self.rope_dim = max(1, self.rope_dim)  # at least 1
        self.nope_dim = self.head_dim - self.rope_dim

        H = config.hidden_size

        # Query projections
        self.q_nope_proj = nn.Linear(H, self.n_heads * self.nope_dim, bias=False)
        self.q_rope_proj = nn.Linear(H, self.n_heads * self.rope_dim, bias=False)
        self.q_nope_norm = RMSNorm(self.nope_dim, eps=config.rms_norm_eps)
        self.q_rope_norm = RMSNorm(self.rope_dim, eps=config.rms_norm_eps)

        # KV down-projection (what gets cached)
        self.kv_down_proj = nn.Linear(H, self.kv_latent_dim, bias=False)
        self.kv_down_norm = RMSNorm(self.kv_latent_dim, eps=config.rms_norm_eps)

        # KV up-projections (recomputed from cache at inference)
        self.k_nope_proj = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.nope_dim, bias=False)
        self.k_rope_proj = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.rope_dim, bias=False)
        self.v_proj = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        B, S, _ = x.shape
        cos, sin = position_embeddings

        # Query
        q_nope = self.q_nope_norm(
            self.q_nope_proj(x).view(B, S, self.n_heads, self.nope_dim)
        )
        q_rope = self.q_rope_norm(
            self.q_rope_proj(x).view(B, S, self.n_heads, self.rope_dim)
        )
        # apply_rotary_pos_emb expects [B, S, nH, d] (NOT transposed)
        # cos/sin are full-length buffers; slice to current step positions
        past_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        cos_r = cos[past_len:past_len + S, :self.rope_dim]   # [S, rope_dim]
        sin_r = sin[past_len:past_len + S, :self.rope_dim]
        q_rope, _ = apply_rotary_pos_emb(q_rope, q_rope, cos_r, sin_r)  # [B, S, nH, rope_dim]
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, S, nH, head_dim]

        # KV latent (this is what we cache)
        c_kv = self.kv_down_norm(self.kv_down_proj(x))  # [B, S, kv_latent_dim]

        if past_key_value is not None:
            past_c_kv = past_key_value[0]
            c_kv_full = torch.cat([past_c_kv, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        full_len = c_kv_full.shape[1]

        # Expand KV from latent
        k_nope = self.k_nope_proj(c_kv_full).view(B, full_len, self.n_kv_heads, self.nope_dim)
        k_rope_raw = self.k_rope_proj(c_kv_full).view(B, full_len, self.n_kv_heads, self.rope_dim)
        v = self.v_proj(c_kv_full).view(B, full_len, self.n_kv_heads, self.head_dim)

        # RoPE on keys over full cached sequence — [B, T, nKV, rope_dim]
        cos_full = cos[:full_len, :self.rope_dim]
        sin_full = sin[:full_len, :self.rope_dim]
        k_rope, _ = apply_rotary_pos_emb(k_rope_raw, k_rope_raw, cos_full, sin_full)
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, T, nKV, head_dim]

        past_kv = (c_kv_full,) if use_cache else None

        # Attention
        q = q.transpose(1, 2)                          # [B, nH, S, dH]
        k = repeat_kv(k, self.n_rep).transpose(1, 2)   # [B, nH, T, dH]
        v = repeat_kv(v, self.n_rep).transpose(1, 2)   # [B, nH, T, dH]

        T = k.shape[2]
        if self.flash and S > 1 and attention_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -S:] += torch.full((S, S), float('-inf'), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            out = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(q)) @ v

        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.resid_dropout(self.o_proj(out)), past_kv


# ── Sliding Window Attention ──────────────────────────────────────

class SlidingWindowAttention(nn.Module):
    """
    Local sliding window attention. KV cache is capped at window_size tokens,
    preventing unbounded growth in long-context generation.

    During prefill: full causal attention within each window.
    During decode:  KV cache evicts oldest tokens beyond window_size.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout
        self.window_size = getattr(config, 'sliding_window_size', 2048)

        H = config.hidden_size
        self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.flash = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        B, S, _ = x.shape
        cos, sin = position_embeddings
        # cos/sin are full-length buffers; slice to current positions
        past_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        cos_s = cos[past_len:past_len + S]
        sin_s = sin[past_len:past_len + S]

        xq = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim))
        xk = self.k_norm(self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim))
        xv = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_pos_emb(xq, xk, cos_s, sin_s)  # [B, S, nH/nKV, d]

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # Evict tokens beyond window_size (sliding window)
        if xk.shape[1] > self.window_size:
            xk = xk[:, -self.window_size:, :, :]
            xv = xv[:, -self.window_size:, :, :]

        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        T = xk.shape[2]
        if self.flash and S > 1 and attention_mask is None:
            out = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=(T == S),
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if T == S:
                scores[:, :, :, -S:] += torch.full((S, S), float('-inf'), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask[:, :, :, -T:].unsqueeze(1)) * -1e9
            out = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.resid_dropout(self.o_proj(out)), past_kv


def make_attention(layer_id: int, config: ChronosConfig) -> nn.Module:
    """
    Layer assignment:
      even layers → MLAAttention  (global context, compressed KV cache)
      odd  layers → SlidingWindowAttention (local patterns, bounded KV cache)
    """
    if layer_id % 2 == 0:
        return MLAAttention(config)
    return SlidingWindowAttention(config)
