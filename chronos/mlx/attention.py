"""
chronos/mlx/attention.py — MLX-native MLA + Sliding Window Attention.

Design notes vs. the PyTorch version:
- No past_key_values list; instead we carry (c_kv_cache, k_rope_cache) as
  plain mx.array and concatenate explicitly each step.
- RoPE: computed with mx operations; no separate apply_rotary_pos_emb import.
- SlidingWindow: slice oldest tokens with array indexing, not assignment.
- mx.compile()-friendly: no Python branches on runtime tensor shapes.
"""
import math
import mlx.core as mx
import mlx.nn as nn


def _rope_freqs(dim: int, max_len: int = 4096, base: float = 10000.0) -> mx.array:
    """Precompute RoPE cos/sin tables [max_len, dim/2].

    `dim` is rounded down to the nearest even number internally so the
    cos/sin half-dimension is well-defined. Callers that pass an odd
    `dim` get tables that match the even portion only.
    """
    even = dim - (dim % 2)
    half = max(even // 2, 1)
    theta = 1.0 / (base ** (mx.arange(0, half, dtype=mx.float32) / half))
    pos = mx.arange(max_len, dtype=mx.float32)
    freqs = mx.outer(pos, theta)               # [T, half]
    return mx.cos(freqs), mx.sin(freqs)        # each [T, half]


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """
    Apply RoPE to x [B, T, nH, d].
    cos/sin: [T, d_eff/2] where d_eff = d rounded down to even.

    When d is odd, the trailing element is passed through unchanged so the
    output shape matches the input. (Some Chronos checkpoints were saved
    with rope_dim = head_dim - 1, which is odd whenever head_dim is even
    — head_dim = hidden_size / num_heads = 256/8 = 32 → rope_dim = 31.)
    """
    B, T, nH, d = x.shape
    d_eff = d - (d % 2)
    half = d_eff // 2
    x1 = x[..., :half]                          # [B, T, nH, half]
    x2 = x[..., half:d_eff]                     # [B, T, nH, half]
    c = cos[:T, :half][:, None, :]              # [T, 1, half]
    s = sin[:T, :half][:, None, :]
    rotated = mx.concatenate([x1 * c - x2 * s,
                              x1 * s + x2 * c], axis=-1)  # [B, T, nH, d_eff]
    if d_eff < d:
        # Append the unrotated tail so the output keeps the original last dim.
        tail = x[..., d_eff:]
        rotated = mx.concatenate([rotated, tail], axis=-1)
    return rotated


class MLAAttentionMLX(nn.Module):
    """
    Multi-head Latent Attention for MLX.

    KV cache stores (c_kv [B, T, kv_latent_dim], k_rope [B, T, nH, rope_dim]).
    Both are concatenated each decode step — no copy in/out of VRAM.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.n_heads
        self.head_dim = config.head_dim
        self.kv_latent_dim = getattr(config, 'kv_latent_dim', 64)
        # rope_dim is clamped to ≤ head_dim−1 to leave room for nope. The
        # MLX _apply_rope below handles odd rope_dim safely (trailing element
        # passes through unchanged), so there's no evenness requirement.
        self.rope_dim = min(
            max(1, getattr(config, 'rope_dim', min(64, self.head_dim))),
            self.head_dim - 1,
        )
        self.nope_dim = self.head_dim - self.rope_dim
        H = config.hidden_size

        self.q_nope_proj = nn.Linear(H, self.n_heads * self.nope_dim, bias=False)
        self.q_rope_proj = nn.Linear(H, self.n_heads * self.rope_dim, bias=False)
        self.kv_down_proj = nn.Linear(H, self.kv_latent_dim, bias=False)
        self.k_nope_proj  = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.nope_dim, bias=False)
        self.k_rope_proj  = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.rope_dim, bias=False)
        self.v_proj       = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj       = nn.Linear(self.n_heads * self.head_dim, H, bias=False)

        cos, sin = _rope_freqs(self.rope_dim)
        self.cos = cos   # [max_len, rope_dim/2]  — not a parameter
        self.sin = sin

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,       # (c_kv [B, T_past, kv_latent_dim], k_rope [B, T_past, nH, rope_dim])
    ):
        B, S, H = x.shape

        # ── Queries ───────────────────────────────────────────────
        q_nope = self.q_nope_proj(x).reshape(B, S, self.n_heads, self.nope_dim)
        q_rope = self.q_rope_proj(x).reshape(B, S, self.n_heads, self.rope_dim)

        # ── KV latent (what gets cached) ──────────────────────────
        c_kv = self.kv_down_proj(x)             # [B, S, kv_latent_dim]

        # Append to cache
        if cache is not None:
            c_kv_past, k_rope_past = cache
            c_kv_full = mx.concatenate([c_kv_past, c_kv], axis=1)
        else:
            c_kv_full = c_kv
            k_rope_past = None

        T_full = c_kv_full.shape[1]

        # ── Keys / Values (recomputed from cache) ─────────────────
        k_nope = self.k_nope_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.nope_dim)
        k_rope = self.k_rope_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.rope_dim)
        v      = self.v_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.head_dim)

        # ── RoPE ──────────────────────────────────────────────────
        past_len = T_full - S
        q_rope = _apply_rope(q_rope,
                              self.cos[past_len:past_len + S],
                              self.sin[past_len:past_len + S])
        k_rope = _apply_rope(k_rope, self.cos[:T_full], self.sin[:T_full])

        # Concat nope + rope
        q = mx.concatenate([q_nope, q_rope], axis=-1)  # [B, S, nH, head_dim]
        k = mx.concatenate([k_nope, k_rope], axis=-1)  # [B, T, nKV, head_dim]

        # GQA repeat
        if self.n_heads != self.n_kv_heads:
            rep = self.n_heads // self.n_kv_heads
            k = mx.repeat(k, rep, axis=2)
            v = mx.repeat(v, rep, axis=2)

        # ── Attention ─────────────────────────────────────────────
        scale = 1.0 / math.sqrt(self.head_dim)
        q = q.transpose(0, 2, 1, 3)    # [B, nH, S, d]
        k = k.transpose(0, 2, 1, 3)    # [B, nH, T, d]
        v = v.transpose(0, 2, 1, 3)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale   # [B, nH, S, T]
        if mask is not None:
            attn = attn + mask[None, None, :, :]
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, -1)

        new_cache = (c_kv_full, k_rope)
        return self.o_proj(out), new_cache


class SlidingWindowAttentionMLX(nn.Module):
    """
    Standard multi-head attention with a sliding KV window for MLX.
    KV cache is capped at window_size tokens — O(1) memory per step.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.n_heads
        self.head_dim = config.head_dim
        self.window_size = getattr(config, 'sliding_window_size', 2048)
        H = config.hidden_size

        self.q_proj = nn.Linear(H, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, H, bias=False)

        rope_dim = self.head_dim
        cos, sin = _rope_freqs(rope_dim)
        self.cos = cos
        self.sin = sin

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,  # (k [B, T_past, nKV, d], v [B, T_past, nKV, d])
    ):
        B, S, H = x.shape

        q = self.q_proj(x).reshape(B, S, self.n_heads,    self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim)

        if cache is not None:
            k_past, v_past = cache
            k = mx.concatenate([k_past, k], axis=1)
            v = mx.concatenate([v_past, v], axis=1)

        T_full = k.shape[1]
        past_len = T_full - S

        # Apply RoPE to new queries and all keys (before GQA repeat)
        q = _apply_rope(q, self.cos[past_len:past_len + S], self.sin[past_len:past_len + S])
        k = _apply_rope(k, self.cos[:T_full], self.sin[:T_full])

        # Slide window — keep only last window_size tokens
        if T_full > self.window_size:
            k = k[:, -self.window_size:, :, :]
            v = v[:, -self.window_size:, :, :]

        # GQA repeat for attention (k/v expand to n_heads)
        k_attn = k
        v_attn = v
        if self.n_heads != self.n_kv_heads:
            rep = self.n_heads // self.n_kv_heads
            k_attn = mx.repeat(k, rep, axis=2)
            v_attn = mx.repeat(v, rep, axis=2)

        scale = 1.0 / math.sqrt(self.head_dim)
        # Transpose to [B, nH, T, d] for batched matmul
        qt = q.transpose(0, 2, 1, 3)
        kt = k_attn.transpose(0, 2, 1, 3)
        vt = v_attn.transpose(0, 2, 1, 3)
        attn = (qt @ kt.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            # mask is [S, T_k] (2D causal); broadcast over [B, nH, S, T_k]
            attn = attn + mask[None, None, :, :]
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(q.dtype)
        out = (attn @ vt).transpose(0, 2, 1, 3).reshape(B, S, -1)

        # Cache stores [B, T, nKV, d] (pre-repeat, pre-transpose)
        new_cache = (k, v)
        return self.o_proj(out), new_cache
