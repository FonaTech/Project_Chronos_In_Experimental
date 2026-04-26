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


def _rope_freqs(
    dim: int,
    max_len: int = 4096,
    base: float = 1_000_000.0,
    rope_scaling: dict | None = None,
) -> mx.array:
    """Precompute MiniMind/PyTorch-compatible RoPE tables [max_len, dim].

    The PyTorch path uses duplicated cos/sin columns and rotate-half over the
    whole last dimension. MLX must mirror that exactly for checkpoint parity.
    """
    half = max(dim // 2, 1)
    theta = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32)[:half] / dim))
    attn_factor = 1.0
    if rope_scaling is not None:
        orig_max = float(rope_scaling.get("original_max_position_embeddings", 2048))
        factor = float(rope_scaling.get("factor", 16))
        beta_fast = float(rope_scaling.get("beta_fast", 32.0))
        beta_slow = float(rope_scaling.get("beta_slow", 1.0))
        attn_factor = float(rope_scaling.get("attention_factor", 1.0))
        if max_len / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), half - 1)
            ramp = mx.clip((mx.arange(half, dtype=mx.float32) - low) / max(high - low, 0.001), 0, 1)
            theta = theta * (1 - ramp + ramp / factor)
    pos = mx.arange(max_len, dtype=mx.float32)
    freqs = mx.outer(pos, theta)               # [T, half]
    cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1) * attn_factor
    sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1) * attn_factor
    return cos, sin


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """
    Apply RoPE to x [B, T, nH, d].
    cos/sin: full duplicated MiniMind tables, sliced to x.shape[-1].
    """
    _B, T, _nH, d = x.shape
    c = cos[:T, :d][:, None, :]
    s = sin[:T, :d][:, None, :]
    half = d // 2
    rotated_half = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return (x * c + rotated_half * s).astype(x.dtype)


class _RopeCacheMixin:
    def _init_rope_cache(self, config, dim: int) -> None:
        rope_len = int(getattr(config, "max_position_embeddings", 4096))
        object.__setattr__(self, "_rope_dim", int(dim))
        object.__setattr__(self, "_rope_base", float(getattr(config, "rope_theta", 1_000_000.0)))
        object.__setattr__(self, "_rope_scaling", getattr(config, "rope_scaling", None))
        cos, sin = _rope_freqs(
            int(dim),
            max_len=rope_len,
            base=self._rope_base,
            rope_scaling=self._rope_scaling,
        )
        object.__setattr__(self, "_cos", cos)
        object.__setattr__(self, "_sin", sin)

    def _ensure_rope_cache(self, needed_len: int) -> None:
        current_len = int(self._cos.shape[0])
        if int(needed_len) <= current_len:
            return
        # Grow geometrically to avoid reallocating every decode step past
        # max_position_embeddings.
        new_len = max(int(needed_len), current_len * 2)
        cos, sin = _rope_freqs(
            int(self._rope_dim),
            max_len=new_len,
            base=float(self._rope_base),
            rope_scaling=self._rope_scaling,
        )
        object.__setattr__(self, "_cos", cos)
        object.__setattr__(self, "_sin", sin)


class MLAAttentionMLX(_RopeCacheMixin, nn.Module):
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
        self.q_nope_norm = nn.RMSNorm(self.nope_dim, eps=config.rms_norm_eps)
        self.q_rope_norm = nn.RMSNorm(self.rope_dim, eps=config.rms_norm_eps)
        self.kv_down_proj = nn.Linear(H, self.kv_latent_dim, bias=False)
        self.kv_down_norm = nn.RMSNorm(self.kv_latent_dim, eps=config.rms_norm_eps)
        self.k_nope_proj  = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.nope_dim, bias=False)
        self.k_rope_proj  = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.rope_dim, bias=False)
        self.v_proj       = nn.Linear(self.kv_latent_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj       = nn.Linear(self.n_heads * self.head_dim, H, bias=False)

        self._init_rope_cache(config, self.head_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,       # (c_kv [B, T_past, kv_latent_dim], k_rope [B, T_past, nH, rope_dim])
    ):
        B, S, H = x.shape

        # ── Queries ───────────────────────────────────────────────
        q_nope = self.q_nope_norm(
            self.q_nope_proj(x).reshape(B, S, self.n_heads, self.nope_dim)
        )
        q_rope = self.q_rope_norm(
            self.q_rope_proj(x).reshape(B, S, self.n_heads, self.rope_dim)
        )

        # ── KV latent (what gets cached) ──────────────────────────
        c_kv = self.kv_down_norm(self.kv_down_proj(x))             # [B, S, kv_latent_dim]

        # Append to cache
        if cache is not None:
            c_kv_past, k_rope_past = cache
            c_kv_full = mx.concatenate([c_kv_past, c_kv], axis=1)
        else:
            c_kv_full = c_kv
            k_rope_past = None

        T_full = c_kv_full.shape[1]
        self._ensure_rope_cache(T_full)

        # ── Keys / Values (recomputed from cache) ─────────────────
        k_nope = self.k_nope_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.nope_dim)
        k_rope = self.k_rope_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.rope_dim)
        v      = self.v_proj(c_kv_full).reshape(B, T_full, self.n_kv_heads, self.head_dim)

        # ── RoPE ──────────────────────────────────────────────────
        past_len = T_full - S
        q_rope = _apply_rope(q_rope,
                              self._cos[past_len:past_len + S],
                              self._sin[past_len:past_len + S])
        k_rope = _apply_rope(k_rope, self._cos[:T_full], self._sin[:T_full])

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


class SlidingWindowAttentionMLX(_RopeCacheMixin, nn.Module):
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
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self._init_rope_cache(config, self.head_dim)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,  # (k [B, T_past, nKV, d], v [B, T_past, nKV, d])
    ):
        B, S, H = x.shape

        q = self.q_norm(self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim))
        k_new = self.k_norm(self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim))
        v_new = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim)

        past_len = cache[0].shape[1] if cache is not None else 0
        self._ensure_rope_cache(past_len + S)
        q = _apply_rope(q, self._cos[past_len:past_len + S], self._sin[past_len:past_len + S])
        k_new = _apply_rope(k_new, self._cos[past_len:past_len + S], self._sin[past_len:past_len + S])

        if cache is not None:
            k_past, v_past = cache
            k = mx.concatenate([k_past, k_new], axis=1)
            v = mx.concatenate([v_past, v_new], axis=1)
        else:
            k = k_new
            v = v_new

        T_full = k.shape[1]

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
