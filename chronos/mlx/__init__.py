"""
chronos/mlx/__init__.py

MLX-native backend for Project Chronos (Apple Silicon / Metal).

MLX uses unified memory — CPU and GPU share the same physical DRAM.
There is no H2D copy, no pinned memory, and no CUDA streams.
The "SSD prefetch" concept maps differently:
  - Tier 0 (hot):  mx.array on GPU (materialised via mx.eval())
  - Tier 1 (warm): mx.array in unified memory (not yet eval'd)
  - Tier 2 (cold): .safetensors / .npy files on SSD

Key differences from the CUDA backend:
  - No ExpertStore.promote_to_vram() — mx.eval() pins to Metal on first use
  - No H2D stream — Metal handles async via its own command queue
  - torch.compile unavailable; mx.compile() used instead
  - All modules are mlx.nn.Module, not torch.nn.Module
"""
from chronos.mlx.model import ChronosMLXModel
from chronos.mlx.moe import ChronosMLXMOE
from chronos.mlx.attention import MLAAttentionMLX, SlidingWindowAttentionMLX
from chronos.mlx.expert_store import MLXExpertStore
from chronos.mlx.inference import ChronosMLXInferenceEngine

__all__ = [
    "ChronosMLXModel",
    "ChronosMLXMOE",
    "MLAAttentionMLX",
    "SlidingWindowAttentionMLX",
    "MLXExpertStore",
    "ChronosMLXInferenceEngine",
]
