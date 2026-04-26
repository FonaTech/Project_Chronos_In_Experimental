"""MLX/PyTorch checkpoint conversion helpers."""
from __future__ import annotations

import torch
import mlx.core as mx
import numpy as np

from chronos.mlx.attention import MLAAttentionMLX, SlidingWindowAttentionMLX
from chronos.mlx.moe import ChronosMLXMOE


def _torch_tensor(arr) -> torch.Tensor:
    if isinstance(arr, mx.array):
        arr = arr.astype(mx.float32)
        mx.eval(arr)
    return torch.from_numpy(np.array(arr, dtype=np.float32)).contiguous()


def mlx_state_to_torch(mlx_model, config) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    sd["model.embed_tokens.weight"] = _torch_tensor(mlx_model.embed_tokens.weight)
    sd["lm_head.weight"] = _torch_tensor(mlx_model.lm_head.weight)
    sd["model.norm.weight"] = _torch_tensor(mlx_model.norm.weight)

    for i, blk in enumerate(mlx_model.layers):
        prefix = f"model.layers.{i}"
        sd[f"{prefix}.input_layernorm.weight"] = _torch_tensor(blk.input_norm.weight)
        sd[f"{prefix}.post_attention_layernorm.weight"] = _torch_tensor(blk.post_norm.weight)
        attn = blk.self_attn
        if isinstance(attn, MLAAttentionMLX):
            for name in (
                "q_nope_proj", "q_rope_proj", "kv_down_proj",
                "k_nope_proj", "k_rope_proj", "v_proj", "o_proj",
            ):
                sd[f"{prefix}.self_attn.{name}.weight"] = _torch_tensor(getattr(attn, name).weight)
            for name in ("q_nope_norm", "q_rope_norm", "kv_down_norm"):
                sd[f"{prefix}.self_attn.{name}.weight"] = _torch_tensor(getattr(attn, name).weight)
        elif isinstance(attn, SlidingWindowAttentionMLX):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sd[f"{prefix}.self_attn.{name}.weight"] = _torch_tensor(getattr(attn, name).weight)
            for name in ("q_norm", "k_norm"):
                sd[f"{prefix}.self_attn.{name}.weight"] = _torch_tensor(getattr(attn, name).weight)

        moe = blk.mlp
        if isinstance(moe, ChronosMLXMOE):
            sd[f"{prefix}.mlp.gate.weight"] = _torch_tensor(moe.gate.weight)
            for ei, expert in enumerate(moe.experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    sd[f"{prefix}.mlp.experts.{ei}.{proj}.weight"] = _torch_tensor(
                        getattr(expert, proj).weight
                    )
            for si, expert in enumerate(moe.shared_experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    sd[f"{prefix}.mlp.shared_experts.{si}.{proj}.weight"] = _torch_tensor(
                        getattr(expert, proj).weight
                    )

    proj_layers = mlx_model.lookahead_router.proj.layers
    sd["model.lookahead_router.proj.0.weight"] = _torch_tensor(proj_layers[0].weight)
    sd["model.lookahead_router.proj.2.weight"] = _torch_tensor(proj_layers[2].weight)
    return sd
