"""Shared helpers for standalone Chronos stage scripts."""
from __future__ import annotations

import os

from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_controlled,
    resolve_checkpoint_path,
)
from chronos.model.config import ChronosConfig


TOPOLOGY_ARG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_experts",
    "num_experts_per_tok",
    "num_shared_experts",
    "lookahead_steps",
    "moe_intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "rope_dim",
    "kv_latent_dim",
    "vocab_size",
)


def add_topology_args(parser, *, defaults: bool = False):
    """Add topology CLI args.

    Alignment scripts use ``defaults=False`` so omitted args do not override
    the upstream checkpoint. Pretraining can pass ``defaults=True`` because it
    creates the first topology.
    """
    d = None if not defaults else 0
    parser.add_argument("--hidden_size", type=int, default=(512 if defaults else None))
    parser.add_argument("--num_hidden_layers", type=int, default=(8 if defaults else None))
    parser.add_argument("--num_experts", type=int, default=(4 if defaults else None))
    parser.add_argument("--num_experts_per_tok", type=int, default=(2 if defaults else None))
    parser.add_argument("--num_shared_experts", type=int, default=(1 if defaults else None))
    parser.add_argument("--lookahead_steps", type=int, default=(2 if defaults else None))
    parser.add_argument("--moe_intermediate_size", type=int, default=d)
    parser.add_argument("--num_attention_heads", type=int, default=(8 if defaults else None))
    parser.add_argument("--num_key_value_heads", type=int, default=(4 if defaults else None))
    parser.add_argument("--rope_dim", type=int, default=(32 if defaults else None))
    parser.add_argument("--kv_latent_dim", type=int, default=(64 if defaults else None))
    parser.add_argument("--vocab_size", type=int, default=(6400 if defaults else None))


def topology_overrides_from_args(args) -> dict:
    out = {}
    for key in TOPOLOGY_ARG_FIELDS:
        value = getattr(args, key, None)
        if value is None:
            continue
        if key == "moe_intermediate_size" and value == 0:
            continue
        out[key] = value
    return out


def build_pretrain_config(args) -> ChronosConfig:
    kwargs = topology_overrides_from_args(args)
    kwargs.update({
        "max_position_embeddings": getattr(args, "max_seq_len", 512),
        "lambda_balance": getattr(args, "lambda_balance", 5e-4),
        "lambda_temporal": getattr(args, "lambda_temporal", 1e-3),
        "lambda_lookahead": getattr(args, "lambda_lookahead", 0.1),
        "lambda_lookahead_topk": getattr(args, "lambda_lookahead_topk", 0.05),
        "vram_budget_gb": getattr(args, "vram_budget_gb", 4.0),
        "use_moe": True,
    })
    fallback_mask_prob = getattr(args, "fallback_mask_prob", None)
    if fallback_mask_prob is not None:
        kwargs["fallback_mask_prob"] = fallback_mask_prob
    return ChronosConfig(**kwargs)


def build_config_from_upstream(
    args,
    *,
    default_stem: str,
    max_positions: int,
    lambda_router_anchor: float = 0.0,
) -> tuple[ChronosConfig, str, list[str]]:
    hidden = getattr(args, "hidden_size", None)
    requested = getattr(args, "from_weight", default_stem) or default_stem
    explicit = None
    if requested.endswith(".pth") or os.path.sep in requested:
        explicit = requested
        requested = os.path.splitext(os.path.basename(requested))[0]
    ckpt = resolve_checkpoint_path(
        getattr(args, "save_dir", "out"),
        requested,
        hidden_size=hidden,
        explicit_path=explicit,
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"required upstream checkpoint not found: {ckpt}. "
            "Run the prior stage first or pass --from_weight/--save_dir."
        )
    overrides = topology_overrides_from_args(args)
    overrides["lambda_router_anchor"] = lambda_router_anchor
    cfg, sources = chronos_config_from_checkpoint(
        ckpt,
        overrides=overrides,
        require_unsniffable=True,
    )
    cfg.max_position_embeddings = max(int(getattr(cfg, "max_position_embeddings", 0) or 0), int(max_positions))
    return cfg, ckpt, sources


def load_required_checkpoint(model, checkpoint_path: str, device: str) -> dict:
    # Always deserialize checkpoints on CPU first. Loading directly to MPS/CUDA
    # can create a large transient device-memory spike before the model copy
    # settles.
    state = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    load_state_dict_controlled(model, state)
    return state
