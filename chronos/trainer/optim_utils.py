"""
chronos/trainer/optim_utils.py

Shared optimizer + LR-schedule helpers used by every Chronos training stage
(pretrain, SFT, DPO, ORPO, GRPO, distill) and the WebUI Train tab.

Two reasons to centralize this:

1. Decay groups: AdamW with weight_decay applied uniformly to all params
   destabilizes LayerNorm / RMSNorm scales and hurts embeddings. Standard
   recipe is to exclude norms, biases, and embeddings from decay. Doing
   this in one place means every trainer benefits from the same correct
   recipe and we don't get drift.

2. Warmup-then-cosine schedule. minimind's get_lr is pure cosine (effective
   LR at step 0 is already 0.55·base_lr), which destabilizes the first few
   hundred steps with small batches. Linear warmup over the first ~5% of
   steps then cosine decay to 10% of base is the boring-and-correct choice.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn


def _decay_groups(model: nn.Module) -> Tuple[list, list]:
    """Split trainable parameters into (decay, no_decay) lists.

    No-decay: any parameter whose owning module is a LayerNorm/RMSNorm,
    any 1-D parameter (typically bias / norm scale), and any embedding
    weight (regardless of dim — embeddings are looked-up rows, not
    rotated subspaces, so decay would push *all* tokens toward zero).
    """
    decay, no_decay = [], []
    seen = set()
    for mod in model.modules():
        is_norm = isinstance(mod, nn.LayerNorm) or mod.__class__.__name__.endswith("RMSNorm")
        is_embed = isinstance(mod, nn.Embedding)
        for p in mod.parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if is_norm or is_embed or p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    lr: float,
    *,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """AdamW with the standard transformer recipe applied to a Chronos model."""
    decay, no_decay = _decay_groups(model)
    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr), betas=betas, eps=eps,
    )


def get_lr(
    current_step: int,
    total_steps: int,
    base_lr: float,
    *,
    warmup_steps: int | None = None,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup → cosine decay to `min_lr_ratio · base_lr`.

    `current_step` is 1-indexed (matches every trainer's `enumerate(loader, start=1)`).
    Falls back to a constant `base_lr` when `total_steps <= 0`.
    """
    if total_steps is None or total_steps <= 0:
        return float(base_lr)
    if warmup_steps is None:
        warmup_steps = max(1, min(500, total_steps // 20))
    step = max(1, int(current_step))
    if step <= warmup_steps:
        return float(base_lr) * (step / float(warmup_steps))
    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)


def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate on every param-group in `optimizer`."""
    for g in optimizer.param_groups:
        g["lr"] = float(lr)
