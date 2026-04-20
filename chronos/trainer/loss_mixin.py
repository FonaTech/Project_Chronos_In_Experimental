"""
chronos/trainer/loss_mixin.py

Reusable loss-mixing helpers shared across all Chronos training stages
(pretrain, SFT, DPO, ORPO, GRPO). Centralizing this keeps each stage
trainer thin and guarantees lookahead supervision is preserved through
the whole 6-stage chain rather than being silently dropped by a stage
that forgot to add it.

Two helpers are exposed:

- ``chronos_loss_term(model, base_loss, lookahead_probs, config)`` —
  appends λ_balance·L_aux + λ_temporal·L_temporal + λ_lookahead·L_lookahead
  to any base loss (CE for pretrain/SFT, DPO loss, ORPO loss, etc.).

- ``router_kl_anchor(current_router_probs, ref_router_probs, lambda_anchor)`` —
  KL(current || reference) penalty. Captured at the start of an alignment
  stage by running the freshly-loaded model in eval mode on a calibration
  batch; prevents the alignment gradients from drifting routing away from
  the pretrained distribution that the cluster layout was optimized for.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from chronos.model.moe_chronos import ChronosMOEFeedForward
from chronos.model.temporal_loss import (
    temporal_locality_loss,
    lookahead_supervision_loss,
)


def collect_router_probs(model) -> Optional[torch.Tensor]:
    """Stack `last_router_probs` from every ChronosMOEFeedForward layer.

    Returns: ``[B, S, L, E]`` or ``None`` if no MoE layers fired in the
    last forward pass (e.g. before the first forward).
    """
    probs: List[torch.Tensor] = []
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, ChronosMOEFeedForward):
            p = mlp.last_router_probs
            if p is not None:
                probs.append(p)
    if not probs:
        return None
    return torch.stack(probs, dim=2)


def chronos_loss_term(
    model,
    base_loss: torch.Tensor,
    lookahead_probs: Optional[torch.Tensor],
    config,
    *,
    aux_loss: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Combine `base_loss` with the Chronos-specific regularizers:

        L = L_base
            + λ_balance  · L_aux            (if aux_loss provided)
            + λ_temporal · L_temporal       (if seq has ≥2 tokens)
            + λ_lookahead · L_lookahead     (if lookahead_probs + λ_la > 0)

    The function is safe to call from any stage: it gracefully degrades
    when the necessary tensors are unavailable (e.g. lookahead_probs is
    None during a model that doesn't expose it).
    """
    loss = base_loss

    if aux_loss is not None:
        lambda_balance = float(getattr(config, "lambda_balance", 0.0))
        if lambda_balance > 0.0:
            loss = loss + lambda_balance * aux_loss

    router_4d = collect_router_probs(model)
    if router_4d is not None and router_4d.shape[1] > 1:
        router_mean = router_4d.mean(dim=2)  # [B, S, E]
        lambda_temporal = float(getattr(config, "lambda_temporal", 0.0))
        if lambda_temporal > 0.0:
            loss = loss + lambda_temporal * temporal_locality_loss(router_mean)

        lambda_lookahead = float(getattr(config, "lambda_lookahead", 0.0))
        lookahead_steps = int(getattr(config, "lookahead_steps", 0))
        if (
            lookahead_probs is not None
            and lambda_lookahead > 0.0
            and lookahead_steps > 0
        ):
            teacher = router_mean.detach()
            la = lookahead_supervision_loss(lookahead_probs, teacher, lookahead_steps)
            loss = loss + lambda_lookahead * la
    return loss


def router_kl_anchor(
    current_router_probs: torch.Tensor,
    ref_router_probs: torch.Tensor,
    lambda_anchor: float,
) -> torch.Tensor:
    """KL(current || reference) over routing distributions, scaled by λ.

    Both inputs are layer-averaged ``[B, S, E]`` distributions (post-softmax).
    Reference is treated as a fixed teacher (no grad needed; caller passes
    a detached tensor). Reduces to mean over ``B*S``.
    """
    if lambda_anchor <= 0.0:
        return current_router_probs.new_zeros(())
    eps = 1e-9
    kl = (current_router_probs * (
        current_router_probs.clamp_min(eps).log()
        - ref_router_probs.clamp_min(eps).log()
    )).sum(dim=-1).mean()
    return lambda_anchor * kl


def capture_reference_routing(model, calibration_batch, device) -> torch.Tensor:
    """Run `model` in eval mode on the calibration batch, return the
    layer-averaged routing distribution as a detached reference for the
    `router_kl_anchor` term.

    The calibration batch is a single ``input_ids`` tensor (already on
    device-or-not is handled here). Returns ``[B, S, E]`` detached.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        ids = calibration_batch.to(device)
        try:
            _ = model(ids, use_cache=False)
        except TypeError:
            _ = model(ids)  # minimind models don't accept use_cache kwarg here
        router_4d = collect_router_probs(model)
        if router_4d is None:
            ref = None
        else:
            ref = router_4d.mean(dim=2).detach()
    if was_training:
        model.train()
    return ref
