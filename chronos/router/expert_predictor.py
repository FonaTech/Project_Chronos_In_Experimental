"""
chronos/router/expert_predictor.py

Layer 2 of the routing pipeline: converts IntentVector probabilities into
a concrete expert set to pre-load before generation begins.

Responsibilities:
  1. Threshold-based selection: keep experts with prob > threshold
  2. Budget enforcement: cap at vram_capacity experts
  3. Per-layer masks for soft gating (avail_masks)
  4. Confidence reporting: if max_prob < min_confidence, fall back to
     top-K heuristic (always load the K most common experts)

The ExpertPredictor is stateless — it takes an IntentVector and returns
an ExpertSet. No weights, no training needed.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Set

import torch


@dataclass
class ExpertSet:
    """
    The concrete set of experts to pre-load for a generation session.

    expert_ids: flat set of expert IDs to load (union across all layers)
    per_layer_masks: List[Tensor([num_experts], bool)] — per-layer avail
    confidence: float — how confident the classifier was (0–1)
    fallback: bool — True if we fell back to heuristic selection
    """
    expert_ids: Set[int]
    per_layer_masks: List[torch.Tensor]
    confidence: float
    fallback: bool = False

    def __repr__(self):
        return (f"ExpertSet(experts={sorted(self.expert_ids)}, "
                f"confidence={self.confidence:.2f}, fallback={self.fallback})")


class ExpertPredictor:
    """
    Converts IntentVector → ExpertSet.

    Args:
        num_experts: total number of experts
        num_moe_layers: number of MoE layers
        vram_capacity: max experts that fit in VRAM simultaneously
        threshold: minimum probability to include an expert
        min_confidence: if best prob < this, use fallback heuristic
        fallback_top_k: number of experts to always include as fallback
    """

    def __init__(
        self,
        num_experts: int,
        num_moe_layers: int,
        vram_capacity: int = 4,
        threshold: float = 0.25,
        min_confidence: float = 0.15,
        fallback_top_k: int = 2,
    ):
        self.num_experts = num_experts
        self.num_moe_layers = num_moe_layers
        self.vram_capacity = vram_capacity
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.fallback_top_k = fallback_top_k

        # Running frequency table: updated from actual activations over time
        self._frequency: torch.Tensor = torch.ones(num_experts) / num_experts

    def predict(self, intent: "IntentVector") -> ExpertSet:
        """
        Convert IntentVector to a concrete ExpertSet.

        Strategy:
          1. Check confidence (max prob across all layers)
          2. If confident: threshold + budget cap
          3. If not confident: fallback to global frequency heuristic
        """
        from chronos.router.intent_classifier import IntentVector

        global_p = intent.global_probs  # [E]
        confidence = float(global_p.max().item())
        fallback = confidence < self.min_confidence

        if fallback:
            # Heuristic: always load the historically most-activated experts
            top_ids = self._frequency.topk(
                min(self.fallback_top_k, self.vram_capacity)
            ).indices.tolist()
            expert_ids = set(top_ids)
        else:
            # Per-layer selection
            selected_per_layer = []
            all_ids: Set[int] = set()
            for layer_p in intent.per_layer_probs:
                above = (layer_p >= self.threshold).nonzero().flatten().tolist()
                selected_per_layer.append(set(above))
                all_ids.update(above)

            # Enforce budget: keep highest-prob experts if over capacity
            if len(all_ids) > self.vram_capacity:
                top_ids = global_p.topk(self.vram_capacity).indices.tolist()
                all_ids = set(top_ids)

            expert_ids = all_ids

        # Build per-layer boolean masks
        per_layer_masks = []
        for li in range(self.num_moe_layers):
            mask = torch.zeros(self.num_experts, dtype=torch.bool)
            layer_p = intent.per_layer_probs[li] if li < len(intent.per_layer_probs) else global_p
            if fallback:
                for eid in expert_ids:
                    mask[eid] = True
            else:
                for eid in expert_ids:
                    if layer_p[eid] >= self.threshold or eid in expert_ids:
                        mask[eid] = True
            per_layer_masks.append(mask)

        return ExpertSet(
            expert_ids=expert_ids,
            per_layer_masks=per_layer_masks,
            confidence=confidence,
            fallback=fallback,
        )

    def update_frequency(self, actual_expert_ids: List[int]):
        """
        Update running activation frequency from observed generation.
        Called after each generation session with the experts that were
        actually activated.
        """
        alpha = 0.05  # EMA smoothing
        observed = torch.zeros(self.num_experts)
        for eid in actual_expert_ids:
            if 0 <= eid < self.num_experts:
                observed[eid] += 1
        if observed.sum() > 0:
            observed = observed / observed.sum()
        self._frequency = (1 - alpha) * self._frequency + alpha * observed

    def avail_masks_float(self, expert_set: ExpertSet, device=None) -> List[torch.Tensor]:
        """
        Convert per_layer_masks (bool) to float tensors for soft gating.
        0.0 = cache miss → shared expert fallback
        1.0 = in VRAM → use directly
        """
        result = []
        for mask in expert_set.per_layer_masks:
            f = mask.float()
            if device is not None:
                f = f.to(device)
            result.append(f)
        return result
