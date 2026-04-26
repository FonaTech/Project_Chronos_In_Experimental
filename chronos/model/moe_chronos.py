import sys
import chronos.deps  # ensure minimind on sys.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from model.model_minimind import FeedForward, MOEFeedForward
from .config import ChronosConfig


class LazyExpertPlaceholder(nn.Module):
    """Inference-only placeholder for experts that live on SSD/RAM.

    If this module executes directly, Chronos failed to materialize the
    expert before use or failed to route the token through the shared
    expert fallback. That is a correctness bug, so we raise loudly.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.loaded = False

    def forward(self, _x):
        raise RuntimeError("lazy expert placeholder executed before materialization")


class ChronosMOEFeedForward(nn.Module):
    """
    Extends MOEFeedForward with:
    1. shared_experts: always resident in VRAM, used as soft fallback
    2. soft_gate: when target experts are unavailable (cache miss), blend
       their routing weight into shared experts instead of blocking on I/O

    Training (available_expert_mask=None): identical to MOEFeedForward,
    plus stores per-token routing probs for temporal_locality_loss.

    Inference (available_expert_mask provided): routes cache-miss tokens
    to shared experts with zero latency.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_aux_loss_coef = config.router_aux_loss_coef

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.shared_experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_shared_experts)
        ])
        self.aux_loss = torch.zeros(1)
        # raw, *unscaled* load-balance value computed in forward(). The
        # trainer-side λ_balance lives in chronos_loss_term so we don't
        # double-scale (router_aux_loss_coef × λ_balance gave a 2 000×
        # under-weighting and let the gate degenerate). Set to None until
        # the first training forward populates it.
        self.aux_loss_raw: torch.Tensor = None
        # Stores routing probs for downstream loss / metrics.
        # We keep TWO refs:
        #   last_router_probs       — detached, safe for logging / cluster
        #                              layout / inference engine prefetch
        #                              (consumers that must NOT change grad).
        #   last_router_probs_grad  — live tensor with autograd attached,
        #                              consumed by chronos_loss_term so the
        #                              λ_temporal / λ_lookahead penalties
        #                              actually shape the gate.
        self.last_router_probs: torch.Tensor = None
        self.last_router_probs_grad: torch.Tensor = None
        self.fallback_mask_prob = float(getattr(config, "fallback_mask_prob", 0.0))
        self.runtime_miss_policy = "fallback_diagnostic"
        self.runtime_on_demand_loader = None
        self.runtime_touch_expert = None
        self.last_on_demand_experts: list[int] = []
        self.last_fallback_experts: list[int] = []
        self.last_on_demand_weight_mass = 0.0
        self.last_fallback_weight_mass = 0.0

    def _shared_expert_output(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Average output of all shared experts."""
        out = sum(e(x_flat) for e in self.shared_experts)
        return out / len(self.shared_experts)

    def forward(
        self,
        x: torch.Tensor,
        available_expert_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, S, H]
            available_expert_mask: [num_experts] bool or float — which experts
                are in VRAM. None during training (all experts available).

        When available_expert_mask is provided the forward pass is written
        as pure tensor arithmetic (no Python if/else per expert) so that
        torch.compile can trace a single static graph without graph breaks.
        """
        B, S, H = x.shape
        x_flat = x.view(-1, H)                                   # [N, H]
        scores = F.softmax(self.gate(x_flat), dim=-1)            # [N, E]
        scores_bse = scores.view(B, S, self.num_experts)
        self.last_on_demand_experts = []
        self.last_fallback_experts = []
        self.last_on_demand_weight_mass = 0.0
        self.last_fallback_weight_mass = 0.0
        # Live (with-grad) view drives the temporal / lookahead losses.
        self.last_router_probs_grad = scores_bse
        # Detached view for read-only consumers (logging, cluster layout,
        # inference-engine prefetch). Same tensor data, no autograd edge.
        self.last_router_probs = scores_bse.detach()

        topk_weight, topk_idx = torch.topk(
            scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)

        training_fallback_mask = None
        if (
            available_expert_mask is None
            and self.training
            and self.fallback_mask_prob > 0.0
            and len(self.shared_experts) > 0
        ):
            # Simulate SSD/RAM cache misses during training so the shared
            # fallback sees the same distribution it must cover at inference.
            keep = torch.rand(self.num_experts, device=x_flat.device) >= self.fallback_mask_prob
            if not torch.any(keep):
                keep[torch.randint(0, self.num_experts, (1,), device=x_flat.device)] = True
            training_fallback_mask = keep

        if available_expert_mask is None and training_fallback_mask is None:
            # --- Standard path: routed experts + always-on shared residual ---
            dead_param_sum = None
            for i, expert in enumerate(self.experts):
                mask = (topk_idx == i)
                if mask.any():
                    token_idx = mask.any(dim=-1).nonzero().flatten()
                    weight = topk_weight[mask].view(-1, 1)
                    y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
                elif self.training:
                    # Dead-expert handling: keep the autograd edge alive
                    # WITHOUT touching y[0,0] (the previous trick poisoned
                    # the BOS position every step). We accumulate a
                    # 0·sum(params) scalar that gets routed into aux_loss
                    # below, which the trainer always backprops on.
                    contrib = sum(p.sum() for p in expert.parameters())
                    dead_param_sum = (
                        contrib if dead_param_sum is None else dead_param_sum + contrib
                    )

            # Always-on shared expert (mirrors the inference branch). The
            # previous code only ran shared_experts under the inference
            # mask path, so during training they received zero gradient
            # — and at inference, the cache-miss fallback fired untrained
            # weights, which is the dominant root cause of "loss looks
            # fine but generations are gibberish."
            if len(self.shared_experts) > 0:
                y = y + self._shared_expert_output(x_flat).to(y.dtype)

            if self.training:
                # Raw, un-scaled load balance (model exposes raw; trainer scales).
                load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
                aux_raw = (load * scores.mean(0)).sum() * self.num_experts
                if dead_param_sum is not None:
                    aux_raw = aux_raw + 0.0 * dead_param_sum
                self.aux_loss_raw = aux_raw
                # Backwards-compat: the legacy `aux_loss` field stays
                # populated with the pre-scaled value some callers still
                # read (chronos_loss_term now prefers aux_loss_raw).
                self.aux_loss = aux_raw * self.router_aux_loss_coef
            else:
                self.aux_loss_raw = scores.new_zeros(())
                self.aux_loss = scores.new_zeros(())

        else:
            # --- Masked/offload path: real lazy experts + shared fallback ---
            # We intentionally branch at the Python level here: when an
            # expert is cold we must not execute it at all, because the
            # live module may have been replaced by a placeholder after
            # expert offload. A multiply-by-zero path would still call the
            # unloaded expert and defeat lazy loading entirely.
            if available_expert_mask is None:
                avail = training_fallback_mask.to(dtype=x_flat.dtype, device=x_flat.device)
            else:
                avail = available_expert_mask.to(dtype=x_flat.dtype, device=x_flat.device)  # [E]
            shared_out_full = (
                self._shared_expert_output(x_flat)
                if len(self.shared_experts) > 0 else None
            )
            dead_param_sum = None

            for i, expert in enumerate(self.experts):
                mask = (topk_idx == i)                          # [N, K] bool
                expert_selected = mask.any()
                if not expert_selected:
                    if self.training:
                        contrib = sum(p.sum() for p in expert.parameters())
                        dead_param_sum = (
                            contrib if dead_param_sum is None else dead_param_sum + contrib
                        )
                    continue
                token_idx = mask.any(dim=-1).nonzero().flatten()   # [T]
                weight = topk_weight[mask].view(-1, 1)             # [T, 1]
                selected_mass = float(weight.detach().to(torch.float32).sum().item())
                expert_live = not isinstance(self.experts[i], LazyExpertPlaceholder)
                expert_available = float(avail[i].item()) > 0.0
                inference_stale_mask = (
                    available_expert_mask is not None
                    and not self.training
                    and expert_live
                )
                if expert_live and (expert_available or inference_stale_mask):
                    touch = getattr(self, "runtime_touch_expert", None)
                    if touch is not None and not self.training:
                        touch(int(i))
                    if not expert_available and available_expert_mask is not None and not self.training:
                        self.last_on_demand_experts.append(int(i))
                        self.last_on_demand_weight_mass += selected_mass
                    blended = self.experts[i](x_flat[token_idx])    # [T, H]
                else:
                    loaded = False
                    loader = getattr(self, "runtime_on_demand_loader", None)
                    if (
                        available_expert_mask is not None
                        and not self.training
                        and getattr(self, "runtime_miss_policy", "fallback_diagnostic") in {"on_demand", "sync_on_demand"}
                        and loader is not None
                    ):
                        result = loader(int(i))
                        if isinstance(result, dict):
                            loaded = bool(result.get("ok", False))
                        else:
                            loaded = bool(result)

                    if loaded and not isinstance(self.experts[i], LazyExpertPlaceholder):
                        self.last_on_demand_experts.append(int(i))
                        self.last_on_demand_weight_mass += selected_mass
                        blended = self.experts[i](x_flat[token_idx])
                    else:
                        self.last_fallback_experts.append(int(i))
                        self.last_fallback_weight_mass += selected_mass
                        blended = (
                            shared_out_full[token_idx]
                            if shared_out_full is not None
                            else x_flat.new_zeros((token_idx.numel(), H))
                        )
                y.index_add_(0, token_idx, (blended * weight).to(y.dtype))

            # The shared expert is a residual component, not only the cold
            # fallback. This makes available_mask=all_true equivalent to the
            # normal path and keeps SSD/DRAM inference faithful to training.
            if shared_out_full is not None:
                y = y + shared_out_full.to(y.dtype)

            if self.training:
                load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
                aux_raw = (load * scores.mean(0)).sum() * self.num_experts
                if dead_param_sum is not None:
                    aux_raw = aux_raw + 0.0 * dead_param_sum
                self.aux_loss_raw = aux_raw
                self.aux_loss = aux_raw * self.router_aux_loss_coef
            else:
                self.aux_loss_raw = scores.new_zeros(())
                self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(B, S, H)
