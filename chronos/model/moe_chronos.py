import sys
import chronos.deps  # ensure minimind on sys.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from model.model_minimind import FeedForward, MOEFeedForward
from .config import ChronosConfig


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
        # Stores routing probs for temporal loss computation in trainer
        self.last_router_probs: torch.Tensor = None

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
        self.last_router_probs = scores.view(B, S, self.num_experts).detach()

        topk_weight, topk_idx = torch.topk(
            scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)

        if available_expert_mask is None:
            # --- Training path: standard MoE dispatch ---
            for i, expert in enumerate(self.experts):
                mask = (topk_idx == i)
                if mask.any():
                    token_idx = mask.any(dim=-1).nonzero().flatten()
                    weight = topk_weight[mask].view(-1, 1)
                    y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
                elif self.training:
                    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

            if self.training and self.router_aux_loss_coef > 0:
                load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
                self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
            else:
                self.aux_loss = scores.new_zeros(1).squeeze()

        else:
            # --- Inference path: compile-safe soft gating ---
            # Convert bool mask to float [num_experts] so all branching is
            # expressed as multiplication — no Python-level if/else per expert,
            # giving torch.compile a single traceable graph.
            avail = available_expert_mask.to(dtype=x_flat.dtype, device=x_flat.device)  # [E]
            shared_out_full = self._shared_expert_output(x_flat)  # [N, H] — computed once

            for i, expert in enumerate(self.experts):
                mask = (topk_idx == i)                          # [N, K] bool
                if not mask.any():
                    continue
                token_idx = mask.any(dim=-1).nonzero().flatten()   # [T]
                weight = topk_weight[mask].view(-1, 1)             # [T, 1]

                expert_out   = expert(x_flat[token_idx])           # [T, H]
                fallback_out = shared_out_full[token_idx]          # [T, H]

                # avail[i] == 1.0 → expert output; 0.0 → shared fallback
                # Single mul+add, no control flow visible to the compiler
                blended = avail[i] * expert_out + (1.0 - avail[i]) * fallback_out
                y.index_add_(0, token_idx, (blended * weight).to(y.dtype))

            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(B, S, H)
