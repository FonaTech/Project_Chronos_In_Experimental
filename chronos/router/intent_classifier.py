"""
chronos/router/intent_classifier.py

Layer 1 of the dual-layer architecture: a small dense model that reads
the full prompt during prefill and outputs an "intent vector" —
a distribution over expert-activation patterns.

Design:
  - 2-layer Transformer encoder (~8-15M params, much smaller than main MoE)
  - Input: tokenised prompt (first min(512, seq_len) tokens)
  - Output: IntentVector — a [num_experts] float tensor (per layer) that
    predicts which experts the main MoE will need during generation

Why separate from LookaheadRouter:
  LookaheadRouter predicts 1-2 steps ahead within the generation loop.
  IntentClassifier reads the ENTIRE prompt ONCE before generation starts,
  producing a global expert-set prediction that covers the whole reply.
  This moves all IO uncertainty to the prefill phase.

Training:
  Supervised on (prompt, expert_activation_log) pairs collected from
  the main model's forward passes. Loss = MSE between predicted and
  actual per-layer expert activation frequencies.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IntentVector:
    """
    Output of IntentClassifier for a single prompt.

    per_layer_probs: List[Tensor([num_experts])] — one per MoE layer.
        Values are probabilities; higher means more likely to be needed.
    global_probs: Tensor([num_experts]) — averaged across layers.
    """
    per_layer_probs: List[torch.Tensor]
    global_probs: torch.Tensor


class IntentClassifier(nn.Module):
    """
    Small dense encoder that maps a prompt to an expert activation prediction.

    Architecture: embedding → 2 Transformer encoder layers → mean-pool →
                  linear head → sigmoid per-layer expert probabilities

    Parameters: ~10-15M (hidden=256, heads=4, layers=2)
    """

    def __init__(
        self,
        vocab_size: int,
        num_experts: int,
        num_moe_layers: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_moe_layers = num_moe_layers
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(hidden_size)

        # Per-layer expert probability heads
        # Output: [num_moe_layers, num_experts] per sample
        self.head = nn.Linear(hidden_size, num_moe_layers * num_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> IntentVector:
        """
        Args:
            input_ids: [B, S] token ids (truncated to max_seq_len)
        Returns:
            IntentVector with per-layer and global expert probabilities
        """
        B, S = input_ids.shape
        S = min(S, self.max_seq_len)
        ids = input_ids[:, :S]

        pos = torch.arange(S, device=ids.device).unsqueeze(0)  # [1, S]
        x = self.embed(ids) + self.pos_embed(pos)               # [B, S, H]
        x = self.dropout(x)

        # Padding mask for attention (0 = pad token)
        pad_mask = (ids == 0)  # [B, S]

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Mean-pool over non-padding tokens
        lengths = (~pad_mask).float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        x_masked = x * (~pad_mask).float().unsqueeze(-1)
        pooled = x_masked.sum(dim=1) / lengths                 # [B, H]

        logits = self.head(pooled)                              # [B, L*E]
        probs = torch.sigmoid(logits).view(B, self.num_moe_layers, self.num_experts)

        # Build IntentVector (for batch size 1 during inference)
        per_layer = [probs[0, i] for i in range(self.num_moe_layers)]
        global_p = probs[0].mean(dim=0)
        return IntentVector(per_layer_probs=per_layer, global_probs=global_p)

    # ── Training helpers ─────────────────────────────────────────

    @staticmethod
    def collect_activation_targets(
        model,
        input_ids: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """
        Run main model forward pass and collect actual expert activation
        frequencies as training labels for IntentClassifier.

        Returns: [num_moe_layers, num_experts] float tensor (frequencies)
        """
        from chronos.model.moe_chronos import ChronosMOEFeedForward
        model.eval()
        with torch.no_grad():
            model(input_ids.to(device))

        moe_layers = [l.mlp for l in model.model.layers
                      if isinstance(l.mlp, ChronosMOEFeedForward)]
        targets = []
        for moe in moe_layers:
            if moe.last_router_probs is not None:
                # Mean activation probability across tokens
                targets.append(moe.last_router_probs.mean(dim=(0, 1)))  # [E]
            else:
                targets.append(torch.zeros(model.config.num_experts))
        return torch.stack(targets)  # [L, E]

    def train_step(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """
        Single training step.
        targets: [B, num_moe_layers, num_experts] float
        """
        self.train()
        optimizer.zero_grad(set_to_none=True)
        B, S = input_ids.shape
        logits = self.head(
            self.norm(self.encoder(
                self.dropout(self.embed(input_ids[:, :self.max_seq_len])
                             + self.pos_embed(
                                 torch.arange(min(S, self.max_seq_len),
                                              device=input_ids.device).unsqueeze(0)))
            ).mean(dim=1))
        ).view(B, self.num_moe_layers, self.num_experts)
        probs = torch.sigmoid(logits)
        loss = F.mse_loss(probs, targets.to(probs.device))
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "IntentClassifier":
        m = cls(**kwargs)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m
