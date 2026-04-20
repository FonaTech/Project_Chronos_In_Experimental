import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ChronosConfig


class LookaheadRouter(nn.Module):
    """
    Global lookahead routing predictor inserted after the first transformer block.

    Input:  hidden_states [B, S, H]  (output of block 0)
    Output: expert_probs  [B, S, lookahead_steps+1, num_experts]
              dim-2 index 0 = current step t
              dim-2 index 1 = predicted t+1
              dim-2 index 2 = predicted t+2  (if lookahead_steps >= 2)

    During training: output drives temporal_locality_loss and lookahead accuracy.
    During inference: output drives async prefetch for future expert weights.
    """

    def __init__(self, config: ChronosConfig):
        super().__init__()
        self.lookahead_steps = config.lookahead_steps
        self.num_experts = config.num_experts
        H = config.hidden_size
        K = config.lookahead_steps + 1  # current + future steps

        # Lightweight 2-layer MLP: H -> H//4 -> E*K
        self.proj = nn.Sequential(
            nn.Linear(H, H // 4, bias=False),
            nn.SiLU(),
            nn.Linear(H // 4, self.num_experts * K, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, S, H]
        B, S, H = hidden_states.shape
        K = self.lookahead_steps + 1
        logits = self.proj(hidden_states)                    # [B, S, E*K]
        logits = logits.view(B, S, K, self.num_experts)      # [B, S, K, E]
        return F.softmax(logits, dim=-1)                     # [B, S, K, E]

    def predict_next_experts(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 1,
    ) -> torch.Tensor:
        """
        Inference helper: returns top-k expert indices for each future step.
        Returns: [B, S, lookahead_steps, top_k]
        """
        probs = self.forward(hidden_states)          # [B, S, K, E]
        future = probs[:, :, 1:, :]                  # [B, S, lookahead_steps, E]
        return torch.topk(future, k=top_k, dim=-1).indices
