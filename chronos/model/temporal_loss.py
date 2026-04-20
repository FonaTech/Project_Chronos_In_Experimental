import torch
import torch.nn.functional as F


def temporal_locality_loss(expert_probs_seq: torch.Tensor) -> torch.Tensor:
    """
    Penalizes large routing distribution shifts between adjacent time steps.

    L_temporal = (1/T) * sum_{t=2}^{T} ||E_t - E_{t-1}||_2^2

    Args:
        expert_probs_seq: [B, T, num_experts]  per-step routing probabilities
    Returns:
        scalar loss
    """
    diff = expert_probs_seq[:, 1:, :] - expert_probs_seq[:, :-1, :]
    return (diff ** 2).sum(dim=-1).mean()


def load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    """
    Standard auxiliary load-balancing loss (same formulation as minimind).

    Args:
        router_logits: [N, num_experts]  (N = B*S flattened)
    Returns:
        scalar loss
    """
    scores = F.softmax(router_logits, dim=-1)
    _, topk_idx = torch.topk(scores, k=num_experts_per_tok, dim=-1, sorted=False)
    load = F.one_hot(topk_idx, num_experts).float().mean(0)
    return (load * scores.mean(0)).sum() * num_experts


def lookahead_supervision_loss(
    lookahead_probs: torch.Tensor,
    teacher_router_probs: torch.Tensor,
    lookahead_steps: int,
) -> torch.Tensor:
    """
    Teacher-forcing loss for the LookaheadRouter.

    At time t the router predicts the routing distribution at t+k for
    k ∈ {1..lookahead_steps}. We supervise those predictions with the
    *actual* layer-averaged router probabilities from step t+k (stop-grad).

    Formulation (soft-target cross-entropy, equivalent to forward KL
    up to a teacher entropy constant):

        L = (1 / K) · Σ_{k=1..K} [ - mean_{t} Σ_e P_{t+k}[e] · log Q_t^(k)[e] ]

    Args:
        lookahead_probs:       [B, S, K+1, E] — softmax'd output of
                                LookaheadRouter. Index 0 = current step,
                                indices 1..K = future-step predictions.
        teacher_router_probs:  [B, S, E]       — layer-averaged real router
                                probabilities. Caller MUST pass a detached
                                (stop-grad) tensor.
        lookahead_steps:       K, how many future steps to supervise.

    Returns:
        scalar loss. Zero when the sequence is too short to form a pair.
    """
    if lookahead_probs is None or teacher_router_probs is None:
        return lookahead_probs.new_zeros(()) if lookahead_probs is not None else \
               teacher_router_probs.new_zeros(())

    B, S, Kp1, E = lookahead_probs.shape
    K = min(lookahead_steps, Kp1 - 1)
    if K <= 0 or S <= 1:
        return lookahead_probs.new_zeros(())

    teacher = teacher_router_probs  # caller is responsible for .detach()
    total = lookahead_probs.new_zeros(())
    valid_terms = 0
    for k in range(1, K + 1):
        if S - k <= 0:
            continue
        teacher_k = teacher[:, k:, :]                  # [B, S-k, E]
        pred_k = lookahead_probs[:, :-k, k, :]         # [B, S-k, E]
        # Soft-target CE: −Σ_e P · log Q
        log_pred = pred_k.clamp_min(1e-9).log()
        ce = -(teacher_k * log_pred).sum(dim=-1)        # [B, S-k]
        total = total + ce.mean()
        valid_terms += 1
    if valid_terms == 0:
        return lookahead_probs.new_zeros(())
    return total / valid_terms


def total_loss(
    ce_loss: torch.Tensor,
    balance_loss: torch.Tensor,
    expert_probs_seq: torch.Tensor,
    lambda1: float,
    lambda2: float,
    *,
    lookahead_probs: torch.Tensor = None,
    teacher_probs: torch.Tensor = None,
    lookahead_steps: int = 0,
    lambda_lookahead: float = 0.0,
) -> torch.Tensor:
    """
    L_total = L_CE + λ1·L_balance + λ2·L_temporal + λ_lookahead·L_lookahead

    The lookahead term is only applied when `lookahead_probs`, `teacher_probs`
    and `lambda_lookahead > 0` are all provided — preserves the M1-era
    signature for legacy callers.
    """
    loss = ce_loss + lambda1 * balance_loss + lambda2 * temporal_locality_loss(expert_probs_seq)
    if (
        lookahead_probs is not None
        and teacher_probs is not None
        and lambda_lookahead > 0.0
        and lookahead_steps > 0
    ):
        loss = loss + lambda_lookahead * lookahead_supervision_loss(
            lookahead_probs, teacher_probs, lookahead_steps,
        )
    return loss
