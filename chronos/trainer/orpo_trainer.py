"""
chronos/trainer/orpo_trainer.py — Stage 4 ORPO trainer.

ORPO (Hong et al. 2024) combines supervised NLL on the chosen response
with an odds-ratio penalty on the rejected response, in a single objective
that does NOT require a separate reference model. This matters for
memory-constrained Chronos deployments: DPO needs two full models in
memory; ORPO needs one.

Loss:
    L_NLL  = −mean_token log p_θ(y_w | x)       # supervised on chosen
    L_OR   = −log σ( β · log [ odds(y_w) / odds(y_l) ] )
             where odds(y) = p_θ(y) / (1 - p_θ(y))
    L_ORPO = L_NLL + λ · L_OR

Plus Chronos mixin (temporal / lookahead / aux) and router KL anchor.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import chronos.deps  # noqa
from trainer.trainer_utils import Logger, is_main_process  # type: ignore
from chronos.data.flexible_dataset import StreamingDPODataset as DPODataset

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.temporal_loss import lookahead_supervision_loss
from chronos.trainer.optim_utils import get_lr
from chronos.trainer.loss_mixin import (
    chronos_loss_term,
    collect_router_probs,
    router_kl_anchor,
    capture_reference_routing,
)
from chronos.model.checkpoint import save_state_dict_with_config
from chronos.trainer.device_utils import (
    autocast_context,
    configure_cpu_threads,
    dataloader_kwargs,
    grad_scaler,
    optimizer_step_with_scaler,
    runtime_summary,
)


def _mean_logprob(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean log-prob of `labels` under `logits`, averaged over masked positions."""
    log_probs = F.log_softmax(logits, dim=-1)
    lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    lp = (lp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return lp


class ChronosORPOTrainer:
    def __init__(self, model: ChronosForCausalLM, config: ChronosConfig, args, tokenizer):
        self.model = model
        self.config = config
        self.args = args
        self.device = args.device
        self.beta = float(getattr(args, "beta", 0.1))
        self.lambda_or = float(getattr(args, "lambda_or", 0.1))

        configure_cpu_threads(
            getattr(args, "cpu_threads", None),
            budget_percent=getattr(args, "cpu_budget_percent", 100),
        )
        self.autocast_ctx = autocast_context(self.device, getattr(args, "dtype", "float32"))
        self.scaler = grad_scaler(self.device, getattr(args, "dtype", "float32"))
        summary = runtime_summary(self.device, getattr(args, "dtype", "float32"))
        Logger(
            "Runtime: "
            f"device={summary.device} type={summary.device_type} dtype={summary.dtype} "
            f"cpu_threads={summary.cpu_threads} autocast={summary.autocast} scaler={summary.scaler}"
        )
        from chronos.trainer.optim_utils import build_optimizer
        self.optimizer = build_optimizer(
            model, lr=args.learning_rate,
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
        )
        self._router_ref = None

    def set_calibration_batch(self, x: torch.Tensor):
        self._router_ref = capture_reference_routing(self.model, x, self.device)

    def train_step(self, batch, step, total_steps):
        x_c = batch["x_chosen"].to(self.device)
        x_r = batch["x_rejected"].to(self.device)
        y_c = batch["y_chosen"].to(self.device)
        y_r = batch["y_rejected"].to(self.device)
        m_c = batch["mask_chosen"].to(self.device).float()
        m_r = batch["mask_rejected"].to(self.device).float()
        x = torch.cat([x_c, x_r], dim=0)
        y = torch.cat([y_c, y_r], dim=0)

        lr = get_lr(step, total_steps, self.args.learning_rate)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        with self.autocast_ctx:
            outputs, lookahead_probs = self.model(x)
            B2 = x_c.shape[0]
            logits_c = outputs.logits[:B2]
            logits_r = outputs.logits[B2:]

            lp_c = _mean_logprob(logits_c, y_c, m_c)   # [B]
            lp_r = _mean_logprob(logits_r, y_r, m_r)

            # NLL on chosen
            nll = -lp_c.mean()

            # Odds-ratio term (numerically stable via log-space)
            # log odds(y) = lp - log(1 - exp(lp)). Use log(1 - exp(lp)) ≈
            # log1p(-exp(lp)) with clamping to avoid NaN at lp ≈ 0.
            def _log_odds(lp):
                one_minus = (1.0 - lp.exp()).clamp_min(1e-6)
                return lp - one_minus.log()
            log_odds_ratio = _log_odds(lp_c) - _log_odds(lp_r)
            l_or = -F.logsigmoid(self.beta * log_odds_ratio).mean()

            base = nll + self.lambda_or * l_or

            loss = chronos_loss_term(
                self.model, base, lookahead_probs, self.config,
                aux_loss=outputs.aux_loss,
            )

            anc_val = 0.0
            lam = float(getattr(self.config, "lambda_router_anchor", 0.0))
            if lam > 0.0 and self._router_ref is not None:
                r4 = collect_router_probs(self.model)
                if r4 is not None and r4.shape[1] > 1:
                    cur = r4.mean(dim=2)
                    ref = self._router_ref
                    T = min(cur.shape[1], ref.shape[1])
                    Bm = min(cur.shape[0], ref.shape[0])
                    anc = router_kl_anchor(cur[:Bm, :T, :], ref[:Bm, :T, :], lam)
                    loss = loss + anc
                    anc_val = float(anc.item())

            la_val = 0.0
            r4 = collect_router_probs(self.model)
            if (r4 is not None and r4.shape[1] > 1
                    and lookahead_probs is not None
                    and self.config.lookahead_steps > 0):
                la_val = float(lookahead_supervision_loss(
                    lookahead_probs, r4.mean(dim=2).detach(),
                    self.config.lookahead_steps,
                ).item())

            loss = loss / self.args.accumulation_steps

        self.scaler.scale(loss).backward()

        if step % self.args.accumulation_steps == 0:
            optimizer_step_with_scaler(
                self.scaler,
                self.optimizer,
                self.model.parameters(),
                self.args.grad_clip,
            )

        return (
            loss.item() * self.args.accumulation_steps,
            float(nll.item()),
            float(l_or.item()),
            la_val,
            anc_val,
        )

    def train_epoch(self, epoch, loader, iters, max_steps=None):
        self.model.train()
        total_steps = self.args.epochs * iters
        for step, batch in enumerate(loader, start=1):
            if max_steps is not None and step > max_steps:
                break
            loss, nll, lor, la, anc = self.train_step(
                batch, epoch * iters + step, total_steps,
            )
            if step % self.args.log_interval == 0 or step == iters:
                lr = self.optimizer.param_groups[-1]["lr"]
                Logger(
                    f"[ORPO] Epoch[{epoch+1}/{self.args.epochs}]({step}/{iters}) "
                    f"loss:{loss:.4f} nll:{nll:.4f} or:{lor:.4f} "
                    f"la:{la:.4f} anchor:{anc:.4f} lr:{lr:.2e}"
                )

    def _save(self, epoch, step):
        os.makedirs(self.args.save_dir, exist_ok=True)
        ckp = os.path.join(self.args.save_dir, f"orpo_{self.config.hidden_size}_moe.pth")
        self.model.eval()
        save_state_dict_with_config(self.model, ckp, self.config, stage="orpo")
        self.model.train()


def build_orpo_loader(data_path, tokenizer, max_seq_len, batch_size, device="cpu", num_workers="auto"):
    ds = DPODataset(data_path, tokenizer, max_length=max_seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        **dataloader_kwargs(device=device, num_workers=num_workers, shuffle=True),
    )
