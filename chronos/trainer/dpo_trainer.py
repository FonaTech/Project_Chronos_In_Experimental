"""
chronos/trainer/dpo_trainer.py — Stage 3 DPO trainer.

Implements DPO with a frozen reference Chronos model + Chronos loss mixin
(lookahead/temporal/balance preserved) + strong router KL anchor (default
λ=0.1) so the alignment gradients don't blow up the cluster cache layout.

Mirrors minimind's train_dpo.py loss but uses the Chronos forward signature
``(outputs, lookahead_probs) = model(x)`` and tolerates its extra return.
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext
from copy import deepcopy

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


def _logits_to_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def dpo_loss(ref_lp: torch.Tensor, pol_lp: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    ref_lp = (ref_lp * mask).sum(dim=1)
    pol_lp = (pol_lp * mask).sum(dim=1)
    B = ref_lp.shape[0]
    ch_ref, rj_ref = ref_lp[:B // 2], ref_lp[B // 2:]
    ch_pol, rj_pol = pol_lp[:B // 2], pol_lp[B // 2:]
    pi_logr = ch_pol - rj_pol
    rf_logr = ch_ref - rj_ref
    return -F.logsigmoid(beta * (pi_logr - rf_logr)).mean()


class ChronosDPOTrainer:
    def __init__(self, model: ChronosForCausalLM, config: ChronosConfig, args, tokenizer):
        self.model = model
        self.config = config
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.beta = float(getattr(args, "beta", 0.1))

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.autocast_ctx = nullcontext() if device_type == "cpu" \
            else torch.cuda.amp.autocast(dtype=dtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
        from chronos.trainer.optim_utils import build_optimizer
        self.optimizer = build_optimizer(
            model, lr=args.learning_rate,
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
        )

        # Frozen reference model — DPO needs π_ref(y|x). Build a fresh
        # instance and load the current state_dict (deepcopy fails on
        # autograd-tracked buffers in some torch versions).
        self.ref_model = model.__class__(config).to(self.device)
        self.ref_model.load_state_dict(model.state_dict(), strict=False)
        self.ref_model.eval().requires_grad_(False)

        self._router_ref = None

    def set_calibration_batch(self, x_calib: torch.Tensor):
        self._router_ref = capture_reference_routing(self.model, x_calib, self.device)

    def train_step(self, batch, step, total_steps):
        x_c = batch["x_chosen"].to(self.device)
        x_r = batch["x_rejected"].to(self.device)
        y_c = batch["y_chosen"].to(self.device)
        y_r = batch["y_rejected"].to(self.device)
        m_c = batch["mask_chosen"].to(self.device).float()
        m_r = batch["mask_rejected"].to(self.device).float()
        x = torch.cat([x_c, x_r], dim=0)
        y = torch.cat([y_c, y_r], dim=0)
        m = torch.cat([m_c, m_r], dim=0)

        lr = get_lr(step, total_steps, self.args.learning_rate)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        with self.autocast_ctx:
            with torch.no_grad():
                ref_out = self.ref_model(x)
                ref_logits = ref_out[0].logits if isinstance(ref_out, tuple) else ref_out.logits
            ref_lp = _logits_to_log_probs(ref_logits, y)

            outputs, lookahead_probs = self.model(x)
            pol_lp = _logits_to_log_probs(outputs.logits, y)
            base = dpo_loss(ref_lp, pol_lp, m, beta=self.beta)

            # Mix in Chronos regularizers
            loss = chronos_loss_term(
                self.model, base, lookahead_probs, self.config,
                aux_loss=outputs.aux_loss,
            )

            # Router KL anchor (DPO default: strong)
            anc_val = 0.0
            lam = float(getattr(self.config, "lambda_router_anchor", 0.0))
            if lam > 0.0 and self._router_ref is not None:
                router_4d = collect_router_probs(self.model)
                if router_4d is not None and router_4d.shape[1] > 1:
                    cur = router_4d.mean(dim=2)
                    ref = self._router_ref
                    T = min(cur.shape[1], ref.shape[1])
                    B = min(cur.shape[0], ref.shape[0])
                    anc = router_kl_anchor(cur[:B, :T, :], ref[:B, :T, :], lam)
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
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return (
            loss.item() * self.args.accumulation_steps,
            float(base.item()),
            float(outputs.aux_loss.item()) if outputs.aux_loss is not None else 0.0,
            la_val,
            anc_val,
        )

    def train_epoch(self, epoch, loader, iters, max_steps=None):
        self.model.train()
        start_time = time.time()
        total_steps = self.args.epochs * iters
        for step, batch in enumerate(loader, start=1):
            if max_steps is not None and step > max_steps:
                break
            loss, dpo_v, aux, la, anc = self.train_step(
                batch, epoch * iters + step, total_steps,
            )
            if step % self.args.log_interval == 0 or step == iters:
                lr = self.optimizer.param_groups[-1]["lr"]
                Logger(
                    f"[DPO] Epoch[{epoch+1}/{self.args.epochs}]({step}/{iters}) "
                    f"loss:{loss:.4f} dpo:{dpo_v:.4f} aux:{aux:.4f} "
                    f"la:{la:.4f} anchor:{anc:.4f} lr:{lr:.2e}"
                )

    def _save(self, epoch, step):
        os.makedirs(self.args.save_dir, exist_ok=True)
        ckp = os.path.join(self.args.save_dir, f"dpo_{self.config.hidden_size}_moe.pth")
        self.model.eval()
        state = self.model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state.items()}, ckp)
        self.model.train()


def build_dpo_loader(data_path, tokenizer, max_seq_len, batch_size):
    ds = DPODataset(data_path, tokenizer, max_length=max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
