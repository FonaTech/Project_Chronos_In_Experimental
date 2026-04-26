"""
chronos/trainer/distill_trainer.py — Stage 6 distillation trainer.

Loss:
    L_KD    = T² · KL( softmax(student_logits/T) || softmax(teacher_logits/T) )
    L_label = cross_entropy(student_logits, labels, ignore_index=-100)
    L_total = α · L_KD + (1 − α) · L_label  +  chronos_loss_term (lookahead etc.)

The teacher is any frozen model that returns ``.logits`` on forward (Chronos
or MiniMind checkpoints both work). Router KL anchor still applies — Stage
6 shouldn't corrupt the cache layout any more than SFT/DPO can.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import chronos.deps  # noqa: F401
from trainer.trainer_utils import Logger  # type: ignore
from chronos.data.flexible_dataset import StreamingSFTDataset as SFTDataset

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


def distillation_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                    T: float = 4.0, ignore_mask: torch.Tensor = None) -> torch.Tensor:
    """Temperature-scaled KL divergence; caller is responsible for masking."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1).detach()
    kl = F.kl_div(s, t, reduction="none").sum(dim=-1)  # [B, S]
    if ignore_mask is not None:
        kl = kl * ignore_mask
        denom = ignore_mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(kl.numel(), dtype=kl.dtype, device=kl.device).clamp_min(1.0)
    return (T * T) * (kl.sum() / denom)


class ChronosDistillTrainer:
    def __init__(self, model: ChronosForCausalLM, teacher, config: ChronosConfig,
                 args, tokenizer):
        self.model = model
        self.teacher = teacher.eval().requires_grad_(False)
        self.config = config
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.alpha = float(getattr(args, "alpha", 0.7))
        self.temperature = float(getattr(args, "temperature", 4.0))

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

    def _teacher_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.teacher(input_ids)
            return (out[0].logits if isinstance(out, tuple) else out.logits).detach()

    def train_step(self, input_ids, labels, step, total_steps):
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        lr = get_lr(step, total_steps, self.args.learning_rate)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        with self.autocast_ctx:
            outputs, lookahead_probs = self.model(input_ids, labels=labels)
            ce_loss = outputs.loss
            student_logits = outputs.logits

            # Teacher forward (no grad)
            teacher_logits = self._teacher_logits(input_ids)
            # Align vocab dims if teacher is smaller (pad or truncate)
            if teacher_logits.shape[-1] != student_logits.shape[-1]:
                V = min(teacher_logits.shape[-1], student_logits.shape[-1])
                teacher_logits = teacher_logits[..., :V]
                student_logits_kd = student_logits[..., :V]
            else:
                student_logits_kd = student_logits
            # Align seq len
            S = min(teacher_logits.shape[1], student_logits_kd.shape[1])
            teacher_logits = teacher_logits[:, :S, :]
            student_logits_kd = student_logits_kd[:, :S, :]

            # Loss mask = positions where label != -100
            ignore_mask = (labels[:, :S] != -100).float() if labels.shape[1] >= S else None
            kd = distillation_kl(
                student_logits_kd, teacher_logits,
                T=self.temperature, ignore_mask=ignore_mask,
            )

            base = self.alpha * kd + (1.0 - self.alpha) * ce_loss
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
                    T_ax = min(cur.shape[1], ref.shape[1])
                    B = min(cur.shape[0], ref.shape[0])
                    anc = router_kl_anchor(cur[:B, :T_ax, :], ref[:B, :T_ax, :], lam)
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

        return {
            "loss": float(loss.item() * self.args.accumulation_steps),
            "kd": float(kd.item()),
            "ce": float(ce_loss.item()),
            "la": la_val,
            "anchor": anc_val,
        }

    def train_epoch(self, epoch, loader, iters, max_steps=None):
        self.model.train()
        total_steps = self.args.epochs * iters
        for step, (ids, labels) in enumerate(loader, start=1):
            if max_steps is not None and step > max_steps:
                break
            s = self.train_step(ids, labels, epoch * iters + step, total_steps)
            if step % self.args.log_interval == 0 or step == iters:
                lr = self.optimizer.param_groups[-1]["lr"]
                Logger(
                    f"[Distill] Epoch[{epoch+1}/{self.args.epochs}]({step}/{iters}) "
                    f"loss:{s['loss']:.4f} kd:{s['kd']:.4f} ce:{s['ce']:.4f} "
                    f"la:{s['la']:.4f} anchor:{s['anchor']:.4f} lr:{lr:.2e}"
                )

    def _save(self, epoch, step):
        os.makedirs(self.args.save_dir, exist_ok=True)
        ckp = os.path.join(self.args.save_dir, f"distill_{self.config.hidden_size}_moe.pth")
        self.model.eval()
        save_state_dict_with_config(self.model, ckp, self.config, stage="distill")
        self.model.train()


def build_distill_loader(data_path, tokenizer, max_seq_len, batch_size, device="cpu", num_workers="auto"):
    ds = SFTDataset(data_path, tokenizer, max_length=max_seq_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        **dataloader_kwargs(device=device, num_workers=num_workers, shuffle=True),
    )
