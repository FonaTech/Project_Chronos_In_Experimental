"""
chronos/trainer/sft_trainer.py

Chronos-aware supervised fine-tuning trainer.

Built around minimind's SFTDataset (jsonl with `conversations` field,
tokenizer.apply_chat_template + per-assistant-token loss mask) so the data
pipeline is identical to the one in minimind-master/trainer/train_full_sft.py.
What we add:

1. The model returns ``(outputs, lookahead_probs)``, which minimind's trainer
   does not know about. We forward it explicitly.
2. ``chronos_loss_term`` mixes in λ_balance·aux + λ_temporal·temporal +
   λ_lookahead·lookahead on top of the base CE.
3. Optional ``router_kl_anchor`` pulls the current router distribution back
   toward the pretrain reference (default λ=0.01 for SFT — mild drift is
   OK). The reference is captured once at stage start.
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext

import torch
from torch import optim
from torch.utils.data import DataLoader

import chronos.deps  # ensure minimind on sys.path

from trainer.trainer_utils import Logger, is_main_process  # type: ignore
from chronos.data.flexible_dataset import StreamingSFTDataset

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


class ChronosSFTTrainer:
    """Stage 2 (SFT) trainer for Chronos."""

    def __init__(self, model: ChronosForCausalLM, config: ChronosConfig, args, tokenizer):
        self.model = model
        self.config = config
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.autocast_ctx = (
            nullcontext() if device_type == "cpu"
            else torch.cuda.amp.autocast(dtype=dtype)
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
        from chronos.trainer.optim_utils import build_optimizer
        self.optimizer = build_optimizer(
            model, lr=args.learning_rate,
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
        )

        # Router anchor reference — captured at the start of a stage
        self._router_ref = None
        self._calib_batch = None

    def set_calibration_batch(self, input_ids: torch.Tensor):
        """Provide a representative input_ids batch; at stage start the
        current (pre-SFT) model's router is frozen as the anchor teacher."""
        self._calib_batch = input_ids
        self._router_ref = capture_reference_routing(self.model, input_ids, self.device)

    def train_step(self, input_ids, labels, step, total_steps):
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        lr = get_lr(step, total_steps, self.args.learning_rate)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

        with self.autocast_ctx:
            outputs, lookahead_probs = self.model(input_ids, labels=labels)
            ce_loss = outputs.loss
            aux_loss = outputs.aux_loss

            loss = chronos_loss_term(
                self.model, ce_loss, lookahead_probs, self.config,
                aux_loss=aux_loss,
            )

            # Router KL anchor
            anchor_val = 0.0
            lam = float(getattr(self.config, "lambda_router_anchor", 0.0))
            if lam > 0.0 and self._router_ref is not None:
                router_4d = collect_router_probs(self.model)
                if router_4d is not None and router_4d.shape[1] > 1:
                    cur = router_4d.mean(dim=2)
                    # Shape-align: calibration batch shape may differ; crop
                    # the shorter axis on both sides.
                    ref = self._router_ref
                    T = min(cur.shape[1], ref.shape[1])
                    B = min(cur.shape[0], ref.shape[0])
                    anchor = router_kl_anchor(
                        cur[:B, :T, :], ref[:B, :T, :], lam,
                    )
                    loss = loss + anchor
                    anchor_val = float(anchor.item())

            # Standalone lookahead loss for logging
            la_val = 0.0
            router_4d = collect_router_probs(self.model)
            if (router_4d is not None and router_4d.shape[1] > 1
                    and lookahead_probs is not None
                    and self.config.lookahead_steps > 0):
                teacher = router_4d.mean(dim=2).detach()
                la = lookahead_supervision_loss(
                    lookahead_probs, teacher, self.config.lookahead_steps,
                )
                la_val = float(la.item())

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
            ce_loss.item(),
            float(aux_loss.item()) if aux_loss is not None else 0.0,
            la_val,
            anchor_val,
        )

    def train_epoch(self, epoch, loader, iters, start_step=0, wandb=None, max_steps=None):
        self.model.train()
        start_time = time.time()
        total_steps = self.args.epochs * iters

        for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
            if max_steps is not None and step > max_steps:
                break
            loss, ce, aux, la, anc = self.train_step(
                input_ids, labels, epoch * iters + step, total_steps,
            )
            if step % self.args.log_interval == 0 or step == iters:
                elapsed = time.time() - start_time
                eta = elapsed / max(step - start_step, 1) * (iters - step) // 60
                lr = self.optimizer.param_groups[-1]['lr']
                Logger(
                    f'[SFT] Epoch[{epoch+1}/{self.args.epochs}]({step}/{iters}) '
                    f'loss:{loss:.4f} ce:{ce:.4f} aux:{aux:.4f} '
                    f'la:{la:.4f} anchor:{anc:.4f} lr:{lr:.2e} eta:{eta:.1f}min'
                )
                if wandb:
                    wandb.log({
                        "sft_loss": loss, "sft_ce": ce, "sft_aux": aux,
                        "sft_lookahead": la, "sft_anchor": anc, "lr": lr,
                    })

            if (step % self.args.save_interval == 0 or step == iters) and is_main_process():
                self._save(epoch, step)

    def _save(self, epoch, step):
        os.makedirs(self.args.save_dir, exist_ok=True)
        ckp = os.path.join(
            self.args.save_dir,
            f'sft_{self.config.hidden_size}_moe.pth',
        )
        self.model.eval()
        state = self.model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state.items()}, ckp)
        self.model.train()


def build_sft_loader(data_path, tokenizer, max_seq_len, batch_size):
    ds = StreamingSFTDataset(data_path, tokenizer, max_length=max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
