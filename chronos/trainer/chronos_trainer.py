import sys
import chronos.deps  # ensure minimind on sys.path

import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, setup_seed,
)
from chronos.trainer.optim_utils import get_lr  # warmup→cosine override
from chronos.data.flexible_dataset import (
    FlexibleDataset as PretrainDataset,
    StreamingSFTDataset as SFTDataset,
)

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.temporal_loss import lookahead_supervision_loss
from chronos.trainer.loss_mixin import chronos_loss_term, collect_router_probs


class ChronosTrainer:
    """
    Training orchestrator for Project Chronos.

    Wraps minimind's training loop and injects:
    - temporal_locality_loss via λ2 * ||E_t - E_{t-1}||^2
    - load_balance_loss via λ1 (already computed inside ChronosMOEFeedForward)
    - LookaheadRouter accuracy logging
    """

    def __init__(self, config: ChronosConfig, args):
        self.config = config
        self.args = args
        self.device = args.device
        seed = getattr(args, "seed", None)
        if seed is not None:
            setup_seed(int(seed))
        device_type = "cuda" if "cuda" in self.device else "cpu"
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        self.autocast_ctx = (
            nullcontext() if device_type == "cpu"
            else torch.cuda.amp.autocast(dtype=dtype)
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

        self.model = ChronosForCausalLM(config).to(self.device)
        from chronos.trainer.optim_utils import build_optimizer
        self.optimizer = build_optimizer(
            self.model, lr=args.learning_rate,
            weight_decay=float(getattr(args, "weight_decay", 0.01)),
        )

    def _collect_router_probs(self) -> torch.Tensor:
        """Gather last_router_probs from all MoE layers → [B, S, L, E]."""
        return collect_router_probs(self.model)

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

            # Standalone lookahead loss value for logging
            lookahead_loss_val = 0.0
            router_4d = self._collect_router_probs()
            if (
                router_4d is not None
                and router_4d.shape[1] > 1
                and lookahead_probs is not None
                and self.config.lookahead_steps > 0
            ):
                teacher = router_4d.mean(dim=2).detach()
                la_term = lookahead_supervision_loss(
                    lookahead_probs, teacher, self.config.lookahead_steps,
                )
                lookahead_loss_val = float(la_term.item())

            loss = loss / self.args.accumulation_steps

        self.scaler.scale(loss).backward()

        if step % self.args.accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.item() * self.args.accumulation_steps, ce_loss.item(), aux_loss.item(), lookahead_loss_val

    def train_epoch(self, epoch, loader, iters, start_step=0, wandb=None):
        self.model.train()
        start_time = time.time()
        total_steps = self.args.epochs * iters

        for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
            loss, ce_loss, aux_loss, la_loss = self.train_step(
                input_ids, labels, epoch * iters + step, total_steps
            )

            if step % self.args.log_interval == 0 or step == iters:
                elapsed = time.time() - start_time
                eta = elapsed / max(step - start_step, 1) * (iters - step) // 60
                lr = self.optimizer.param_groups[-1]['lr']
                Logger(
                    f'Epoch:[{epoch+1}/{self.args.epochs}]({step}/{iters}) '
                    f'loss:{loss:.4f} ce:{ce_loss:.4f} aux:{aux_loss:.4f} '
                    f'la:{la_loss:.4f} lr:{lr:.2e} eta:{eta:.1f}min'
                )
                if wandb:
                    wandb.log({
                        "loss": loss, "ce_loss": ce_loss, "aux_loss": aux_loss,
                        "lookahead_loss": la_loss, "lr": lr,
                    })

            if (step % self.args.save_interval == 0 or step == iters) and is_main_process():
                self._save(epoch, step, wandb)

    def _save(self, epoch, step, wandb=None):
        self.model.eval()
        os.makedirs(self.args.save_dir, exist_ok=True)
        ckp = f'{self.args.save_dir}/chronos_{self.config.hidden_size}_moe.pth'
        state = self.model.state_dict()
        torch.save({k: v.half().cpu() for k, v in state.items()}, ckp)
        lm_checkpoint(
            self.config, weight='chronos', model=self.model,
            optimizer=self.optimizer, scaler=self.scaler,
            epoch=epoch, step=step, wandb=wandb,
            save_dir=self.args.save_dir,
        )
        self.model.train()
        del state
