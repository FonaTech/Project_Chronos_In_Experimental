"""
chronos/trainer/grpo_trainer.py — Stage 5 GRPO trainer (self-contained).

Group Relative Policy Optimization for Chronos. Unlike minimind's
train_grpo.py which depends on a rollout engine + external reward model
+ sglang, this trainer is self-contained:

  - Rollouts use ``model.generate`` (HF-style) with sampling temperature.
  - Reward is a toy function (length + vocab purity + repetition penalty)
    pluggable via ``reward_fn``.
  - Reference model is a frozen deep-copy of the starting model (same
    idea as our DPO trainer).

Loss:
    For each prompt p, sample G completions {y_i}.
    advantages   = (r_i - mean(r_group)) / (std(r_group) + eps)
    per_token_lp = log π_θ(y_i | p)                         # current policy
    per_token_lp_ref = log π_ref(y_i | p)                   # stop-grad
    kl           = π_ratio^{-1}·exp(lp_ref - lp) - (lp_ref - lp) - 1   (k3)
    L = -E[ advantage · lp_π ] + β · E[kl]
"""
from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import chronos.deps  # noqa
from trainer.trainer_utils import Logger  # type: ignore

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


from chronos.trainer.reward import ToyReward, build_reward_fn  # noqa: F401


# Toy vocabulary mirrors tests/fixtures/generate_tiny_alignment_data.py
TOY_VOCAB = {
    "the", "quick", "brown", "fox", "machine", "learning", "model", "expert",
    "router", "cluster", "memory", "cache", "prefetch", "latency", "throughput",
    "tensor", "training", "inference", "decode", "weight", "gradient",
    "softmax", "attention", "layer", "embedding", "vocab", "sequence",
    "loss", "optimizer", "epoch", "batch", "step", "token", "logit",
    "minimind", "chronos", "hybrid", "mixture", "of", "experts", "sparse",
    "dense", "neural", "network", "transformer", "rope", "rotary", "kv",
}


def default_reward(prompt: str, response_tokens: List[int], tokenizer, target_len: int = 24) -> float:
    """Backward-compat wrapper; prefer chronos.trainer.reward.ToyReward."""
    return ToyReward(target_len=target_len, vocab=TOY_VOCAB)(prompt, response_tokens, tokenizer)


class ChronosGRPOTrainer:
    def __init__(
        self,
        model: ChronosForCausalLM,
        config: ChronosConfig,
        args,
        tokenizer,
        reward_fn: Callable = None,
    ):
        self.model = model
        self.config = config
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        self.beta = float(getattr(args, "beta", 0.04))
        self.num_generations = int(getattr(args, "num_generations", 4))
        self.max_gen_len = int(getattr(args, "max_gen_len", 24))
        self.temperature = float(getattr(args, "temperature", 1.0))
        self.reward_fn = reward_fn or default_reward

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

        self.ref_model = model.__class__(config).to(self.device)
        self.ref_model.load_state_dict(model.state_dict(), strict=False)
        self.ref_model.eval().requires_grad_(False)
        self._router_ref = None

    def set_calibration_batch(self, x: torch.Tensor):
        self._router_ref = capture_reference_routing(self.model, x, self.device)

    # ── rollout ──────────────────────────────────────────────────

    def _rollout(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """Sample a single completion. Returns (full_ids [1, P+L], response_len)."""
        self.model.eval()
        with torch.no_grad():
            ids = prompt_ids.clone().to(self.device)
            past = None
            generated: List[int] = []
            for _ in range(self.max_gen_len):
                past_len = past[0][0].shape[1] if past else 0
                out = self.model(
                    ids[:, past_len:] if past_len > 0 else ids,
                    past_key_values=past, use_cache=True,
                )
                outputs = out[0] if isinstance(out, tuple) else out
                past = outputs.past_key_values
                logits = outputs.logits[:, -1, :] / max(self.temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=-1)
                generated.append(int(next_id.item()))
                if next_id.item() == self.tokenizer.eos_token_id:
                    break
        self.model.train()
        return ids, generated

    # ── core step ────────────────────────────────────────────────

    def train_step(self, prompt_text: str, step: int, total_steps: int):
        # Tokenize prompt
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.args.max_seq_len,
        ).input_ids.to(self.device)

        # Generate G completions, compute rewards
        trajectories: List[Tuple[torch.Tensor, int, float]] = []
        for _ in range(self.num_generations):
            full_ids, gen_tokens = self._rollout(prompt_ids)
            r = self.reward_fn(prompt_text, gen_tokens, self.tokenizer)
            trajectories.append((full_ids, len(gen_tokens), r))

        rewards = torch.tensor([t[2] for t in trajectories], device=self.device)
        adv = (rewards - rewards.mean()) / (rewards.std().clamp_min(1e-6))

        # Build a padded batch of full sequences for the policy update
        max_len = max(t[0].shape[1] for t in trajectories)
        pad_id = self.tokenizer.pad_token_id or 0
        ids_batch = torch.full(
            (len(trajectories), max_len), pad_id, dtype=torch.long, device=self.device,
        )
        response_mask = torch.zeros(
            (len(trajectories), max_len), dtype=torch.float, device=self.device,
        )
        prompt_len = prompt_ids.shape[1]
        for i, (full, glen, _) in enumerate(trajectories):
            L = full.shape[1]
            ids_batch[i, :L] = full[0]
            response_mask[i, prompt_len:L] = 1.0

        lr = get_lr(step, total_steps, self.args.learning_rate)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        with self.autocast_ctx:
            outputs, lookahead_probs = self.model(ids_batch)
            log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            target = ids_batch[:, 1:]
            lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [G, L-1]
            mask = response_mask[:, 1:]

            with torch.no_grad():
                ref_out = self.ref_model(ids_batch)
                ref_outputs = ref_out[0] if isinstance(ref_out, tuple) else ref_out
                ref_lp = F.log_softmax(ref_outputs.logits[:, :-1, :], dim=-1).gather(
                    -1, target.unsqueeze(-1)
                ).squeeze(-1)

            # k3 KL estimator (Schulman 2020)
            log_ratio = ref_lp - lp
            kl_per_tok = (log_ratio.exp() - log_ratio - 1.0) * mask
            kl = kl_per_tok.sum() / mask.sum().clamp_min(1.0)

            # Advantage-weighted policy gradient (per-sequence)
            seq_lp = (lp * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pg_loss = -(adv.detach() * seq_lp).mean()
            base = pg_loss + self.beta * kl

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

        return {
            "loss": float(loss.item() * self.args.accumulation_steps),
            "pg_loss": float(pg_loss.item()),
            "kl": float(kl.item()),
            "mean_reward": float(rewards.mean().item()),
            "la": la_val,
            "anchor": anc_val,
        }

    def train_epoch(self, epoch, prompts: List[str], iters, max_steps=None):
        self.model.train()
        total_steps = self.args.epochs * iters
        for step, prompt in enumerate(prompts, start=1):
            if max_steps is not None and step > max_steps:
                break
            if step > iters:
                break
            stats = self.train_step(prompt, epoch * iters + step, total_steps)
            if step % self.args.log_interval == 0 or step == iters:
                lr = self.optimizer.param_groups[-1]["lr"]
                Logger(
                    f"[GRPO] Epoch[{epoch+1}/{self.args.epochs}]({step}/{iters}) "
                    f"loss:{stats['loss']:.4f} pg:{stats['pg_loss']:.4f} "
                    f"kl:{stats['kl']:.4f} r:{stats['mean_reward']:.3f} "
                    f"la:{stats['la']:.4f} anchor:{stats['anchor']:.4f} lr:{lr:.2e}"
                )

    def _save(self, epoch, step):
        os.makedirs(self.args.save_dir, exist_ok=True)
        ckp = os.path.join(self.args.save_dir, f"grpo_{self.config.hidden_size}_moe.pth")
        self.model.eval()
        save_state_dict_with_config(self.model, ckp, self.config, stage="grpo")
        self.model.train()


def load_grpo_prompts(data_path: str, max_prompts: int = None) -> List[str]:
    import json
    prompts = []
    with open(data_path) as f:
        for line in f:
            row = json.loads(line)
            # Extract first user message
            for msg in row.get("conversations", []):
                if msg.get("role") == "user":
                    prompts.append(msg["content"])
                    break
            if max_prompts and len(prompts) >= max_prompts:
                break
    return prompts
