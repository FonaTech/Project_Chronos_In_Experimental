"""
tools/compare_minimind_chronos_v2.py

Full-stack comparison: pretrain → SFT → DPO → ORPO → GRPO for Chronos;
pretrain → SFT → DPO → GRPO for minimind (ORPO skipped — not in minimind).
Plus a dedicated M3 micro-benchmark under simulated SSD latency.

Reports per stage: final loss, decode tokens/sec, resident expert bytes,
expert activation fraction, and (Chronos-only) lookahead loss + router KL
drift from pretrain reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import chronos.deps  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MOEFeedForward  # noqa: E402
from dataset.lm_dataset import PretrainDataset, SFTDataset, DPODataset  # noqa: E402

from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402
from chronos.model.moe_chronos import ChronosMOEFeedForward  # noqa: E402
from chronos.trainer.loss_mixin import (  # noqa: E402
    chronos_loss_term,
    collect_router_probs,
    capture_reference_routing,
)
from chronos.trainer.dpo_trainer import dpo_loss, _logits_to_log_probs  # noqa: E402
from chronos.trainer.orpo_trainer import _mean_logprob  # noqa: E402
from chronos.trainer.grpo_trainer import default_reward, load_grpo_prompts  # noqa: E402


# ── small harness helpers ─────────────────────────────────────────

@dataclass
class StageResult:
    stage: str
    final_loss: float
    extra: Dict[str, float]


def _expert_bytes_minimind(model) -> int:
    total = 0
    for l in model.model.layers:
        if isinstance(l.mlp, MOEFeedForward):
            for e in l.mlp.experts:
                total += sum(p.numel() * p.element_size() for p in e.parameters())
    return total


def _expert_bytes_chronos(model) -> int:
    """Bytes that WOULD be resident if every expert was held (pre-cache view)."""
    total = 0
    for l in model.model.layers:
        if isinstance(l.mlp, ChronosMOEFeedForward):
            for e in l.mlp.experts:
                total += sum(p.numel() * p.element_size() for p in e.parameters())
    return total


def _chronos_vram_experts_bytes(model, cfg, num_vram_slots: int) -> int:
    """Under a VRAM budget that holds `num_vram_slots` experts, this is the
    actual resident bytes per-layer × num_vram_slots (plus shared)."""
    moe = [l.mlp for l in model.model.layers if isinstance(l.mlp, ChronosMOEFeedForward)]
    if not moe:
        return 0
    per_expert = sum(p.numel() * p.element_size() for p in moe[0].experts[0].parameters())
    return per_expert * len(moe) * num_vram_slots


def measure_decode_tps(model, prompt_ids, n_tokens, device) -> float:
    """Naive decode loop: feed the growing sequence on every step. Slower
    than KV-cache but works identically across model classes."""
    model.eval()
    with torch.no_grad():
        # warmup
        _ = model(prompt_ids.to(device))
        t0 = time.time()
        x = prompt_ids.to(device)
        for _ in range(n_tokens):
            out = model(x)
            outputs = out[0] if isinstance(out, tuple) else out
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_id], dim=-1)
    elapsed = time.time() - t0
    return n_tokens / max(elapsed, 1e-9)


def measure_activation_fraction_chronos(model, prompt_ids, steps, device):
    """Naive growing-sequence loop (same as measure_decode_tps)."""
    moes = [(i, l.mlp) for i, l in enumerate(model.model.layers)
            if isinstance(l.mlp, ChronosMOEFeedForward)]
    seen = set()
    model.eval()
    with torch.no_grad():
        x = prompt_ids.to(device)
        for _ in range(steps):
            out = model(x)
            outputs = out[0] if isinstance(out, tuple) else out
            for li, m in moes:
                if m.last_router_probs is not None:
                    for e in m.last_router_probs[:, -1, :].argmax(dim=-1).unique().tolist():
                        seen.add((li, int(e)))
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_id], dim=-1)
    total = len(moes) * (moes[0][1].num_experts if moes else 1)
    return len(seen) / max(total, 1)


def measure_activation_fraction_minimind(model, prompt_ids, steps, device):
    moes = [(i, l.mlp) for i, l in enumerate(model.model.layers)
            if isinstance(l.mlp, MOEFeedForward)]
    seen = set()
    handles = []
    def make_hook(li):
        def hook(mod, inputs, output):
            x = inputs[0]
            B, S, H = x.shape
            x_flat = x.reshape(-1, H)
            scores = F.softmax(mod.gate(x_flat), dim=-1)
            for e in scores.argmax(dim=-1).unique().tolist():
                seen.add((li, int(e)))
        return hook
    for li, m in moes:
        handles.append(m.register_forward_hook(make_hook(li)))
    model.eval()
    with torch.no_grad():
        x = prompt_ids.to(device)
        for _ in range(steps):
            out = model(x)
            x = torch.cat([x, out.logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=-1)
    for h in handles:
        h.remove()
    total = len(moes) * (moes[0][1].config.num_experts if moes else 1)
    return len(seen) / max(total, 1)


# ── stage runners ─────────────────────────────────────────────────

def _build_pretrain_loader(tokenizer, path, bs, sl):
    ds = PretrainDataset(path, tokenizer, max_length=sl)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


def _build_sft_loader(tokenizer, path, bs, sl):
    ds = SFTDataset(path, tokenizer, max_length=sl)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


def _build_dpo_loader(tokenizer, path, bs, sl):
    ds = DPODataset(path, tokenizer, max_length=sl)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)


def train_minimind_pretrain(model, loader, steps, lr, device):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    last = float("nan")
    s = 0
    while s < steps:
        for ids, labels in loader:
            if s >= steps:
                break
            ids, labels = ids.to(device), labels.to(device)
            res = model(ids, labels=labels)
            loss = res.loss + (res.aux_loss if res.aux_loss is not None else 0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(res.loss.item())
            s += 1
    return last


def train_chronos_pretrain(model, cfg, loader, steps, lr, device):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    last = float("nan")
    s = 0
    while s < steps:
        for ids, labels in loader:
            if s >= steps:
                break
            ids, labels = ids.to(device), labels.to(device)
            outputs, lookahead_probs = model(ids, labels=labels)
            loss = chronos_loss_term(
                model, outputs.loss, lookahead_probs, cfg,
                aux_loss=outputs.aux_loss,
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(outputs.loss.item())
            s += 1
    return last


def train_minimind_sft(model, loader, steps, lr, device):
    # Same as pretrain loop — SFT dataset provides loss-masked labels.
    return train_minimind_pretrain(model, loader, steps, lr, device)


def train_chronos_sft(model, cfg, loader, steps, lr, device, router_ref=None):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    last = float("nan")
    s = 0
    lam_anchor = float(getattr(cfg, "lambda_router_anchor", 0.0))
    while s < steps:
        for ids, labels in loader:
            if s >= steps:
                break
            ids, labels = ids.to(device), labels.to(device)
            outputs, lookahead_probs = model(ids, labels=labels)
            loss = chronos_loss_term(
                model, outputs.loss, lookahead_probs, cfg,
                aux_loss=outputs.aux_loss,
            )
            if lam_anchor > 0 and router_ref is not None:
                from chronos.trainer.loss_mixin import router_kl_anchor
                r4 = collect_router_probs(model)
                if r4 is not None and r4.shape[1] > 1:
                    cur = r4.mean(dim=2)
                    T = min(cur.shape[1], router_ref.shape[1])
                    B = min(cur.shape[0], router_ref.shape[0])
                    loss = loss + router_kl_anchor(
                        cur[:B, :T, :], router_ref[:B, :T, :], lam_anchor,
                    )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(outputs.loss.item())
            s += 1
    return last


def _frozen_clone(model):
    """Clone a model for use as a frozen reference (DPO ref). deepcopy
    fails on some autograd-tracked buffers, so we instantiate a fresh
    instance via __class__ and load the state_dict."""
    cfg = model.config
    new = model.__class__(cfg).to(next(model.parameters()).device)
    new.load_state_dict(model.state_dict(), strict=False)
    return new.eval().requires_grad_(False)


def train_minimind_dpo(model, loader, steps, lr, device, beta=0.1):
    ref = _frozen_clone(model)
    opt = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = float("nan")
    s = 0
    while s < steps:
        for batch in loader:
            if s >= steps:
                break
            x_c = batch["x_chosen"].to(device); x_r = batch["x_rejected"].to(device)
            y_c = batch["y_chosen"].to(device); y_r = batch["y_rejected"].to(device)
            m_c = batch["mask_chosen"].to(device).float()
            m_r = batch["mask_rejected"].to(device).float()
            x = torch.cat([x_c, x_r]); y = torch.cat([y_c, y_r])
            m = torch.cat([m_c, m_r])
            with torch.no_grad():
                ref_lp = _logits_to_log_probs(ref(x).logits, y)
            pol_lp = _logits_to_log_probs(model(x).logits, y)
            loss = dpo_loss(ref_lp, pol_lp, m, beta)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(loss.item())
            s += 1
    return last


def train_chronos_dpo(model, cfg, loader, steps, lr, device, beta=0.1, router_ref=None):
    ref = _frozen_clone(model)
    opt = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = float("nan")
    s = 0
    lam_anchor = float(getattr(cfg, "lambda_router_anchor", 0.0))
    while s < steps:
        for batch in loader:
            if s >= steps:
                break
            x_c = batch["x_chosen"].to(device); x_r = batch["x_rejected"].to(device)
            y_c = batch["y_chosen"].to(device); y_r = batch["y_rejected"].to(device)
            m_c = batch["mask_chosen"].to(device).float()
            m_r = batch["mask_rejected"].to(device).float()
            x = torch.cat([x_c, x_r]); y = torch.cat([y_c, y_r])
            m = torch.cat([m_c, m_r])
            with torch.no_grad():
                ref_out = ref(x)
                ref_logits = ref_out[0].logits if isinstance(ref_out, tuple) else ref_out.logits
                ref_lp = _logits_to_log_probs(ref_logits, y)
            outputs, lookahead_probs = model(x)
            pol_lp = _logits_to_log_probs(outputs.logits, y)
            base = dpo_loss(ref_lp, pol_lp, m, beta)
            loss = chronos_loss_term(
                model, base, lookahead_probs, cfg, aux_loss=outputs.aux_loss,
            )
            if lam_anchor > 0 and router_ref is not None:
                from chronos.trainer.loss_mixin import router_kl_anchor
                r4 = collect_router_probs(model)
                if r4 is not None and r4.shape[1] > 1:
                    cur = r4.mean(dim=2)
                    T = min(cur.shape[1], router_ref.shape[1])
                    B = min(cur.shape[0], router_ref.shape[0])
                    loss = loss + router_kl_anchor(
                        cur[:B, :T, :], router_ref[:B, :T, :], lam_anchor,
                    )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(base.item())
            s += 1
    return last


def train_chronos_orpo(model, cfg, loader, steps, lr, device, beta=0.1, lam=0.1):
    opt = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = float("nan")
    s = 0
    while s < steps:
        for batch in loader:
            if s >= steps:
                break
            x_c = batch["x_chosen"].to(device); x_r = batch["x_rejected"].to(device)
            y_c = batch["y_chosen"].to(device); y_r = batch["y_rejected"].to(device)
            m_c = batch["mask_chosen"].to(device).float()
            m_r = batch["mask_rejected"].to(device).float()
            x = torch.cat([x_c, x_r]); B2 = x_c.shape[0]
            outputs, lookahead_probs = model(x)
            lp_c = _mean_logprob(outputs.logits[:B2], y_c, m_c)
            lp_r = _mean_logprob(outputs.logits[B2:], y_r, m_r)
            def _log_odds(lp):
                om = (1.0 - lp.exp()).clamp_min(1e-6)
                return lp - om.log()
            l_or = -F.logsigmoid(beta * (_log_odds(lp_c) - _log_odds(lp_r))).mean()
            base = -lp_c.mean() + lam * l_or
            loss = chronos_loss_term(
                model, base, lookahead_probs, cfg, aux_loss=outputs.aux_loss,
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last = float(l_or.item())
            s += 1
    return last


# ── main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_steps", type=int, default=100)
    p.add_argument("--align_steps", type=int, default=30)
    p.add_argument("--simulated_ssd_ms", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=96)
    p.add_argument("--hidden_size", type=int, default=192)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--lr_pretrain", type=float, default=5e-4)
    p.add_argument("--lr_align", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str,
                   default=os.path.join(_REPO, "results/compare_results_v2.json"))
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--pretrain_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_pretrain.jsonl"))
    p.add_argument("--sft_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_sft.jsonl"))
    p.add_argument("--dpo_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_dpo.jsonl"))
    p.add_argument("--grpo_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_grpo.jsonl"))
    args = p.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())

    print("\n== compare_minimind_chronos_v2 ==")
    print(f"   pretrain_steps={args.pretrain_steps} align_steps={args.align_steps}")
    print(f"   H={args.hidden_size} L={args.num_hidden_layers} E={args.num_experts}")
    print(f"   simulated_ssd_ms={args.simulated_ssd_ms} device={args.device}\n")

    # Build both models
    mm_cfg = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=args.max_seq_len + 64,
        use_moe=True, num_experts=args.num_experts, num_experts_per_tok=1,
        flash_attn=False,
    )
    ch_cfg = ChronosConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=args.max_seq_len + 64,
        use_moe=True, num_experts=args.num_experts, num_experts_per_tok=1,
        flash_attn=False, lambda_router_anchor=0.0,  # set per stage
    )

    mm = MiniMindForCausalLM(mm_cfg).to(args.device)
    ch = ChronosForCausalLM(ch_cfg).to(args.device)

    results: Dict[str, Dict] = {}

    # Stage 1: pretrain
    print("[Stage 1] Pretrain")
    loader = _build_pretrain_loader(tokenizer, args.pretrain_path, args.batch_size, args.max_seq_len)
    mm_last = train_minimind_pretrain(mm, loader, args.pretrain_steps, args.lr_pretrain, args.device)
    torch.manual_seed(args.seed)
    ch_cfg.lambda_router_anchor = 0.0
    ch_last = train_chronos_pretrain(ch, ch_cfg, loader, args.pretrain_steps, args.lr_pretrain, args.device)
    print(f"  minimind ce={mm_last:.4f}  chronos ce={ch_last:.4f}")
    results["pretrain"] = {"minimind_ce": mm_last, "chronos_ce": ch_last}

    # Capture router reference after pretrain for later drift measurement
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, 16))
    calib_ids = next(iter(loader))[0][:2].to(args.device)
    pretrain_ref = capture_reference_routing(ch, calib_ids, args.device)

    # Stage 2: SFT
    print("[Stage 2] SFT")
    sft_loader = _build_sft_loader(tokenizer, args.sft_path, args.batch_size, args.max_seq_len)
    mm_last = train_minimind_sft(mm, sft_loader, args.align_steps, args.lr_align, args.device)
    ch_cfg.lambda_router_anchor = 0.01
    ch_ref_for_sft = capture_reference_routing(ch, calib_ids, args.device)
    ch_last = train_chronos_sft(ch, ch_cfg, sft_loader, args.align_steps, args.lr_align, args.device,
                                router_ref=ch_ref_for_sft)
    print(f"  minimind ce={mm_last:.4f}  chronos ce={ch_last:.4f}")
    results["sft"] = {"minimind_ce": mm_last, "chronos_ce": ch_last}

    # Stage 3: DPO
    print("[Stage 3] DPO")
    dpo_loader = _build_dpo_loader(tokenizer, args.dpo_path, args.batch_size, args.max_seq_len)
    mm_last = train_minimind_dpo(mm, dpo_loader, args.align_steps, args.lr_align * 0.1, args.device)
    ch_cfg.lambda_router_anchor = 0.1
    ch_ref_for_dpo = capture_reference_routing(ch, calib_ids, args.device)
    ch_last = train_chronos_dpo(ch, ch_cfg, dpo_loader, args.align_steps, args.lr_align * 0.1, args.device,
                                router_ref=ch_ref_for_dpo)
    print(f"  minimind dpo={mm_last:.4f}  chronos dpo={ch_last:.4f}")
    results["dpo"] = {"minimind_dpo": mm_last, "chronos_dpo": ch_last}

    # Measure cache-retention: router KL drift from pretrain reference
    with torch.no_grad():
        ch.eval()
        _ = ch(calib_ids, use_cache=False)
        r4 = collect_router_probs(ch)
        cur = r4.mean(dim=2) if r4 is not None else None
        if cur is not None and pretrain_ref is not None:
            T = min(cur.shape[1], pretrain_ref.shape[1])
            B = min(cur.shape[0], pretrain_ref.shape[0])
            kl = (cur[:B, :T, :] * (
                cur[:B, :T, :].clamp_min(1e-9).log()
                - pretrain_ref[:B, :T, :].clamp_min(1e-9).log()
            )).sum(dim=-1).mean().item()
            results["dpo"]["router_kl_drift_from_pretrain"] = float(kl)
        ch.train()

    # Stage 4: ORPO (Chronos only)
    print("[Stage 4] ORPO (chronos only)")
    ch_last = train_chronos_orpo(ch, ch_cfg, dpo_loader, args.align_steps, args.lr_align * 0.1, args.device)
    print(f"  chronos orpo={ch_last:.4f}")
    results["orpo"] = {"chronos_or": ch_last}

    # Stage 5: GRPO (skip minimind baseline for simplicity; Chronos only)
    print("[Stage 5] GRPO (chronos only)")
    from chronos.trainer.grpo_trainer import ChronosGRPOTrainer

    class _A:
        pass
    a = _A()
    a.device = args.device
    a.dtype = "float16"
    a.max_seq_len = args.max_seq_len
    a.max_gen_len = 16
    a.num_generations = 2
    a.temperature = 1.0
    a.learning_rate = args.lr_align * 0.1
    a.accumulation_steps = 1
    a.grad_clip = 1.0
    a.log_interval = 10
    a.epochs = 1
    a.save_dir = tempfile.mkdtemp()
    a.beta = 0.04
    ch_cfg.lambda_router_anchor = 0.1
    grpo_trainer = ChronosGRPOTrainer(ch, ch_cfg, a, tokenizer)
    prompts = load_grpo_prompts(args.grpo_path, max_prompts=args.align_steps)
    grpo_trainer.train_epoch(0, prompts, len(prompts), max_steps=args.align_steps)
    # Evaluate mean reward on a few final prompts
    grpo_trainer.model.eval()
    final_rewards = []
    for pt in prompts[:5]:
        ids = tokenizer(pt, return_tensors="pt", truncation=True,
                        max_length=args.max_seq_len).input_ids
        _, gen = grpo_trainer._rollout(ids)
        final_rewards.append(default_reward(pt, gen, tokenizer))
    grpo_trainer.model.train()
    results["grpo"] = {"chronos_mean_reward_post": sum(final_rewards) / len(final_rewards)}

    # ── Final measurements ──
    print("\n[Measurements] decode tokens/sec, activation fraction, resident expert bytes")
    ch.eval(); mm.eval()
    tps_mm = measure_decode_tps(mm, prompt_ids, args.max_new_tokens, args.device)
    tps_ch = measure_decode_tps(ch, prompt_ids, args.max_new_tokens, args.device)
    frac_mm = measure_activation_fraction_minimind(mm, prompt_ids, 32, args.device)
    frac_ch = measure_activation_fraction_chronos(ch, prompt_ids, 32, args.device)
    # Chronos "resident" = VRAM slots ≈ 2 experts/layer (1 shared + 1 topk)
    bytes_mm = _expert_bytes_minimind(mm)
    bytes_ch = _chronos_vram_experts_bytes(ch, ch_cfg, num_vram_slots=2)
    results["measurements"] = {
        "tps_minimind": tps_mm, "tps_chronos": tps_ch,
        "activation_frac_minimind": frac_mm, "activation_frac_chronos": frac_ch,
        "resident_expert_bytes_minimind": bytes_mm,
        "resident_expert_bytes_chronos_cached": bytes_ch,
    }

    # ── M3 micro-benchmark ──
    print("\n[M3 micro-benchmark] prefetch vs legacy step() under simulated SSD")
    from chronos.runtime.cache_manager import CacheManager
    ssd = tempfile.mkdtemp(prefix="chronos_v2_m3_")
    mgr = CacheManager(ch, ch_cfg, ssd_dir=ssd)
    mgr.expert_store.offload_all_to_ssd(
        clusters=[[i] for i in range(ch_cfg.num_experts)]  # force one-per-cluster
    )
    mgr.expert_store.attach_cluster_manifest()
    mgr.warm_up(initial_expert_ids=[0])
    mgr.start()

    E = ch_cfg.num_experts
    lp_prefetch = torch.zeros(1, 1, ch_cfg.lookahead_steps + 1, E)
    lp_prefetch[..., 0] = 1.0 / E
    lp_prefetch[0, 0, 1, E - 1] = 1.0

    os.environ["CHRONOS_SIM_SSD_MS"] = str(args.simulated_ssd_ms)
    try:
        t0 = time.monotonic()
        mgr.prefetch_for_next_step(lp_prefetch)
        async_ms = (time.monotonic() - t0) * 1000

        time.sleep(max(0.1, args.simulated_ssd_ms / 1000 + 0.05))

        # legacy step() on a fresh expert (must block)
        lp2 = torch.zeros(1, 1, ch_cfg.lookahead_steps + 1, E)
        lp2[..., 0] = 1.0 / E
        fresh = (E // 2) if E >= 3 else 1
        t0 = time.monotonic()
        mgr.step(lp2, [fresh])
        legacy_ms = (time.monotonic() - t0) * 1000
    finally:
        os.environ.pop("CHRONOS_SIM_SSD_MS", None)
        mgr.stop()

    results["m3_overlap_benchmark"] = {
        "simulated_ssd_ms": args.simulated_ssd_ms,
        "prefetch_for_next_step_ms": async_ms,
        "legacy_step_ms": legacy_ms,
        "overlap_headroom_ms": legacy_ms - async_ms,
    }

    # ── Save + print ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("STAGE RESULTS")
    print("-" * 78)
    for stage, r in results.items():
        if stage in ("measurements", "m3_overlap_benchmark"):
            continue
        line = f"  [{stage:<8}] " + "  ".join(f"{k}={v:.4f}" for k, v in r.items()
                                               if isinstance(v, (float, int)))
        print(line)
    print("\nFINAL DECODE MEASUREMENTS")
    print("-" * 78)
    m = results["measurements"]
    print(f"  tokens/sec           minimind={m['tps_minimind']:.1f}  chronos={m['tps_chronos']:.1f}")
    print(f"  activation fraction  minimind={m['activation_frac_minimind']:.3f}  chronos={m['activation_frac_chronos']:.3f}")
    print(f"  resident expert MB   minimind={m['resident_expert_bytes_minimind']/1e6:.2f}  chronos(cached)={m['resident_expert_bytes_chronos_cached']/1e6:.2f}")
    print("\nM3 OVERLAP BENCHMARK")
    print("-" * 78)
    m3 = results["m3_overlap_benchmark"]
    print(f"  simulated SSD latency:         {m3['simulated_ssd_ms']}ms")
    print(f"  prefetch_for_next_step:        {m3['prefetch_for_next_step_ms']:.2f}ms  (non-blocking)")
    print(f"  legacy step() with miss:       {m3['legacy_step_ms']:.2f}ms     (blocking)")
    print(f"  pipeline headroom per token:   {m3['overlap_headroom_ms']:.2f}ms")
    print("=" * 78)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
