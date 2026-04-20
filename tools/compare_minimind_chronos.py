"""
tools/compare_minimind_chronos.py

Short-step head-to-head comparison between vanilla minimind MoE and Project
Chronos (M1 + M2). Both models are trained on the same deterministic tiny
corpus with matched hyperparameters. After training, we measure:

  1. resident_expert_bytes  — VRAM/RAM bytes occupied by *expert* weights.
                              For Chronos this is the sum of expert-store
                              VRAM-resident experts only; for minimind it is
                              every MoE expert (always resident).
  2. expert_activation_fraction — unique (layer, expert) pairs touched per
                              100 generated tokens, divided by L·E. Lower
                              means cache locality is working.
  3. tokens_per_sec (decode) — wall-clock speed of greedy decode.
  4. lookahead_t+1_top1     — Chronos only; sanity-check that M2 supervision
                              is doing something.
  5. final_ce_loss          — sanity that Chronos is in the same ballpark.

Run:
    python tools/compare_minimind_chronos.py --steps 200 --device cpu \
        --output results/compare_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make the repo importable when run from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import chronos.deps  # noqa: E402  -- ensures minimind on sys.path

from transformers import AutoTokenizer  # noqa: E402

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MOEFeedForward  # noqa: E402
from dataset.lm_dataset import PretrainDataset  # noqa: E402

from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402
from chronos.model.moe_chronos import ChronosMOEFeedForward  # noqa: E402


# ── instrumentation ───────────────────────────────────────────────

@dataclass
class ActivationLog:
    pairs: set  # set[(layer, expert)]
    num_layers: int
    num_experts: int

    def fraction(self) -> float:
        return len(self.pairs) / max(self.num_layers * self.num_experts, 1)


def _instrument_minimind(model) -> ActivationLog:
    """Hook every MOEFeedForward to record top-1 expert per token."""
    moe_layers = [(i, l.mlp) for i, l in enumerate(model.model.layers)
                  if isinstance(l.mlp, MOEFeedForward)]
    log = ActivationLog(
        pairs=set(),
        num_layers=len(moe_layers),
        num_experts=moe_layers[0][1].config.num_experts if moe_layers else 0,
    )

    def make_hook(layer_idx):
        def hook(mod, inputs, output):
            x = inputs[0]
            B, S, H = x.shape
            x_flat = x.reshape(-1, H)
            scores = F.softmax(mod.gate(x_flat), dim=-1)
            top1 = scores.argmax(dim=-1)  # [N]
            for e in top1.unique().tolist():
                log.pairs.add((layer_idx, int(e)))
        return hook

    for i, mlp in moe_layers:
        mlp.register_forward_hook(make_hook(i))
    return log


def _instrument_chronos(model) -> ActivationLog:
    moe_layers = [(i, l.mlp) for i, l in enumerate(model.model.layers)
                  if isinstance(l.mlp, ChronosMOEFeedForward)]
    log = ActivationLog(
        pairs=set(),
        num_layers=len(moe_layers),
        num_experts=moe_layers[0][1].num_experts if moe_layers else 0,
    )

    def make_hook(layer_idx):
        def hook(mod, inputs, output):
            probs = mod.last_router_probs  # [B, S, E]
            if probs is None:
                return
            top1 = probs.argmax(dim=-1)  # [B, S]
            for e in top1.unique().tolist():
                log.pairs.add((layer_idx, int(e)))
        return hook

    for i, mlp in moe_layers:
        mlp.register_forward_hook(make_hook(i))
    return log


# ── byte counters ─────────────────────────────────────────────────

def _expert_bytes_minimind(model) -> int:
    total = 0
    for l in model.model.layers:
        if isinstance(l.mlp, MOEFeedForward):
            for e in l.mlp.experts:
                total += sum(p.numel() * p.element_size() for p in e.parameters())
    return total


def _resident_expert_bytes_chronos(cache_manager) -> int:
    """VRAM-resident expert bytes only."""
    store = cache_manager.expert_store
    per_expert = store._expert_size_bytes() * store.num_layers
    return per_expert * len(store.vram_lru)


# ── training loops ────────────────────────────────────────────────

def _build_loader(tokenizer, data_path: str, batch_size: int, max_seq_len: int):
    ds = PretrainDataset(data_path, tokenizer, max_length=max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def train_minimind(args, tokenizer) -> Tuple[MiniMindForCausalLM, float]:
    cfg = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads,
        max_position_embeddings=args.max_seq_len,
        use_moe=True,
        num_experts=args.num_experts,
        num_experts_per_tok=1,
        flash_attn=False,
    )
    model = MiniMindForCausalLM(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = _build_loader(tokenizer, args.data_path, args.batch_size, args.max_seq_len)

    model.train()
    last_loss = float("nan")
    step = 0
    while step < args.steps:
        for input_ids, labels in loader:
            if step >= args.steps:
                break
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            res = model(input_ids, labels=labels)
            loss = res.loss + (res.aux_loss if res.aux_loss is not None else 0.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(res.loss.item())
            step += 1
            if step % max(1, args.steps // 10) == 0:
                print(f"  [minimind] step {step}/{args.steps}  ce={last_loss:.4f}")
    return model, last_loss


def train_chronos(args, tokenizer) -> Tuple[ChronosForCausalLM, float, float]:
    cfg = ChronosConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads,
        max_position_embeddings=args.max_seq_len,
        use_moe=True,
        num_experts=args.num_experts,
        num_experts_per_tok=1,
        flash_attn=False,
        # Chronos-specific
        lookahead_steps=2,
        num_shared_experts=1,
        lambda_balance=5e-4,
        lambda_temporal=1e-3,
        lambda_lookahead=0.1,
        vram_budget_gb=0.05,  # tiny on purpose so VRAM cache must evict
        kv_latent_dim=16,
        rope_dim=8,
        sliding_window_size=64,
    )
    from chronos.model.temporal_loss import (
        total_loss as chronos_total_loss,
        lookahead_supervision_loss,
    )

    model = ChronosForCausalLM(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = _build_loader(tokenizer, args.data_path, args.batch_size, args.max_seq_len)

    model.train()
    last_ce = float("nan")
    last_la = float("nan")
    step = 0
    while step < args.steps:
        for input_ids, labels in loader:
            if step >= args.steps:
                break
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            outputs, lookahead_probs = model(input_ids, labels=labels)
            ce = outputs.loss
            aux = outputs.aux_loss

            probs = []
            for layer in model.model.layers:
                if isinstance(layer.mlp, ChronosMOEFeedForward):
                    p = layer.mlp.last_router_probs
                    if p is not None:
                        probs.append(p)
            if probs:
                router_4d = torch.stack(probs, dim=2)  # [B,S,L,E]
                router_mean = router_4d.mean(dim=2)
                teacher = router_mean.detach()
                loss = chronos_total_loss(
                    ce, aux, router_mean,
                    cfg.lambda_balance, cfg.lambda_temporal,
                    lookahead_probs=lookahead_probs,
                    teacher_probs=teacher,
                    lookahead_steps=cfg.lookahead_steps,
                    lambda_lookahead=cfg.lambda_lookahead,
                )
                la_term = lookahead_supervision_loss(
                    lookahead_probs, teacher, cfg.lookahead_steps,
                )
                last_la = float(la_term.item())
            else:
                loss = ce + cfg.lambda_balance * aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            last_ce = float(ce.item())
            step += 1
            if step % max(1, args.steps // 10) == 0:
                print(f"  [chronos]  step {step}/{args.steps}  ce={last_ce:.4f}  la={last_la:.4f}")
    return model, last_ce, last_la


# ── inference benchmark ───────────────────────────────────────────

def benchmark_minimind(model, prompt_ids, max_new_tokens, device) -> Dict:
    model.eval()
    log = _instrument_minimind(model)
    expert_bytes = _expert_bytes_minimind(model)

    # Warmup
    with torch.no_grad():
        _ = model(prompt_ids.to(device), use_cache=True)

    log.pairs.clear()
    t0 = time.time()
    n_generated = 0
    with torch.no_grad():
        x = prompt_ids.to(device)
        past = None
        for _ in range(max_new_tokens):
            out = model(x, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = next_id
            n_generated += 1
    elapsed = time.time() - t0
    return {
        "tokens_per_sec": n_generated / max(elapsed, 1e-9),
        "resident_expert_bytes": expert_bytes,
        "expert_activation_fraction": log.fraction(),
        "n_generated": n_generated,
    }


def benchmark_chronos(model, prompt_ids, max_new_tokens, device, ssd_dir) -> Dict:
    from chronos.runtime.cache_manager import CacheManager
    from chronos.io.cluster_layout import (
        collect_activation_log,
        build_cooccurrence_matrix,
        try_louvain_clustering,
    )

    model.eval()
    cfg = model.config

    # 1. Calibrate co-occurrence on a few forwards (the prompt itself).
    moe_layers = [l.mlp for l in model.model.layers
                  if isinstance(l.mlp, ChronosMOEFeedForward)]
    with torch.no_grad():
        _ = model(prompt_ids.to(device), use_cache=False)
    seq = moe_layers[0].last_router_probs.argmax(dim=-1)[0].cpu().tolist() if moe_layers else []
    C = build_cooccurrence_matrix([seq], cfg.num_experts) if seq else None
    if C is not None:
        clusters = try_louvain_clustering(C)
    else:
        clusters = [list(range(cfg.num_experts))]

    # 2. Stand up CacheManager with safetensors layout.
    mgr = CacheManager(model, cfg, ssd_dir=ssd_dir)
    mgr.expert_store.offload_all_to_ssd(clusters=clusters)
    mgr.expert_store.attach_cluster_manifest()
    # Warm cache to a *small* number of experts so eviction actually fires.
    warm_k = max(1, min(2, cfg.num_experts))
    mgr.warm_up(initial_expert_ids=list(range(warm_k)))
    mgr.start()

    log = _instrument_chronos(model)
    # Reset hooks' set after warm-up so the count reflects only generation.
    log.pairs.clear()

    t0 = time.time()
    n_generated = 0
    with torch.no_grad():
        x = prompt_ids.to(device)
        past = None
        for _ in range(max_new_tokens):
            mask = mgr.availability_mask()
            outputs, lookahead_probs = model(
                x, past_key_values=past, use_cache=True,
                available_expert_masks=[mask] * mgr._num_layers,
            )
            past = outputs.past_key_values
            # current step expert IDs (top-1 of layer-0 router for this token)
            current_ids = []
            if moe_layers and moe_layers[0].last_router_probs is not None:
                current_ids = moe_layers[0].last_router_probs[:, -1, :].argmax(dim=-1).unique().cpu().tolist()
            mgr.step(lookahead_probs, current_ids)
            next_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = next_id
            n_generated += 1
    elapsed = time.time() - t0
    mgr.stop()

    stats = mgr.stats()
    return {
        "tokens_per_sec": n_generated / max(elapsed, 1e-9),
        "resident_expert_bytes": _resident_expert_bytes_chronos(mgr),
        "expert_activation_fraction": log.fraction(),
        "n_generated": n_generated,
        "vram_experts": stats.get("vram_experts"),
        "ram_experts": stats.get("ram_experts"),
        "cluster_aware": stats.get("cluster_aware"),
        "num_clusters": stats.get("num_clusters"),
        "prefetch_hit_rate": stats.get("hit_rate"),
    }


# ── lookahead accuracy ────────────────────────────────────────────

def lookahead_t1_accuracy(model, loader, device, max_batches=4) -> float:
    """Top-1 hit rate of LookaheadRouter t+1 vs real layer-avg routing at t+1."""
    model.eval()
    moe_layers = [l.mlp for l in model.model.layers
                  if isinstance(l.mlp, ChronosMOEFeedForward)]
    hit = 0
    total = 0
    with torch.no_grad():
        for i, (input_ids, _) in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device)
            outputs, lookahead_probs = model(input_ids, use_cache=False)
            probs = [m.last_router_probs for m in moe_layers if m.last_router_probs is not None]
            if not probs or lookahead_probs is None:
                continue
            teacher = torch.stack(probs, dim=2).mean(dim=2)  # [B,S,E]
            teacher_top1 = teacher.argmax(dim=-1)            # [B,S]
            pred_t1 = lookahead_probs[:, :, 1, :].argmax(dim=-1)  # [B,S]
            # Compare: prediction at position t about t+1 vs teacher at t+1
            B, S = teacher_top1.shape
            if S < 2:
                continue
            match = (pred_t1[:, :-1] == teacher_top1[:, 1:]).sum().item()
            hit += int(match)
            total += int(B * (S - 1))
    return hit / max(total, 1)


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(_REPO, "tests/fixtures/tiny_pretrain.jsonl"))
    parser.add_argument("--output", type=str, default=os.path.join(_REPO, "results/compare_results.json"))
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--decode_runs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"== Comparison harness ==")
    print(f"   steps={args.steps} batch={args.batch_size} seq={args.max_seq_len} "
          f"H={args.hidden_size} L={args.num_hidden_layers} E={args.num_experts} "
          f"device={args.device}")

    tokenizer_path = chronos.deps.get_tokenizer_path()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ── train both ──
    print("\n[1/3] Training minimind baseline ...")
    mm_model, mm_final_ce = train_minimind(args, tokenizer)

    print("\n[2/3] Training Chronos (M1+M2) ...")
    torch.manual_seed(args.seed)  # match data order
    ch_model, ch_final_ce, ch_final_la = train_chronos(args, tokenizer)

    # ── benchmark ──
    print("\n[3/3] Inference benchmark ...")
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, 16))

    # average across decode_runs (drop first warmup)
    mm_runs, ch_runs = [], []
    ssd_dir = tempfile.mkdtemp(prefix="chronos_compare_")
    for r in range(args.decode_runs + 1):
        mm_runs.append(benchmark_minimind(mm_model, prompt_ids, args.max_new_tokens, args.device))
        ch_runs.append(benchmark_chronos(ch_model, prompt_ids, args.max_new_tokens, args.device, ssd_dir))
        # fresh ssd dir each time so cluster manifest writes don't contend
        ssd_dir = tempfile.mkdtemp(prefix="chronos_compare_")
    mm_runs, ch_runs = mm_runs[1:], ch_runs[1:]

    def avg(key, runs):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    mm_avg = {k: avg(k, mm_runs) for k in ("tokens_per_sec", "expert_activation_fraction", "resident_expert_bytes")}
    ch_avg = {k: avg(k, ch_runs) for k in ("tokens_per_sec", "expert_activation_fraction", "resident_expert_bytes",
                                            "vram_experts", "ram_experts", "prefetch_hit_rate")}

    # Lookahead accuracy on Chronos
    eval_loader = _build_loader(tokenizer, args.data_path, args.batch_size, args.max_seq_len)
    ch_la_acc = lookahead_t1_accuracy(ch_model, eval_loader, args.device, max_batches=4)

    # Total parameter bytes (whole model)
    mm_param_bytes = sum(p.numel() * p.element_size() for p in mm_model.parameters())
    ch_param_bytes = sum(p.numel() * p.element_size() for p in ch_model.parameters())

    results = {
        "config": vars(args),
        "minimind": {
            **mm_avg,
            "param_bytes": mm_param_bytes,
            "final_ce_loss": mm_final_ce,
        },
        "chronos": {
            **ch_avg,
            "param_bytes": ch_param_bytes,
            "final_ce_loss": ch_final_ce,
            "final_lookahead_loss": ch_final_la,
            "lookahead_t+1_top1": ch_la_acc,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Pretty table
    def _mb(b):
        return f"{b / (1024*1024):8.2f} MB" if b is not None else "    n/a   "
    def _f(v, w=8, p=4):
        return f"{v:>{w}.{p}f}" if v is not None else " " * w + "n/a"

    print("\n" + "=" * 78)
    print(f"{'metric':<32} {'minimind':>20} {'chronos':>20}")
    print("-" * 78)
    print(f"{'param_bytes (whole model)':<32} {_mb(mm_param_bytes):>20} {_mb(ch_param_bytes):>20}")
    print(f"{'resident_expert_bytes':<32} {_mb(mm_avg['resident_expert_bytes']):>20} {_mb(ch_avg['resident_expert_bytes']):>20}")
    print(f"{'expert_activation_fraction':<32} {_f(mm_avg['expert_activation_fraction']):>20} {_f(ch_avg['expert_activation_fraction']):>20}")
    print(f"{'tokens_per_sec (decode)':<32} {_f(mm_avg['tokens_per_sec'], 8, 2):>20} {_f(ch_avg['tokens_per_sec'], 8, 2):>20}")
    print(f"{'final_ce_loss':<32} {_f(mm_final_ce):>20} {_f(ch_final_ce):>20}")
    print(f"{'lookahead_t+1_top1':<32} {'n/a':>20} {_f(ch_la_acc):>20}")
    print(f"{'prefetch_hit_rate':<32} {'n/a':>20} {_f(ch_avg.get('prefetch_hit_rate')):>20}")
    print(f"{'cluster_aware':<32} {'n/a':>20} {str(ch_runs[-1].get('cluster_aware')):>20}")
    print("=" * 78)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
