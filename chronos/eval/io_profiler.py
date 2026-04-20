"""
chronos/eval/io_profiler.py

Phase 1 validation tool: measures LookaheadRouter prediction accuracy
and temporal routing smoothness offline, without running full inference.

Usage:
    python -m chronos.eval.io_profiler \
        --model_path /path/to/chronos_checkpoint \
        --data_path  /path/to/eval.jsonl \
        --device     cuda

Outputs:
    - t+1 Top-1 accuracy (target: >= 85%)
    - t+2 Top-1 accuracy (target: >= 75%)
    - Mean L2 routing shift (lower = more cache-friendly)
    - Cache hit rate estimate given VRAM budget
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.moe_chronos import ChronosMOEFeedForward


def collect_ground_truth_routing(model, input_ids, device):
    """
    Run a full forward pass and collect per-layer, per-step top-1 expert indices.
    Returns: [B, T, num_layers] int tensor of selected expert indices.
    """
    model.eval()
    input_ids = input_ids.to(device)
    with torch.no_grad():
        _, _ = model(input_ids, use_cache=False)

    layers = [l for l in model.model.layers if isinstance(l.mlp, ChronosMOEFeedForward)]
    if not layers:
        return None

    # last_router_probs: [B, S, E] per layer
    probs_per_layer = [l.mlp.last_router_probs for l in layers]  # list of [B, S, E]
    stacked = torch.stack(probs_per_layer, dim=2)                 # [B, S, L, E]
    top1 = stacked.argmax(dim=-1)                                 # [B, S, L]
    return top1


def evaluate_lookahead_accuracy(model, dataloader, device, lookahead_steps=2):
    """
    For each sequence, compare LookaheadRouter's predicted expert at t+k
    against the ground-truth expert selected at t+k.

    Returns dict with accuracy per step and mean L2 routing shift.
    """
    model.eval()
    correct = {k: 0 for k in range(1, lookahead_steps + 1)}
    total = 0
    l2_shifts = []

    for input_ids, _ in dataloader:
        input_ids = input_ids.to(device)
        B, T = input_ids.shape
        if T < lookahead_steps + 2:
            continue

        # Ground truth routing for full sequence
        gt_routing = collect_ground_truth_routing(model, input_ids, device)
        if gt_routing is None:
            continue
        # gt_routing: [B, T, L] — use layer 0 as reference
        gt_top1 = gt_routing[:, :, 0]  # [B, T]

        # LookaheadRouter predictions from block 0 hidden states
        with torch.no_grad():
            hidden_after_block0 = None

            def _hook(module, inp, out):
                nonlocal hidden_after_block0
                hidden_after_block0 = out[0].detach()

            handle = model.model.layers[0].register_forward_hook(_hook)
            model(input_ids, use_cache=False)
            handle.remove()

        if hidden_after_block0 is None:
            continue

        lookahead_probs = model.model.lookahead_router(hidden_after_block0)
        # lookahead_probs: [B, T, K+1, E], index 1..K = future predictions

        for k in range(1, lookahead_steps + 1):
            if T <= k:
                continue
            # Predicted expert at step t for future t+k
            pred_top1 = lookahead_probs[:, :-k, k, :].argmax(dim=-1)  # [B, T-k]
            gt_at_tk = gt_top1[:, k:]                                   # [B, T-k]
            correct[k] += (pred_top1 == gt_at_tk).sum().item()
            total_k = pred_top1.numel()
            if k == 1:
                total += total_k

        # L2 routing shift between adjacent steps
        # Use mean router probs across layers
        from chronos.model.moe_chronos import ChronosMOEFeedForward
        moe_layers = [l for l in model.model.layers if isinstance(l.mlp, ChronosMOEFeedForward)]
        if moe_layers:
            probs = torch.stack([l.mlp.last_router_probs for l in moe_layers], dim=2).mean(dim=2)
            diff = probs[:, 1:, :] - probs[:, :-1, :]
            l2_shifts.append((diff ** 2).sum(dim=-1).mean().item())

    accuracies = {}
    for k in range(1, lookahead_steps + 1):
        denom = total if k == 1 else max(total, 1)
        accuracies[f"top1_acc_t+{k}"] = correct[k] / max(denom, 1)

    mean_l2 = sum(l2_shifts) / max(len(l2_shifts), 1)
    return {**accuracies, "mean_l2_routing_shift": mean_l2}


def estimate_cache_hit_rate(model, dataloader, device, vram_budget_gb=4.0):
    """
    Simulate a VRAM LRU cache of expert weights and estimate hit rate
    given the LookaheadRouter's prefetch predictions.
    """
    import collections

    # Estimate single expert weight size in GB
    moe_layers = [l for l in model.model.layers if isinstance(l.mlp, ChronosMOEFeedForward)]
    if not moe_layers:
        return {"cache_hit_rate": 1.0}

    expert_params = sum(p.numel() for p in moe_layers[0].mlp.experts[0].parameters())
    expert_size_gb = expert_params * 2 / (1024 ** 3)  # fp16
    num_experts = model.config.num_experts
    cache_capacity = max(1, int(vram_budget_gb / expert_size_gb))

    hits, total = 0, 0
    for input_ids, _ in dataloader:
        input_ids = input_ids.to(device)
        gt_routing = collect_ground_truth_routing(model, input_ids, device)
        if gt_routing is None:
            continue
        gt_top1 = gt_routing[:, :, 0]  # [B, T]

        lru = collections.OrderedDict()
        for t in range(gt_top1.shape[1]):
            for b in range(gt_top1.shape[0]):
                eid = gt_top1[b, t].item()
                if eid in lru:
                    lru.move_to_end(eid)
                    hits += 1
                else:
                    if len(lru) >= cache_capacity:
                        lru.popitem(last=False)
                    lru[eid] = True
                total += 1

    return {
        "cache_hit_rate": hits / max(total, 1),
        "cache_capacity_experts": cache_capacity,
        "expert_size_gb": round(expert_size_gb, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Chronos Phase 1 Validation")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--vram_budget_gb", type=float, default=4.0)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_experts", type=int, default=4)
    args = parser.parse_args()

    config = ChronosConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_experts=args.num_experts,
        use_moe=True,
        vram_budget_gb=args.vram_budget_gb,
    )
    model = ChronosForCausalLM(config).to(args.device)

    if args.model_path:
        weights = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(weights, strict=False)
        print(f"Loaded weights from {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        chronos.deps.get_tokenizer_path()
    )

    from dataset.lm_dataset import PretrainDataset
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\n=== Phase 1 Validation: LookaheadRouter Accuracy ===")
    acc_results = evaluate_lookahead_accuracy(model, loader, args.device)
    for k, v in acc_results.items():
        status = ""
        if "t+1" in k:
            status = "✓ PASS" if v >= 0.85 else "✗ FAIL (target: ≥85%)"
        elif "t+2" in k:
            status = "✓ PASS" if v >= 0.75 else "✗ FAIL (target: ≥75%)"
        print(f"  {k}: {v:.4f}  {status}")

    print("\n=== Cache Hit Rate Estimate ===")
    cache_results = estimate_cache_hit_rate(model, loader, args.device, args.vram_budget_gb)
    for k, v in cache_results.items():
        print(f"  {k}: {v}")

    print("\n=== Summary ===")
    t1 = acc_results.get("top1_acc_t+1", 0)
    t2 = acc_results.get("top1_acc_t+2", 0)
    phase1_pass = t1 >= 0.85 and t2 >= 0.75
    print(f"  Phase 1 Gate: {'PASS ✓' if phase1_pass else 'FAIL ✗'}")
    print(f"  t+1 acc={t1:.1%}  t+2 acc={t2:.1%}  "
          f"L2 shift={acc_results['mean_l2_routing_shift']:.4f}")


if __name__ == "__main__":
    main()
