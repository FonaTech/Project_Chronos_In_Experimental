"""
chronos/eval/benchmark.py

Phase 3 end-to-end evaluation:
- Perplexity on held-out text (accuracy proxy)
- Throughput (tokens/s) with and without async prefetch
- KV cache memory usage comparison: standard vs hybrid attention
- Soft gating fallback rate

Usage:
    python -m chronos.eval.benchmark \
        --model_path ./out/chronos_512_moe.pth \
        --data_path  ./dataset/eval.jsonl \
        --device     cpu \
        --max_new_tokens 64
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import argparse
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.hybrid_attention import MLAAttention, SlidingWindowAttention


def compute_perplexity(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, (input_ids, labels) in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            out, _ = model(input_ids, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens
    ppl = math.exp(total_loss / max(total_tokens, 1))
    return ppl


def measure_throughput(model, input_ids, device, max_new_tokens=64):
    model.eval()
    input_ids = input_ids.to(device)
    t0 = time.monotonic()
    with torch.no_grad():
        out, lp = model(input_ids, use_cache=True)
        past_kv = out.past_key_values
        for _ in range(max_new_tokens):
            x = torch.randint(0, model.config.vocab_size, (1, 1), device=device)
            out2, _ = model(x, past_key_values=past_kv, use_cache=True)
            past_kv = out2.past_key_values
    elapsed = time.monotonic() - t0
    return max_new_tokens / elapsed


def kv_cache_memory_bytes(past_key_values, model):
    """Estimate total KV cache memory in bytes (fp32)."""
    total = 0
    for i, kv in enumerate(past_key_values):
        if kv is None:
            continue
        for t in kv:
            total += t.numel() * t.element_size()
    return total


def run_benchmark(args):
    config = ChronosConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_experts=args.num_experts,
        use_moe=True,
        use_hybrid_attention=True,
        kv_latent_dim=args.kv_latent_dim,
        sliding_window_size=args.sliding_window_size,
        vram_budget_gb=args.vram_budget_gb,
    )
    model = ChronosForCausalLM(config).to(args.device)

    if args.model_path:
        weights = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(weights, strict=False)
        print(f"Loaded: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        chronos.deps.get_tokenizer_path()
    )

    from dataset.lm_dataset import PretrainDataset
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\n=== Phase 3 Benchmark ===")

    # 1. Perplexity
    ppl = compute_perplexity(model, loader, args.device)
    print(f"Perplexity: {ppl:.2f}")

    # 2. Throughput
    sample_ids = torch.randint(0, config.vocab_size, (1, 32))
    tps = measure_throughput(model, sample_ids, args.device, args.max_new_tokens)
    print(f"Throughput: {tps:.1f} tokens/s")

    # 3. KV cache memory after N decode steps
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, config.vocab_size, (1, 32), device=args.device)
        out, _ = model(x, use_cache=True)
        past_kv = out.past_key_values
        for _ in range(args.max_new_tokens):
            xn = torch.randint(0, config.vocab_size, (1, 1), device=args.device)
            out2, _ = model(xn, past_key_values=past_kv, use_cache=True)
            past_kv = out2.past_key_values

    kv_bytes = kv_cache_memory_bytes(past_kv, model)
    print(f"KV cache after {args.max_new_tokens} steps: {kv_bytes/1024:.1f} KB")

    # 4. Layer-wise cache sizes
    print("\nLayer-wise KV cache:")
    for i, kv in enumerate(past_kv):
        if kv is not None:
            layer_type = type(model.model.layers[i].self_attn).__name__
            seq_len = kv[0].shape[1]
            print(f"  Layer {i:2d} ({layer_type:25s}): seq_len={seq_len:4d}  "
                  f"bytes={sum(t.numel()*t.element_size() for t in kv)//1024}KB")

    print(f"\n=== Summary ===")
    print(f"  PPL={ppl:.2f}  TPS={tps:.1f}  KV={kv_bytes//1024}KB  "
          f"hybrid_attn=True  window={config.sliding_window_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--kv_latent_dim", type=int, default=64)
    parser.add_argument("--sliding_window_size", type=int, default=2048)
    parser.add_argument("--vram_budget_gb", type=float, default=4.0)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
