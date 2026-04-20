"""Generate a deterministic tiny pretrain corpus for the comparison harness.

Run once: ``python tests/fixtures/generate_tiny_dataset.py``.
The output ``tiny_pretrain.jsonl`` is committed alongside this script so CI
gets a stable, hash-pinned corpus without needing to regenerate.
"""
import json
import os
import random


VOCAB = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "machine", "learning", "model", "expert", "router", "cluster",
    "memory", "cache", "prefetch", "latency", "throughput", "tensor",
    "training", "inference", "decode", "prefill", "weight", "gradient",
    "softmax", "attention", "layer", "embedding", "vocab", "sequence",
    "loss", "optimizer", "adam", "epoch", "batch", "step", "token", "logit",
    "minimind", "chronos", "hybrid", "mixture", "of", "experts", "sparse",
    "dense", "neural", "network", "transformer", "rope", "rotary", "kv",
]

NUM_LINES = 2000
LINE_TOKENS = 64
SEED = 20260421


def main():
    rng = random.Random(SEED)
    out_path = os.path.join(os.path.dirname(__file__), "tiny_pretrain.jsonl")
    with open(out_path, "w") as f:
        for _ in range(NUM_LINES):
            words = [rng.choice(VOCAB) for _ in range(LINE_TOKENS)]
            f.write(json.dumps({"text": " ".join(words)}) + "\n")
    print(f"Wrote {NUM_LINES} lines to {out_path}")


if __name__ == "__main__":
    main()
