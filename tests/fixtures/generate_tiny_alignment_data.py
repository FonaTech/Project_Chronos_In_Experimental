"""Generate deterministic tiny SFT / DPO / GRPO fixtures.

Run once: ``python tests/fixtures/generate_tiny_alignment_data.py``.
Outputs ``tiny_sft.jsonl``, ``tiny_dpo.jsonl``, ``tiny_grpo.jsonl`` next to
the existing ``tiny_pretrain.jsonl``. Schema matches minimind's
SFTDataset / DPODataset / RLAIFDataset.
"""
import json
import os
import random


HERE = os.path.dirname(__file__)
SEED = 20260421

# Toy "good" vocabulary that the model is meant to stay on
VOCAB = [
    "the", "quick", "brown", "fox", "machine", "learning", "model", "expert",
    "router", "cluster", "memory", "cache", "prefetch", "latency", "throughput",
    "tensor", "training", "inference", "decode", "weight", "gradient",
    "softmax", "attention", "layer", "embedding", "vocab", "sequence",
    "loss", "optimizer", "epoch", "batch", "step", "token", "logit",
    "minimind", "chronos", "hybrid", "mixture", "of", "experts", "sparse",
    "dense", "neural", "network", "transformer", "rope", "rotary", "kv",
]
NOISE = ["zzz", "qqq", "@@@", "###", "%%%", "&&&", "***", "qwxz", "aybc", "fghk"]

PROMPTS = [
    "describe a chronos cluster cache",
    "explain mixture of experts routing",
    "what is lookahead prefetch",
    "summarize sliding window attention",
    "define expert load balancing",
    "tell me about latent kv compression",
    "why use sparse moe",
    "outline the temporal locality loss",
    "describe the prefill stage",
    "what is the expert activation fraction",
]


def _rng(seed):
    return random.Random(seed)


def _good_response(rng, n=24):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def _bad_response(rng, n=24):
    # Mostly noise tokens; occasionally a vocab word so the model still has
    # gradient signal but the response is clearly worse than chosen.
    return " ".join(
        rng.choice(NOISE) if rng.random() < 0.7 else rng.choice(VOCAB)
        for _ in range(n)
    )


def write_sft(out_path, n=200):
    rng = _rng(SEED)
    with open(out_path, "w") as f:
        for _ in range(n):
            prompt = rng.choice(PROMPTS)
            resp = _good_response(rng, n=rng.randint(16, 32))
            row = {
                "conversations": [
                    {"role": "user", "content": prompt,
                     "reasoning_content": "", "tools": "", "tool_calls": ""},
                    {"role": "assistant", "content": resp,
                     "reasoning_content": "", "tools": "", "tool_calls": ""},
                ]
            }
            f.write(json.dumps(row) + "\n")


def write_dpo(out_path, n=100):
    rng = _rng(SEED + 1)
    with open(out_path, "w") as f:
        for _ in range(n):
            prompt = rng.choice(PROMPTS)
            chosen_resp = _good_response(rng, n=rng.randint(16, 28))
            rejected_resp = _bad_response(rng, n=rng.randint(16, 28))
            row = {
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen_resp},
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected_resp},
                ],
            }
            f.write(json.dumps(row) + "\n")


def write_grpo(out_path, n=100):
    rng = _rng(SEED + 2)
    with open(out_path, "w") as f:
        for _ in range(n):
            prompt = rng.choice(PROMPTS)
            row = {
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},  # to be filled by rollout
                ]
            }
            f.write(json.dumps(row) + "\n")


def main():
    write_sft(os.path.join(HERE, "tiny_sft.jsonl"))
    write_dpo(os.path.join(HERE, "tiny_dpo.jsonl"))
    write_grpo(os.path.join(HERE, "tiny_grpo.jsonl"))
    print("Wrote tiny_sft.jsonl, tiny_dpo.jsonl, tiny_grpo.jsonl into", HERE)


if __name__ == "__main__":
    main()
