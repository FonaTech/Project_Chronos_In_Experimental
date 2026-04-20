"""
chronos/trainer/reward.py

Pluggable reward models for GRPO. Two built-ins:

- ``ToyReward`` — length + vocab-purity + repetition-penalty heuristic.
  No external dependencies, used in smoke tests and when no real reward
  model is configured.
- ``LMRewardModel`` — wraps an HF AutoModel that exposes a ``.get_score(...)``
  method (the minimind convention, see
  ``minimind-master/trainer/trainer_utils.py:160``). Loads lazily so
  importing this file never forces an HF network download.

Any callable ``score(prompt: str, response_tokens: List[int], tokenizer) -> float``
is accepted by ``ChronosGRPOTrainer``; the classes below simply package
that protocol.
"""
from __future__ import annotations

from typing import List, Optional


TOY_VOCAB = {
    "the", "quick", "brown", "fox", "machine", "learning", "model", "expert",
    "router", "cluster", "memory", "cache", "prefetch", "latency", "throughput",
    "tensor", "training", "inference", "decode", "weight", "gradient",
    "softmax", "attention", "layer", "embedding", "vocab", "sequence",
    "loss", "optimizer", "epoch", "batch", "step", "token", "logit",
    "minimind", "chronos", "hybrid", "mixture", "of", "experts", "sparse",
    "dense", "neural", "network", "transformer", "rope", "rotary", "kv",
}


class ToyReward:
    """Zero-dependency heuristic reward: length + vocab-purity − repetition."""

    def __init__(self, target_len: int = 24, vocab: Optional[set] = None):
        self.target_len = target_len
        self.vocab = vocab or TOY_VOCAB

    def __call__(self, prompt: str, response_tokens: List[int], tokenizer) -> float:
        return self.score(prompt, response_tokens, tokenizer)

    def score(self, prompt: str, response_tokens: List[int], tokenizer) -> float:
        text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        words = text.split()
        if not words:
            return -1.0
        length_reward = min(len(words) / max(self.target_len, 1), 1.0)
        on_vocab = sum(1 for w in words if w.lower() in self.vocab)
        vocab_purity = on_vocab / len(words)
        dupes = sum(1 for a, b in zip(words[:-1], words[1:]) if a == b)
        repetition_penalty = -dupes / max(len(words), 1)
        return length_reward + vocab_purity + repetition_penalty


class LMRewardModel:
    """HF-backed reward model.

    Wraps any ``AutoModel`` that exposes ``get_score(tokenizer, messages) -> float``
    (the reward-model ABI used by minimind and several open reward-model
    checkpoints). The model is loaded lazily on the first call so that
    importing this module is free.

    Example::

        reward = LMRewardModel("path/to/internlm2-1_8b-reward")
        r = reward.score(prompt, response_tokens, tokenizer)

    The score is clipped to [-3.0, 3.0] to keep GRPO advantages well-scaled,
    matching the minimind convention.
    """

    def __init__(self, model_path: str, device: str = "cpu",
                 dtype_str: str = "float16", score_clip: float = 3.0):
        self.model_path = model_path
        self.device = device
        self.dtype_str = dtype_str
        self.score_clip = float(score_clip)
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModel
        dtype = torch.float16 if self.dtype_str == "float16" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device).eval()

    def __call__(self, prompt: str, response_tokens: List[int], tokenizer) -> float:
        return self.score(prompt, response_tokens, tokenizer)

    def score(self, prompt: str, response_tokens: List[int], tokenizer) -> float:
        self._load()
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        s = self._model.get_score(self._tokenizer, messages)
        return max(min(float(s), self.score_clip), -self.score_clip)


def build_reward_fn(spec: Optional[str]) -> callable:
    """Convert a CLI spec string into a reward callable.

    Spec forms:
        None | ""  | "toy"                 → ToyReward()
        "lm:/path/to/reward-model"         → LMRewardModel(path)
        "lm:/path/to/reward,device=cuda"   → LMRewardModel(path, device="cuda")
    """
    if not spec or spec == "toy":
        return ToyReward()
    if spec.startswith("lm:"):
        body = spec[3:]
        parts = body.split(",")
        path = parts[0]
        kwargs = {}
        for kv in parts[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                kwargs[k.strip()] = v.strip()
        return LMRewardModel(path, **kwargs)
    raise ValueError(f"unknown reward spec: {spec!r}")
