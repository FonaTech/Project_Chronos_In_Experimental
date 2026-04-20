"""
chronos/mlx/inference.py — MLX inference engine for Project Chronos.

Mirrors ChronosInferenceEngine (PyTorch) but uses:
- ChronosMLXModel for forward passes
- MLXExpertStore (unified memory, no H2D stream)
- mx.compile() for static-graph acceleration
- Async prefetch via Python threading (SSD → unified memory)

Usage:
    engine = ChronosMLXInferenceEngine(model, config)
    for token_id in engine.generate("Once upon a time", max_new_tokens=128):
        print(tokenizer.decode([token_id]), end="", flush=True)
"""
import threading
import queue
from typing import Iterator, List, Optional

import mlx.core as mx
import mlx.nn as nn

from chronos.mlx.expert_store import MLXExpertStore


class ChronosMLXInferenceEngine:
    """
    Streaming token-by-token inference with async expert prefetch.

    On Apple Silicon the prefetch thread reads .npy files from SSD and
    calls mx.array() to bring weights into unified memory.  The main
    thread calls mx.eval() (promote_to_vram) before the next forward
    pass — Metal overlaps the IO with prior computation automatically.
    """

    def __init__(self, model, config, ssd_dir: str = "./expert_cache_mlx"):
        self.model = model
        self.config = config
        self.store = MLXExpertStore(model, config, ssd_dir=ssd_dir)
        self._prefetch_q: queue.Queue = queue.Queue(maxsize=config.prefetch_depth * 2)
        self._stop = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, daemon=True, name="mlx-prefetch"
        )
        self._prefetch_thread.start()

    # ── Prefetch background thread ────────────────────────────────

    def _prefetch_loop(self):
        while not self._stop.is_set():
            try:
                expert_ids: List[int] = self._prefetch_q.get(timeout=0.05)
            except queue.Empty:
                continue
            self.store.prefetch_to_ram(expert_ids)
            self._prefetch_q.task_done()

    def _schedule_prefetch(self, expert_ids: List[int]):
        try:
            self._prefetch_q.put_nowait(expert_ids)
        except queue.Full:
            pass

    def stop(self):
        self._stop.set()

    # ── Generation loop ───────────────────────────────────────────

    def generate(
        self,
        input_ids: mx.array,           # [1, S] (already tokenised)
        max_new_tokens: int = 128,
        temperature: float = 0.85,
        top_p: float = 0.9,
        scheduler=None,                # Optional[PrefillScheduler] — lazy import
    ) -> Iterator[int]:
        """
        Yields token IDs one at a time.

        If scheduler is provided (PrefillScheduler), runs prefill-time batch
        loading before the generation loop (dual-layer mode).  Otherwise falls
        back to per-token LookaheadRouter-driven prefetch.
        """
        # ── Prefill phase: front-load expert IO ───────────────────────────
        if scheduler is not None:
            import torch, numpy as np
            ids_np = np.array(input_ids.tolist(), dtype=np.int64)
            ids_pt = torch.from_numpy(ids_np)
            scheduler.prepare(ids_pt, device="cpu")
            scheduler.wait()

        # Prefill forward
        logits, lookahead_probs, cache = self.model(input_ids)
        mx.eval(logits)

        # Sample from prefill
        next_token = self._sample(logits[:, -1, :], temperature, top_p)
        activated_ids: List[int] = []
        yield int(next_token.item())

        for _ in range(max_new_tokens - 1):
            if scheduler is not None:
                # Use PrefillScheduler masks (already loaded at prefill)
                n_layers = len(self.model.layers)
                avail_masks = [self.store.vram_availability_mask()] * n_layers
            else:
                # LookaheadRouter-driven prefetch
                if lookahead_probs is not None:
                    future_ids = self._predict_future_experts(lookahead_probs)
                    self._schedule_prefetch(future_ids)
                avail_masks = self._build_avail_masks(next_token)

            token_in = next_token.reshape(1, 1)
            logits, lookahead_probs, cache = self.model(
                token_in,
                cache=cache,
                available_expert_masks=avail_masks,
            )
            mx.eval(logits)

            next_token = self._sample(logits[:, -1, :], temperature, top_p)
            mx.eval(next_token)
            tok_id = int(next_token.item())
            activated_ids.append(tok_id)
            yield tok_id

        # Update frequency heuristic after generation
        if scheduler is not None:
            scheduler.record_activation(list(set(activated_ids)))

    def _predict_future_experts(self, lookahead_probs: mx.array) -> List[int]:
        """Extract top-1 expert IDs for t+1 and t+2 from lookahead probs."""
        # lookahead_probs: [B, S, K+1, E] — take last token, steps 1..K
        future = lookahead_probs[0, -1, 1:, :]         # [K, E]
        top_ids = mx.argmax(future, axis=-1).tolist()   # [K]
        return list(set(top_ids))

    def _build_avail_masks(self, _token) -> List[mx.array]:
        """Promote prefetched experts and return per-layer availability masks."""
        n_layers = len(self.model.layers)
        mask = self.store.vram_availability_mask()
        return [mask] * n_layers

    @staticmethod
    def _sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        scaled = logits / temperature
        probs = mx.softmax(scaled, axis=-1)
        # top-p nucleus sampling
        sorted_idx = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        # mask tokens beyond top-p threshold
        cutoff = mx.array(top_p)
        nucleus = (cumsum - sorted_probs) < cutoff
        filtered = mx.where(nucleus, sorted_probs, mx.zeros_like(sorted_probs))
        filtered = filtered / (filtered.sum(axis=-1, keepdims=True) + 1e-9)
        sample_idx = mx.random.categorical(mx.log(filtered + 1e-9))
        return mx.take(sorted_idx[0], sample_idx)
