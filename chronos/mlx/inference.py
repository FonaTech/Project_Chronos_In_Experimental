"""
chronos/mlx/inference.py — MLX inference engine for Project Chronos.

Mirrors ChronosInferenceEngine (PyTorch) but uses:
- ChronosMLXModel for forward passes
- MLXExpertStore (unified memory, no H2D stream)
- mx.compile() for static-graph acceleration
- Async prefetch via Python threading (SSD → unified memory)

Usage:
    engine = ChronosMLXInferenceEngine(model, config)
    for token_id in engine.generate("Explain how Chronos moves expert IO into prefill.", max_new_tokens=128):
        print(tokenizer.decode([token_id]), end="", flush=True)
"""
import threading
import queue
import time
from typing import Iterator, List, Optional

import mlx.core as mx
import mlx.nn as nn

from chronos.mlx.expert_store import MLXExpertStore
from chronos.mlx.moe import ChronosMLXMOE


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
        self.last_stats: dict = {}
        self._last_predicted: set[int] = set()
        self._stats_lock = threading.Lock()
        self._runtime_stats = {
            "resident_hits": 0,
            "resident_misses": 0,
            "resident_vram_hits": 0,
            "resident_ram_hits": 0,
            "selection_hits": 0,
            "selection_misses": 0,
            "prediction_hits": 0,
            "prediction_total": 0,
            "prefetch_queue_drops": 0,
            "prefetch_wait_time_s": 0.0,
            "sync_ssd_loads": 0,
            "on_demand_load_time_s": 0.0,
        }

    def setup(self, warm_expert_ids: Optional[List[int]] = None):
        self.store.offload_all_to_ssd()
        self.store.replace_live_experts_with_placeholders()
        self.store.warm_up(warm_expert_ids)

    def setup_with_clusters(self, clusters: List[List[int]], warm_expert_ids: Optional[List[int]] = None):
        self.store.offload_all_to_ssd(clusters=clusters)
        self.store.replace_live_experts_with_placeholders()
        self.store.warm_up(warm_expert_ids)

    def _install_runtime_hooks(self):
        def loader(eid: int) -> bool:
            eid = int(eid)
            was_hot = eid in self.store.hot_expert_ids()
            with self.store._lock:
                was_warm = (
                    eid in self.store._warm
                    and self.store._layer_states_complete(self.store._warm.get(eid))
                )
            with self._stats_lock:
                if was_hot or was_warm:
                    self._runtime_stats["selection_hits"] += 1
                else:
                    self._runtime_stats["selection_misses"] += 1
            t0 = time.monotonic()
            if not was_hot and not was_warm:
                with self._stats_lock:
                    self._runtime_stats["sync_ssd_loads"] += 1
            ok = self.store.promote_to_vram(eid)
            elapsed = time.monotonic() - t0
            with self._stats_lock:
                self._runtime_stats["on_demand_load_time_s"] += elapsed
                if ok:
                    self._runtime_stats["resident_hits" if (was_hot or was_warm) else "resident_misses"] += 1
                    if was_hot:
                        self._runtime_stats["resident_vram_hits"] += 1
                    elif was_warm:
                        self._runtime_stats["resident_ram_hits"] += 1
                else:
                    self._runtime_stats["resident_misses"] += 1
                self._runtime_stats["prediction_total"] += 1
                if (was_hot or was_warm) and eid in self._last_predicted:
                    self._runtime_stats["prediction_hits"] += 1
            return ok

        def touch(eid: int) -> None:
            eid = int(eid)
            with self.store._lock:
                if eid in self.store._hot_lru:
                    if self.store._expert_live_all_layers(eid):
                        self.store._hot_lru.move_to_end(eid)
                        with self._stats_lock:
                            self._runtime_stats["selection_hits"] += 1
                            self._runtime_stats["resident_vram_hits"] += 1
                            self._runtime_stats["prediction_total"] += 1
                            if eid in self._last_predicted:
                                self._runtime_stats["prediction_hits"] += 1
                    else:
                        self.store._hot_lru.pop(eid, None)

        for layer in self.model.layers:
            moe = getattr(layer, "mlp", None)
            if isinstance(moe, ChronosMLXMOE):
                moe.runtime_miss_policy = "sync_on_demand"
                moe.runtime_on_demand_loader = loader
                moe.runtime_touch_expert = touch

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
        if not expert_ids:
            return
        try:
            self._prefetch_q.put_nowait(expert_ids)
        except queue.Full:
            with self._stats_lock:
                self._runtime_stats["prefetch_queue_drops"] += 1

    def _prefetch_and_promote_window(self, expert_ids: List[int], timeout_s: float = 0.012) -> None:
        if not expert_ids or self.store.storage_format == "full_dram":
            return
        pending = []
        for eid in dict.fromkeys(int(eid) for eid in expert_ids):
            with self.store._lock:
                if eid in self.store._hot_lru:
                    continue
                if eid in self.store._warm and self.store._layer_states_complete(self.store._warm.get(eid)):
                    pending.append(eid)
                    continue
            pending.append(eid)
        if not pending:
            return
        self._schedule_prefetch(pending)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            self._promote_ready(pending)
            with self.store._lock:
                ready = all(
                    eid in self.store._hot_lru
                    or (eid in self.store._warm and self.store._layer_states_complete(self.store._warm.get(eid)))
                    for eid in pending
                )
            if ready:
                break
            time.sleep(0.001)
        self._promote_ready(pending)
        with self._stats_lock:
            self._runtime_stats["prefetch_wait_time_s"] += max(0.0, time.monotonic() - (deadline - max(0.0, float(timeout_s))))

    def stop(self):
        self._stop.set()
        try:
            self._prefetch_q.put_nowait([])
        except queue.Full:
            pass
        if self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=2.0)

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
        t0 = time.monotonic()
        setup_mem = self._memory_snapshot()
        self._last_predicted = set()
        with self._stats_lock:
            self._runtime_stats = {
                "resident_hits": 0,
                "resident_misses": 0,
                "resident_vram_hits": 0,
                "resident_ram_hits": 0,
                "selection_hits": 0,
                "selection_misses": 0,
                "prediction_hits": 0,
                "prediction_total": 0,
                "prefetch_queue_drops": 0,
                "prefetch_wait_time_s": 0.0,
                "sync_ssd_loads": 0,
                "on_demand_load_time_s": 0.0,
            }
        self._install_runtime_hooks()
        # ── Prefill phase: front-load expert IO ───────────────────────────
        if scheduler is not None:
            import torch, numpy as np
            ids_np = np.array(input_ids.tolist(), dtype=np.int64)
            ids_pt = torch.from_numpy(ids_np)
            scheduler.prepare(ids_pt, device="cpu")
            scheduler.wait()

        # Prefill forward
        prefill_t0 = time.monotonic()
        if self.store.storage_format == "full_dram":
            prefill_masks = None
        else:
            prefill_masks = self._build_avail_masks(None)
        logits, lookahead_probs, cache = self.model(input_ids, available_expert_masks=prefill_masks)
        mx.eval(logits)
        prefill_t1 = time.monotonic()
        prefill_mem = self._memory_snapshot()

        # Sample from prefill
        next_token = self._sample(logits[:, -1, :], temperature, top_p)
        activated_ids: List[int] = []
        tokens = 1
        if scheduler is None and lookahead_probs is not None:
            future_ids = self._predict_future_experts(lookahead_probs)
            self._last_predicted = set(int(eid) for eid in future_ids)
            self._prefetch_and_promote_window(future_ids, timeout_s=0.025)
        yield int(next_token.item())

        for _ in range(max_new_tokens - 1):
            if scheduler is not None:
                avail_masks = self._build_avail_masks(next_token)
            else:
                # LookaheadRouter-driven prefetch
                if lookahead_probs is not None:
                    future_ids = self._predict_future_experts(lookahead_probs)
                    self._last_predicted = set(int(eid) for eid in future_ids)
                    self._prefetch_and_promote_window(future_ids)
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
            tokens += 1
            yield tok_id

        # Update frequency heuristic after generation
        if scheduler is not None:
            scheduler.record_activation(list(set(activated_ids)))
        end = time.monotonic()
        end_mem = self._memory_snapshot()
        self.last_stats = {
            "tokens_generated": tokens,
            "elapsed_s": round(end - t0, 3),
            "tokens_per_sec": round(tokens / max(end - t0, 1e-6), 2),
            "prefill_time_s": round(prefill_t1 - prefill_t0, 3),
            "decode_time_s": round(max(0.0, end - prefill_t1), 3),
            **self._memory_fields(setup_mem, "after_setup"),
            **self._memory_fields(prefill_mem, "after_prefill"),
            **self._memory_fields(end_mem, "after_decode"),
            **self._runtime_stat_fields(),
            **self.store.stats(),
        }

    def _predict_future_experts(self, lookahead_probs: mx.array) -> List[int]:
        """Extract bounded top-k expert IDs for future steps."""
        # lookahead_probs: [B, S, K+1, E] — take last token, steps 1..K
        future = lookahead_probs[0, -1, 1:, :]         # [K, E]
        top_k = max(1, min(
            int(getattr(self.config, "num_experts_per_tok", 1) or 1),
            int(getattr(self.config, "num_experts", future.shape[-1]) or future.shape[-1]),
        ))
        ids: list[int] = []
        for k in range(future.shape[0]):
            step = future[k]
            if top_k == 1:
                ids.append(int(mx.argmax(step).item()))
            else:
                ids.extend(int(v) for v in mx.argpartition(-step, kth=top_k - 1, axis=-1)[:top_k].tolist())
        return list(dict.fromkeys(ids))

    def _build_avail_masks(self, _token) -> List[set[int]] | None:
        """Promote prefetched experts and return per-layer availability masks."""
        n_layers = len(self.model.layers)
        if self.store.storage_format == "full_dram":
            return None
        mask = self.store.hot_expert_ids()
        return [mask] * n_layers

    def _promote_ready(self, expert_ids: List[int]) -> None:
        for eid in expert_ids:
            eid = int(eid)
            with self.store._lock:
                ready = (
                    eid in self.store._warm
                    and self.store._layer_states_complete(self.store._warm.get(eid))
                ) or eid in self.store._hot_lru
            if ready:
                self.store.promote_to_vram(eid)

    def _runtime_stat_fields(self) -> dict:
        with self._stats_lock:
            stats = dict(self._runtime_stats)
        total = int(stats.get("selection_hits", 0)) + int(stats.get("selection_misses", 0))
        pred_total = int(stats.get("prediction_total", 0))
        return {
            "resident_hit_rate": round(float(stats.get("selection_hits", 0)) / max(total, 1), 4),
            "cache_hit_rate": round(float(stats.get("selection_hits", 0)) / max(total, 1), 4),
            "cache_hits": int(stats.get("selection_hits", 0)),
            "cache_misses": int(stats.get("selection_misses", 0)),
            "expert_selection_hits": int(stats.get("selection_hits", 0)),
            "expert_selection_misses": int(stats.get("selection_misses", 0)),
            "prediction_hit_rate": round(float(stats.get("prediction_hits", 0)) / max(pred_total, 1), 4),
            "prediction_hits": int(stats.get("prediction_hits", 0)),
            "prediction_total": pred_total,
            "resident_vram_hits": int(stats.get("resident_vram_hits", 0)),
            "resident_ram_hits": int(stats.get("resident_ram_hits", 0)),
            "prefetch_queue_drops": int(stats.get("prefetch_queue_drops", 0)),
            "prefetch_wait_time_s": round(float(stats.get("prefetch_wait_time_s", 0.0)), 4),
            "sync_ssd_loads": int(stats.get("sync_ssd_loads", 0)),
            "on_demand_loads": int(stats.get("resident_misses", 0)),
            "on_demand_load_time_s": round(float(stats.get("on_demand_load_time_s", 0.0)), 4),
            "fallback_weight_rate": 0.0,
        }

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

    @staticmethod
    def _memory_snapshot() -> dict:
        try:
            from chronos.backend.mac_diagnostics import mlx_memory_snapshot, rss_snapshot

            out = dict(mlx_memory_snapshot())
            out.update(rss_snapshot())
            return out
        except Exception:
            return {}

    @staticmethod
    def _memory_fields(snapshot: dict, suffix: str) -> dict:
        out = {}
        for key in ("rss_gb", "mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb"):
            if key in snapshot:
                prefix = key[:-3] if key.endswith("_gb") else key
                out[f"{prefix}_{suffix}_gb"] = snapshot[key]
        return out
