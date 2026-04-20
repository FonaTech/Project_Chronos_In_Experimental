"""
chronos/io/async_prefetcher.py

Asynchronous expert weight prefetcher driven by LookaheadRouter predictions.

Architecture:
- Background daemon thread runs a prefetch loop
- Main thread submits prefetch requests via a queue
- Prefetcher loads SSD → RAM (pinned) ahead of time
- When main thread needs an expert, it's already warm in RAM → fast H2D

The key insight: LookaheadRouter predicts t+1/t+2 expert needs,
giving us 1-2 decode steps (~10-50ms) of lead time to hide SSD latency.
"""
import threading
import queue
import time
from typing import List, Optional, Callable

from chronos.io.expert_store import ExpertStore

try:
    from chronos.runtime.metrics import safe_record as _metric
except Exception:
    def _metric(name, value):  # fallback
        pass


class AsyncPrefetcher:
    """
    Background thread that prefetches expert weights from SSD → RAM
    based on LookaheadRouter predictions.

    Usage:
        prefetcher = AsyncPrefetcher(expert_store)
        prefetcher.start()

        # After each decode step, submit next predictions:
        prefetcher.submit([expert_id_t1, expert_id_t2])

        # Before using expert at step t:
        prefetcher.wait_for(expert_id_t)  # usually already done

        prefetcher.stop()
    """

    def __init__(
        self,
        expert_store: ExpertStore,
        queue_depth: int = 8,
        log_fn: Optional[Callable] = None,
    ):
        self.store = expert_store
        self.log_fn = log_fn or (lambda m: None)
        self._request_queue: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._done_set: set = set()
        self._done_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stats = {"prefetch_hits": 0, "prefetch_misses": 0, "total_requests": 0}

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            daemon=True,
            name="chronos-prefetcher",
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        # Unblock the queue
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    def submit(self, expert_ids: List[int]):
        """
        Submit a list of expert IDs to prefetch (non-blocking).
        Called after each decode step with LookaheadRouter predictions.
        """
        if not expert_ids:
            return
        try:
            self._request_queue.put_nowait(expert_ids)
        except queue.Full:
            # Drop oldest request to make room
            try:
                self._request_queue.get_nowait()
                self._request_queue.put_nowait(expert_ids)
            except (queue.Empty, queue.Full):
                pass

    def wait_for(self, expert_id: int, timeout: float = 0.05) -> bool:
        """
        Wait until expert_id is in RAM (or VRAM). Returns True if ready.
        In practice this should return immediately if prefetch worked.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (self.store.ram_lru.contains(expert_id) or
                    self.store.vram_lru.contains(expert_id)):
                with self._done_lock:
                    self._stats["prefetch_hits"] += 1
                return True
            time.sleep(0.001)
        with self._done_lock:
            self._stats["prefetch_misses"] += 1
        return False

    def _prefetch_loop(self):
        while not self._stop_event.is_set():
            try:
                expert_ids = self._request_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            if expert_ids is None:  # stop sentinel
                break

            with self._done_lock:
                self._stats["total_requests"] += len(expert_ids)

            # Filter out already-warm experts
            to_fetch = [
                eid for eid in expert_ids
                if not self.store.ram_lru.contains(eid)
                and not self.store.vram_lru.contains(eid)
            ]
            if to_fetch:
                try:
                    self.store.prefetch_to_ram(to_fetch)
                except Exception as e:
                    self.log_fn(f"[prefetcher] error fetching {to_fetch}: {e}")

    @property
    def stats(self) -> dict:
        with self._done_lock:
            s = dict(self._stats)
        total = s["prefetch_hits"] + s["prefetch_misses"]
        s["hit_rate"] = s["prefetch_hits"] / max(total, 1)
        _metric("prefetch_hit_rate", s["hit_rate"])
        return s


class PrefetchScheduler:
    """
    Integrates LookaheadRouter predictions with AsyncPrefetcher.

    Called once per decode step to:
    1. Extract predicted expert IDs from lookahead_probs
    2. Submit them to the prefetcher
    3. Promote the current-step expert from RAM → VRAM
    """

    def __init__(self, prefetcher: AsyncPrefetcher, expert_store: ExpertStore):
        self.prefetcher = prefetcher
        self.store = expert_store

    def step(
        self,
        lookahead_probs,  # [B, S, K+1, E] from LookaheadRouter
        current_step_expert_ids: List[int],
    ):
        """
        Legacy single-call API: promote current experts (blocking), then
        submit future predictions. Kept for backward compatibility.
        Prefer `promote_current` + `prefetch_only` for the M3 pipeline.
        """
        self.promote_current(current_step_expert_ids, blocking=True)
        self.prefetch_only(lookahead_probs)

    def promote_current(
        self,
        current_step_expert_ids: List[int],
        blocking: bool = False,
    ) -> List[int]:
        """
        Move the experts needed right now from RAM → VRAM.

        With ``blocking=False`` (M3 default) the H2D copies are issued on
        the dedicated stream and return immediately; the caller is
        responsible for waiting via ``expert_store.wait_for_experts``.
        Returns the list of experts that were actually newly promoted
        (already-resident experts are skipped).
        """
        newly = []
        for eid in current_step_expert_ids:
            if not self.store.vram_lru.contains(eid):
                if self.store.promote_to_vram(eid, blocking=blocking):
                    newly.append(eid)
        return newly

    def prefetch_only(self, lookahead_probs) -> List[int]:
        """
        Submit future expert predictions to the async prefetcher.
        Does NOT touch VRAM. Safe to call *before* the forward pass so the
        H2D pipeline can overlap with compute.

        Returns the list of unique expert IDs submitted.
        """
        if lookahead_probs is None:
            return []
        import torch
        # lookahead_probs: [B, S, K+1, E] — take last token, steps 1..K.
        # We submit t+1 first (highest priority), then t+2, etc., so the
        # FIFO queue naturally orders them by recency.
        last_token_probs = lookahead_probs[:, -1, 1:, :]  # [B, K, E]
        K = last_token_probs.shape[1]
        all_submitted: List[int] = []
        for k in range(K):
            ids_k = last_token_probs[:, k, :].argmax(dim=-1).unique().cpu().tolist()
            if ids_k:
                self.prefetcher.submit(ids_k)
                all_submitted.extend(ids_k)
        return list(dict.fromkeys(all_submitted))  # preserve order, dedup
