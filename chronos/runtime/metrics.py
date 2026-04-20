"""
chronos/runtime/metrics.py

A tiny in-process metrics bus. ExpertStore and AsyncPrefetcher push events
into the singleton bus; the WebUI's IO Monitor tab polls ``snapshot()`` and
plots the rolling buffers.

Design constraints:

- **Zero deps** — pure stdlib. No Prometheus, no opentelemetry.
- **Thread-safe** — both the prefetcher daemon and the main thread emit.
- **Bounded memory** — each metric is a deque(maxlen=600) so a 1-Hz UI
  poll over a 10-minute window stays under a few KB.
- **Optional** — emitting is wrapped in a try/except so missing
  ``chronos.runtime.metrics`` cannot crash the inference loop.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, List, Tuple


_BUFFER_SIZE = 600  # ~10 min at 1 Hz


class MetricsBus:
    def __init__(self):
        self._lock = threading.Lock()
        # metric_name -> deque[(timestamp, value)]
        self._buffers: Dict[str, Deque[Tuple[float, float]]] = {}

    def record(self, name: str, value: float, ts: float = None) -> None:
        if ts is None:
            ts = time.monotonic()
        with self._lock:
            buf = self._buffers.get(name)
            if buf is None:
                buf = deque(maxlen=_BUFFER_SIZE)
                self._buffers[name] = buf
            buf.append((float(ts), float(value)))

    def latest(self, name: str) -> float:
        with self._lock:
            buf = self._buffers.get(name)
            if not buf:
                return float("nan")
            return buf[-1][1]

    def series(self, name: str) -> List[Tuple[float, float]]:
        with self._lock:
            buf = self._buffers.get(name)
            return list(buf) if buf else []

    def snapshot(self) -> Dict[str, List[Tuple[float, float]]]:
        with self._lock:
            return {k: list(v) for k, v in self._buffers.items()}

    def reset(self) -> None:
        with self._lock:
            self._buffers.clear()


# Module-level singleton. Importing this module is essentially free.
bus = MetricsBus()


def safe_record(name: str, value: float) -> None:
    """Convenience wrapper that swallows any exception so metrics emission
    can never break the hot path."""
    try:
        bus.record(name, value)
    except Exception:
        pass
