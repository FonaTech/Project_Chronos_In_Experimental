"""
chronos/runtime/io_simulator.py

Optional SSD-latency injection for the M3 double-stream pipeline benchmark.

In production, expert weight loads block on real NVMe reads (10–50ms typical
for a clustered safetensors mmap on a consumer SSD). On a CPU smoke test the
load is essentially free, which makes the difference between sequential and
overlapped pipelines invisible.

Setting ``CHRONOS_SIM_SSD_MS=30`` causes ``ExpertStore.prefetch_to_ram`` to
sleep for 30ms per cluster load, simulating SSD-bound conditions so we can
verify on CPU that the new ``prefetch_for_next_step + ensure_resident`` path
actually overlaps with ``forward()``.

Defaults to 0 (no injection). The simulator is a no-op when unset, so it
incurs no overhead in normal use.
"""
import os
import time


def simulated_ssd_delay_ms() -> float:
    """Return the configured simulated SSD latency in milliseconds (0 if off)."""
    raw = os.environ.get("CHRONOS_SIM_SSD_MS", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.0


def maybe_sleep(per_unit_ms: float = None) -> None:
    """Sleep for the configured SSD-simulation duration if non-zero.

    `per_unit_ms` allows the caller to scale by units fetched (e.g. multiply
    by number of clusters). When None, sleeps once for the configured value.
    """
    delay = simulated_ssd_delay_ms() if per_unit_ms is None else per_unit_ms
    if delay > 0:
        time.sleep(delay / 1000.0)
