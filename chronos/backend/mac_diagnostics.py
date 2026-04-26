"""Apple Silicon diagnostics for Chronos backends."""
from __future__ import annotations

import os
import platform
from typing import Any

import torch

from chronos.trainer.device_utils import (
    configure_cpu_threads,
    cpu_thread_snapshot,
)


def mlx_memory_snapshot() -> dict[str, float | bool | str]:
    out: dict[str, float | bool | str] = {
        "mlx_available": False,
        "mlx_metal_available": False,
    }
    try:
        import mlx
        import mlx.core as mx

        out["mlx_available"] = True
        out["mlx_version"] = getattr(mlx, "__version__", "unknown")
        out["mlx_metal_available"] = bool(mx.metal.is_available())
        for new_name, old_name, key in [
            ("get_active_memory", "get_active_memory", "mlx_active_gb"),
            ("get_cache_memory", "get_cache_memory", "mlx_cache_gb"),
            ("get_peak_memory", "get_peak_memory", "mlx_peak_gb"),
        ]:
            fn = getattr(mx, new_name, None) or getattr(mx.metal, old_name, None)
            if fn is not None:
                try:
                    out[key] = round(float(fn()) / (1024 ** 3), 6)
                except Exception:
                    pass
    except Exception as exc:
        out["mlx_error"] = str(exc)
    return out


def mps_memory_snapshot() -> dict[str, float | bool | str]:
    out: dict[str, float | bool | str] = {
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    }
    if not out["mps_available"]:
        return out
    for name, key in [
        ("current_allocated_memory", "mps_allocated_gb"),
        ("driver_allocated_memory", "mps_driver_allocated_gb"),
        ("recommended_max_memory", "mps_recommended_max_gb"),
    ]:
        fn = getattr(torch.mps, name, None)
        if fn is not None:
            try:
                out[key] = round(float(fn()) / (1024 ** 3), 6)
            except Exception:
                pass
    return out


def rss_snapshot() -> dict[str, float]:
    try:
        import psutil

        proc = psutil.Process()
        vm = psutil.virtual_memory()
        return {
            "rss_gb": round(proc.memory_info().rss / (1024 ** 3), 6),
            "system_available_gb": round(vm.available / (1024 ** 3), 6),
            "system_total_gb": round(vm.total / (1024 ** 3), 6),
        }
    except Exception:
        return {"rss_gb": 0.0}


def _configure_cpu_threads() -> int:
    return configure_cpu_threads(
        os.environ.get("CHRONOS_CPU_THREADS", "auto"),
        budget_percent=os.environ.get("CHRONOS_CPU_BUDGET_PERCENT", 100),
        prefer_env=True,
    )


def mac_backend_diagnostics(configure_threads: bool = False) -> dict[str, Any]:
    if configure_threads:
        _configure_cpu_threads()
    thread_diag = cpu_thread_snapshot()
    try:
        physical = os.cpu_count() or 1
        import psutil

        physical = psutil.cpu_count(logical=False) or physical
        logical = psutil.cpu_count(logical=True) or os.cpu_count() or physical
    except Exception:
        logical = os.cpu_count() or physical

    out: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_num_threads": int(thread_diag.get("torch_num_threads") or 1),
        "torch_num_interop_threads": int(thread_diag.get("torch_num_interop_threads") or 1),
        "physical_cores": int(physical),
        "logical_cores": int(logical),
        "env": {
            "OMP_NUM_THREADS": thread_diag.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": thread_diag.get("MKL_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": thread_diag.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": thread_diag.get("NUMEXPR_NUM_THREADS"),
            "CHRONOS_CPU_THREADS": thread_diag.get("CHRONOS_CPU_THREADS"),
            "CHRONOS_CPU_BUDGET_PERCENT": thread_diag.get("CHRONOS_CPU_BUDGET_PERCENT"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        },
    }
    out.update(rss_snapshot())
    out.update(mps_memory_snapshot())
    out.update(mlx_memory_snapshot())
    return out
