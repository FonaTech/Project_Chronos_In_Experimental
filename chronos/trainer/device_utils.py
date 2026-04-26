"""Backend-aware training/runtime helpers.

The original training code treated every non-CUDA device as CPU. On Apple
Silicon that disables MPS autocast and makes CPU benchmarks misleading when
the shell exports OMP_NUM_THREADS=1. Keep the backend policy in one place so
all stages behave consistently.
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import Any

TorchDevice = Any
TorchDType = Any

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)

_LAST_THREAD_CONFIG: dict[str, Any] = {}


def _torch():
    import torch

    return torch


def torch_device_type(device: str | TorchDevice | None) -> str:
    raw = str(device or "cpu").lower()
    if raw.startswith("cuda"):
        return "cuda"
    if raw.startswith("mps"):
        return "mps"
    if raw.startswith("xpu"):
        return "xpu"
    if raw.startswith("mlx"):
        return "mlx"
    return "cpu"


def torch_dtype_from_name(name: str | None) -> TorchDType:
    torch = _torch()
    value = (name or "float32").strip().lower()
    if value == "auto":
        return torch.float32
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def resolve_dtype_name(device: str | TorchDevice | None, dtype_name: str | None) -> str:
    """Resolve user-facing dtype policy to a concrete dtype string.

    `auto` is intentionally backend-specific. Apple MPS/MLX default to bf16:
    it keeps the fp32 exponent range and avoids the fp16 overflows that show up
    in router softmax, CE, and Adam moments. CUDA/XPU keep the existing fp16
    default, while CPU stays fp32 unless the user explicitly requests fp16/bf16.
    """
    value = (dtype_name or "auto").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return "float16"
    if value in {"bf16", "bfloat16"}:
        return "bfloat16"
    if value in {"fp32", "float32", "full"}:
        return "float32"
    device_type = torch_device_type(device)
    if device_type in {"mps", "mlx"}:
        return "bfloat16"
    if device_type in {"cuda", "xpu"}:
        return "float16"
    return "float32"


def autocast_context(device: str | TorchDevice | None, dtype_name: str | None):
    """Return an autocast context suitable for CPU/CUDA/MPS/XPU.

    MPS supports torch.autocast, but GradScaler is not used there. If the user
    requests float32, autocast is disabled explicitly.
    """
    torch = _torch()
    device_type = torch_device_type(device)
    dtype = torch_dtype_from_name(resolve_dtype_name(device, dtype_name))
    if device_type == "mlx" or dtype == torch.float32:
        return contextlib.nullcontext()
    if device_type == "mps" and not hasattr(torch.backends, "mps"):
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type=device_type, dtype=dtype)
    except Exception:
        return contextlib.nullcontext()


def grad_scaler(device: str | TorchDevice | None, dtype_name: str | None):
    """GradScaler is CUDA-fp16 only for this project.

    MPS uses autocast without scaler; enabling CUDA's legacy scaler on MPS
    silently pushes the stage back toward CPU-like behavior or warnings.
    """
    torch = _torch()
    enabled = (
        torch_device_type(device) == "cuda"
        and torch.cuda.is_available()
        and torch_dtype_from_name(resolve_dtype_name(device, dtype_name)) == torch.float16
    )
    return torch.amp.GradScaler("cuda", enabled=enabled)


def optimizer_step_with_scaler(
    scaler,
    optimizer,
    parameters,
    grad_clip: float,
) -> None:
    torch = _torch()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _physical_cores() -> int:
    try:
        import psutil

        return int(psutil.cpu_count(logical=False) or os.cpu_count() or 1)
    except Exception:
        return int(os.cpu_count() or 1)


def _thread_env_snapshot() -> dict[str, str | None]:
    snap = {
        key: os.environ.get(key)
        for key in THREAD_ENV_KEYS
    }
    snap.update({
        "CHRONOS_CPU_THREADS": os.environ.get("CHRONOS_CPU_THREADS"),
        "CHRONOS_CPU_BUDGET_PERCENT": os.environ.get("CHRONOS_CPU_BUDGET_PERCENT"),
    })
    return snap


def _auto_requested(value: int | str | None) -> bool:
    return value in (None, "", "auto", 0, "0")


def _parse_threads(value: int | float | str | None) -> int:
    return max(1, int(float(str(value).strip())))


def _resolve_cpu_threads(
    requested: int | str | None = None,
    *,
    budget_percent: int | float | str | None = None,
    prefer_env: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Resolve the effective CPU thread count without importing torch.

    `prefer_env=False` is intentional for the WebUI/training path: Config tab
    values should win over stale shell exports such as CHRONOS_CPU_THREADS=1.
    Explicit `requested` values still win in all modes.
    """
    physical = _physical_cores()
    source = "requested"
    if not _auto_requested(requested):
        threads = _parse_threads(requested)
        pct = None
    else:
        env_threads = (os.environ.get("CHRONOS_CPU_THREADS") or "").strip().lower()
        if prefer_env and env_threads and env_threads != "auto":
            threads = _parse_threads(env_threads)
            pct = None
            source = "CHRONOS_CPU_THREADS"
        else:
            pct = budget_percent
            if pct in (None, "", "auto"):
                pct = (
                    os.environ.get("CHRONOS_CPU_BUDGET_PERCENT", 100)
                    if prefer_env
                    else 100
                )
            pct = max(1.0, min(100.0, float(pct)))
            threads = max(1, int(round(physical * pct / 100.0)))
            source = "budget_percent"
    return threads, {
        "requested": requested,
        "budget_percent": pct,
        "prefer_env": bool(prefer_env),
        "source": source,
        "physical_cores": physical,
    }


def configure_cpu_thread_env(
    requested: int | str | None = None,
    *,
    budget_percent: int | float | str | None = None,
    prefer_env: bool = False,
) -> int:
    """Set BLAS/OpenMP-style thread env vars before torch/numpy import."""
    threads, meta = _resolve_cpu_threads(
        requested,
        budget_percent=budget_percent,
        prefer_env=prefer_env,
    )
    for key in THREAD_ENV_KEYS:
        os.environ[key] = str(threads)
    _LAST_THREAD_CONFIG.clear()
    _LAST_THREAD_CONFIG.update({"threads": threads, **meta})
    return int(threads)


def configure_cpu_threads(
    requested: int | str | None = None,
    *,
    budget_percent: int | float | str | None = None,
    prefer_env: bool = False,
) -> int:
    """Set a sane PyTorch CPU thread count unless the user chooses one.

    Chronos often runs on Apple Silicon where shell profiles may export
    OMP_NUM_THREADS=1. That is fine for some libraries but makes CPU training
    and inference comparisons invalid here.
    """
    torch = _torch()
    threads = configure_cpu_thread_env(
        requested,
        budget_percent=budget_percent,
        prefer_env=prefer_env,
    )

    # Keep PyTorch and BLAS/OpenMP-style libraries aligned. Many macOS shells
    # export OMP_NUM_THREADS=1; if we only call torch.set_num_threads(), parts
    # of the WebUI process and DataLoader workers can still inherit the stale
    # single-thread environment.
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        # This can only be called before parallel work starts. If torch or a
        # dependency already initialized inter-op pools, keep the current value.
        interop = max(1, min(threads, _physical_cores()))
        torch.set_num_interop_threads(interop)
    except Exception:
        pass
    return int(torch.get_num_threads() or threads)


def cpu_thread_snapshot() -> dict[str, Any]:
    """Return thread settings for logs/UI diagnostics."""
    torch = _torch()
    snap: dict[str, Any] = {
        "torch_num_threads": int(torch.get_num_threads() or 1),
        "torch_num_interop_threads": int(torch.get_num_interop_threads() or 1),
        "physical_cores": _physical_cores(),
    }
    snap.update(_thread_env_snapshot())
    for key, value in _LAST_THREAD_CONFIG.items():
        snap[f"last_{key}"] = value
    return snap


def dataloader_kwargs(
    device: str | TorchDevice | None = "cpu",
    num_workers: int | str | None = None,
    shuffle: bool = True,
) -> dict[str, Any]:
    device_type = torch_device_type(device)
    metal_backend = device_type in {"mps", "mlx"}
    allow_metal_workers = (
        os.environ.get("CHRONOS_ALLOW_METAL_DATALOADER_WORKERS", "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    force_single_process = (
        sys.platform == "darwin"
        and metal_backend
        and not allow_metal_workers
    )
    if num_workers in (None, "", "auto"):
        workers = 0 if force_single_process else max(1, min(4, _physical_cores() // 4))
        if device_type == "xpu":
            workers = min(workers, 4)
    else:
        workers = max(0, int(num_workers))
        if force_single_process:
            workers = 0
    return {
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device_type == "cuda",
        "persistent_workers": workers > 0,
    }


def backend_memory_snapshot(device: str | TorchDevice | None = None) -> dict[str, float | str]:
    torch = _torch()
    out: dict[str, float | str] = {"torch_device_type": torch_device_type(device)}
    try:
        import psutil

        out["rss_gb"] = round(psutil.Process().memory_info().rss / (1024 ** 3), 6)
    except Exception:
        out["rss_gb"] = 0.0

    dtype = torch_device_type(device)
    if dtype == "mps" and hasattr(torch, "mps"):
        for name, key in [
            ("current_allocated_memory", "mps_allocated_gb"),
            ("driver_allocated_memory", "mps_driver_allocated_gb"),
        ]:
            fn = getattr(torch.mps, name, None)
            if fn is not None:
                try:
                    out[key] = round(float(fn()) / (1024 ** 3), 6)
                except Exception:
                    pass
    if dtype == "cuda" and torch.cuda.is_available():
        try:
            out["cuda_allocated_gb"] = round(torch.cuda.memory_allocated(device) / (1024 ** 3), 6)
            out["cuda_reserved_gb"] = round(torch.cuda.memory_reserved(device) / (1024 ** 3), 6)
        except Exception:
            pass
    return out


@dataclass
class DeviceRuntimeSummary:
    device: str
    device_type: str
    dtype: str
    cpu_threads: int
    autocast: bool
    scaler: bool


def runtime_summary(device: str | TorchDevice | None, dtype_name: str | None) -> DeviceRuntimeSummary:
    torch = _torch()
    device_type = torch_device_type(device)
    resolved = resolve_dtype_name(device, dtype_name)
    dtype = torch_dtype_from_name(resolved)
    return DeviceRuntimeSummary(
        device=str(device or "cpu"),
        device_type=device_type,
        dtype=str(dtype).replace("torch.", ""),
        cpu_threads=int(torch.get_num_threads() or 1),
        autocast=(device_type in {"cpu", "cuda", "mps", "xpu"} and dtype != torch.float32),
        scaler=(device_type == "cuda" and dtype == torch.float16),
    )
