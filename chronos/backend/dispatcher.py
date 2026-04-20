"""
chronos/backend/dispatcher.py

Unified compute-backend dispatcher.

Chronos dispatches across six backend names:

  cpu     — PyTorch CPU (always available)
  cuda    — PyTorch CUDA (NVIDIA GPU)
  mps     — PyTorch Metal Performance Shaders (Apple Silicon via torch)
  mlx     — Apple MLX (Apple Silicon, unified memory; non-torch)
  vulkan  — PyTorch Vulkan (only if torch was built with USE_VULKAN=ON)
  opencl  — third-party extension hook (no upstream backend; plug-in only)

Training support:                cpu, cuda, mps, mlx
Inference support (stock torch): cpu, cuda, mps, mlx, (vulkan if built-in)
Inference via ext plugin:        opencl  (requires chronos.backend.ext.opencl
                                         implementation; stub returns False)

The Vulkan and OpenCL hooks exist so that someone running a custom PyTorch
build (or a custom kernel) can plug in without modifying core Chronos code.
On stock pip-installed PyTorch, they report "not available" and fall back.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from typing import List, Optional


BACKENDS = ("cuda", "mlx", "mps", "xpu", "vulkan", "opencl", "cpu")

# Auto-detect priority. mlx > cuda > mps > xpu > vulkan > opencl > cpu.
# The intent: prefer whichever backend has the most optimized training path
# on the host. On Apple Silicon, mlx has unified memory advantages that
# mps (via torch) cannot match; on NVIDIA, cuda is always best.
AUTO_PRIORITY = ("mlx", "cuda", "xpu", "mps", "vulkan", "opencl", "cpu")


@dataclass
class BackendInfo:
    name: str
    available: bool
    supports_training: bool
    supports_amp: bool
    torch_device: Optional[str]
    notes: str = ""


def _probe_cuda() -> BackendInfo:
    try:
        import torch
        avail = bool(torch.cuda.is_available())
    except Exception:
        avail = False
    return BackendInfo(
        name="cuda", available=avail, supports_training=avail,
        supports_amp=avail, torch_device="cuda" if avail else None,
    )


def _probe_mps() -> BackendInfo:
    try:
        import torch
        avail = bool(torch.backends.mps.is_available())
    except Exception:
        avail = False
    return BackendInfo(
        name="mps", available=avail, supports_training=avail,
        # fp16 autocast on MPS is still patchy across torch versions; bf16
        # generally OK. We report True but document the caveat.
        supports_amp=avail,
        torch_device="mps" if avail else None,
        notes="bf16 autocast preferred over fp16 on MPS" if avail else "",
    )


def _probe_xpu() -> BackendInfo:
    """Intel XPU (torch.xpu). Only present on torch with oneAPI."""
    try:
        import torch
        avail = hasattr(torch, "xpu") and bool(torch.xpu.is_available())
    except Exception:
        avail = False
    return BackendInfo(
        name="xpu", available=avail, supports_training=avail,
        supports_amp=avail, torch_device="xpu" if avail else None,
    )


def _probe_mlx() -> BackendInfo:
    spec = importlib.util.find_spec("mlx")
    if spec is None:
        return BackendInfo("mlx", False, False, False, None,
                           notes="mlx not installed")
    try:
        import mlx.core as mx
        avail = bool(mx.metal.is_available())
    except Exception:
        avail = False
    return BackendInfo(
        name="mlx", available=avail, supports_training=avail,
        supports_amp=avail, torch_device=None,
        notes="non-torch backend; Chronos uses chronos.mlx.* paths",
    )


def _probe_vulkan() -> BackendInfo:
    """PyTorch Vulkan is only present in custom builds (USE_VULKAN=ON).
    The stock wheel does not include it. When present, training autograd
    is NOT supported upstream — only inference forward."""
    try:
        import torch
        check = getattr(torch, "is_vulkan_available", None)
        avail = bool(check()) if check is not None else False
    except Exception:
        avail = False
    return BackendInfo(
        name="vulkan", available=avail,
        supports_training=False,  # upstream torch.vulkan has no autograd
        supports_amp=False,
        torch_device="vulkan" if avail else None,
        notes="inference-only; requires custom torch build",
    )


def _probe_opencl() -> BackendInfo:
    """Extension hook. A third-party module can register an OpenCL backend by
    placing ``chronos.backend.ext.opencl.PROBE()`` that returns a BackendInfo
    with available=True. The bundled stub returns False."""
    try:
        from chronos.backend.ext import opencl as opencl_ext
        info = opencl_ext.PROBE()
        if not isinstance(info, BackendInfo):
            info = BackendInfo("opencl", False, False, False, None,
                               notes="ext.opencl.PROBE() malformed")
        return info
    except Exception as e:
        return BackendInfo("opencl", False, False, False, None,
                           notes=f"ext not available: {e}")


_PROBES = {
    "cuda": _probe_cuda, "mps": _probe_mps, "xpu": _probe_xpu,
    "mlx": _probe_mlx, "vulkan": _probe_vulkan, "opencl": _probe_opencl,
    "cpu": lambda: BackendInfo("cpu", True, True, False, "cpu"),
}


class BackendDispatcher:
    """Central entry point for backend selection across Chronos."""

    def __init__(self):
        self._cache: dict[str, BackendInfo] = {}

    def info(self, name: str) -> BackendInfo:
        if name not in self._cache:
            probe = _PROBES.get(name)
            if probe is None:
                raise ValueError(f"unknown backend: {name}")
            self._cache[name] = probe()
        return self._cache[name]

    def available(self) -> List[str]:
        """Returns names in priority order, filtered to those actually usable."""
        return [n for n in AUTO_PRIORITY if self.info(n).available]

    def select(self, prefer: Optional[str] = None) -> str:
        """Resolve a concrete backend name.

        Resolution order:
          1. ``CHRONOS_BACKEND`` env var if set.
          2. ``prefer`` argument if usable.
          3. First available in auto-priority order.
          4. ``cpu`` (always available).
        """
        env = os.environ.get("CHRONOS_BACKEND")
        if env:
            if self.info(env).available:
                return env
            # Fall through with warning; don't crash.
            print(f"[chronos.backend] CHRONOS_BACKEND={env} not available; auto-selecting.")
        if prefer and self.info(prefer).available:
            return prefer
        for n in AUTO_PRIORITY:
            if self.info(n).available:
                return n
        return "cpu"

    def device_str(self, name: str) -> Optional[str]:
        """PyTorch device string for a backend, or None for non-torch backends."""
        return self.info(name).torch_device

    def supports_training(self, name: str) -> bool:
        return self.info(name).supports_training

    def supports_amp(self, name: str) -> bool:
        return self.info(name).supports_amp

    def notes(self, name: str) -> str:
        return self.info(name).notes

    def describe(self) -> str:
        """Human-readable summary, useful for --help and the WebUI."""
        lines = ["Chronos compute backends:"]
        for n in AUTO_PRIORITY:
            i = self.info(n)
            marker = "✓" if i.available else "·"
            tr = "train+inf" if i.supports_training else ("infer" if i.available else "n/a")
            extra = f"  — {i.notes}" if i.notes else ""
            lines.append(f"  {marker} {n:<8} {tr:<10} dev={i.torch_device or '-':<6}{extra}")
        return "\n".join(lines)


# Module-level convenience singleton
_default = BackendDispatcher()


def available() -> List[str]:
    return _default.available()


def select(prefer: Optional[str] = None) -> str:
    return _default.select(prefer)


def device_str(name: Optional[str] = None) -> Optional[str]:
    return _default.device_str(name or select())


def describe() -> str:
    return _default.describe()
