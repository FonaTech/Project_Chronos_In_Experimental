"""
chronos/backend/dispatcher.py

Unified compute-backend dispatcher.

Chronos dispatches across these backend names:

  cpu     — PyTorch CPU (always available)
  cuda    — PyTorch CUDA (NVIDIA GPU)
  mps     — PyTorch Metal Performance Shaders (Apple Silicon via torch)
  mlx     — Apple MLX (Apple Silicon, non-torch inference path in this repo)
  xpu     — PyTorch Intel XPU
  vulkan  — PyTorch Vulkan (only if torch was built with USE_VULKAN=ON)
  opencl  — third-party extension hook (no upstream backend; plug-in only)

Training support in the current repo: cpu, cuda, mps, xpu
Inference support (stock paths):      cpu, cuda, mps, mlx, xpu, (vulkan if built-in)
Inference via ext plugin:             opencl

Important: although MLX is available as an inference backend, this repository
does not currently implement a full MLX-native training stack comparable to
``chronos.trainer.*``. Training resolvers therefore exclude ``mlx`` until a
real MLX trainer exists.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from typing import List, Optional


BACKENDS = ("cuda", "mlx", "mps", "xpu", "vulkan", "opencl", "cpu")

# General runtime auto-detect priority. This covers inference and legacy
# backend selection, where MLX is meaningful on Apple Silicon.
AUTO_PRIORITY = ("mlx", "cuda", "xpu", "mps", "vulkan", "opencl", "cpu")

# Training must only pick backends with an actual training implementation in
# this repository. Keep this separate from AUTO_PRIORITY so inference can
# still prefer MLX while training stays honest.
TRAINING_AUTO_PRIORITY = ("cuda", "xpu", "mps", "cpu")
TRAINING_BACKENDS = ("cuda", "xpu", "mps", "cpu")


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
        name="mlx", available=avail, supports_training=False,
        supports_amp=avail, torch_device=None,
        notes="non-torch backend; Chronos uses chronos.mlx.* inference paths",
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

    def training_available(self) -> List[str]:
        """Returns trainable backend names in training-priority order."""
        return [
            n for n in TRAINING_AUTO_PRIORITY
            if self.info(n).available and self.info(n).supports_training
        ]

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

    def select_training(self, prefer: Optional[str] = None) -> str:
        """Resolve a concrete training backend name.

        Resolution order:
          1. ``CHRONOS_TRAIN_BACKEND`` env var if set.
          2. ``prefer`` argument when available and trainable.
          3. First available backend in training priority order.
          4. ``cpu``.

        ``prefer`` may be ``None`` or ``"auto"`` to request automatic
        selection. Non-trainable values such as ``mlx`` are ignored here.
        """
        env = os.environ.get("CHRONOS_TRAIN_BACKEND")
        if env:
            env = env.strip().lower()
            if env == "auto":
                env = ""
            elif env in BACKENDS and self.info(env).available and self.info(env).supports_training:
                return env
            elif env:
                print(f"[chronos.backend] CHRONOS_TRAIN_BACKEND={env} not available for training; auto-selecting.")

        prefer = (prefer or "").strip().lower()
        if prefer == "auto":
            prefer = ""
        if prefer and prefer in BACKENDS:
            info = self.info(prefer)
            if info.available and info.supports_training:
                return prefer

        for n in TRAINING_AUTO_PRIORITY:
            info = self.info(n)
            if info.available and info.supports_training:
                return n
        return "cpu"

    def device_str(self, name: str) -> Optional[str]:
        """PyTorch device string for a backend, or None for non-torch backends."""
        return self.info(name).torch_device

    def training_device_str(self, name: Optional[str] = None) -> str:
        """PyTorch device string for a training backend."""
        backend = self.select_training(name)
        return self.info(backend).torch_device or "cpu"

    def resolve_training_device(self, prefer: Optional[str] = None) -> tuple[str, str]:
        """Resolve ``(backend, torch_device)`` for training.

        Accepts backend-level requests such as ``auto`` / ``cuda`` / ``mps``
        as well as explicit torch device strings like ``cuda:0``.
        """
        raw = (prefer or "").strip()
        name = raw.lower()

        explicit_map = {
            "cuda:": "cuda",
            "xpu:": "xpu",
        }
        for prefix, backend in explicit_map.items():
            if name.startswith(prefix):
                info = self.info(backend)
                if info.available and info.supports_training:
                    return backend, raw
                print(f"[chronos.backend] requested training device {raw} is not available; auto-selecting.")
                chosen = self.select_training()
                return chosen, self.info(chosen).torch_device or "cpu"

        if name in {"cpu", "mps", "cuda", "xpu"}:
            chosen = self.select_training(name)
            if chosen == name:
                return chosen, self.info(chosen).torch_device or "cpu"

        chosen = self.select_training(name or None)
        return chosen, self.info(chosen).torch_device or "cpu"

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

    def describe_training(self) -> str:
        """Human-readable summary for trainable backends only."""
        lines = ["Chronos training backends:"]
        for n in TRAINING_AUTO_PRIORITY:
            i = self.info(n)
            marker = "✓" if (i.available and i.supports_training) else "·"
            extra = f"  — {i.notes}" if i.notes else ""
            lines.append(f"  {marker} {n:<8} train      dev={i.torch_device or '-':<6}{extra}")
        return "\n".join(lines)


# Module-level convenience singleton
_default = BackendDispatcher()


def available() -> List[str]:
    return _default.available()


def select(prefer: Optional[str] = None) -> str:
    return _default.select(prefer)


def training_available() -> List[str]:
    return _default.training_available()


def select_training(prefer: Optional[str] = None) -> str:
    return _default.select_training(prefer)


def device_str(name: Optional[str] = None) -> Optional[str]:
    return _default.device_str(name or select())


def training_device_str(name: Optional[str] = None) -> str:
    return _default.training_device_str(name)


def resolve_training_device(prefer: Optional[str] = None) -> tuple[str, str]:
    return _default.resolve_training_device(prefer)


def describe() -> str:
    return _default.describe()


def describe_training() -> str:
    return _default.describe_training()
