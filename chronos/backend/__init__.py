"""
chronos/backend/__init__.py

Public re-exports for the backend dispatcher. Also preserves the legacy
``get_backend()`` / ``build_model()`` API from the pre-M5 single-file
``chronos/backend.py`` so existing UI and CLI code keeps working.
"""
from chronos.backend.dispatcher import (
    BackendDispatcher,
    BackendInfo,
    available,
    select,
    training_available,
    select_training,
    device_str,
    training_device_str,
    resolve_training_device,
    describe,
    describe_training,
    AUTO_PRIORITY,
    TRAINING_AUTO_PRIORITY,
)
from chronos.backend.mac_diagnostics import (
    mac_backend_diagnostics,
    mlx_memory_snapshot,
    mps_memory_snapshot,
    rss_snapshot,
)

# Back-compat shim for the pre-M5 API that code elsewhere still calls.
from chronos._backend_legacy import build_model, BackendType  # type: ignore


def get_backend():
    """Back-compat alias for ``select()`` (auto-detected default backend)."""
    return select()


__all__ = [
    "BackendDispatcher", "BackendInfo",
    "available", "select", "training_available", "select_training",
    "device_str", "training_device_str", "resolve_training_device",
    "describe", "describe_training",
    "mac_backend_diagnostics", "mlx_memory_snapshot", "mps_memory_snapshot", "rss_snapshot",
    "AUTO_PRIORITY", "TRAINING_AUTO_PRIORITY",
    "get_backend", "build_model", "BackendType",
]
