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
    device_str,
    describe,
    AUTO_PRIORITY,
)

# Back-compat shim for the pre-M5 API that code elsewhere still calls.
from chronos._backend_legacy import build_model, BackendType  # type: ignore


def get_backend():
    """Back-compat alias for ``select()`` (auto-detected default backend)."""
    return select()


__all__ = [
    "BackendDispatcher", "BackendInfo",
    "available", "select", "device_str", "describe", "AUTO_PRIORITY",
    "get_backend", "build_model", "BackendType",
]
