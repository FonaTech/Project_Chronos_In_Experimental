"""Opt-in OpenCL backend plugin stub.

A real implementation should define::

    from chronos.backend.dispatcher import BackendInfo
    def PROBE() -> BackendInfo: ...
    def to_device(tensor, ...): ...

The bundled stub always reports 'unavailable'. Third parties can replace
this file (or monkey-patch the module) to plug in their own OpenCL backend
without modifying Chronos core.
"""
from chronos.backend.dispatcher import BackendInfo


def PROBE() -> BackendInfo:
    return BackendInfo(
        name="opencl", available=False,
        supports_training=False, supports_amp=False,
        torch_device=None,
        notes="stub; install chronos-opencl-plugin or provide a custom ext module",
    )
