"""
chronos/backend.py — Unified backend selector.

Usage:
    from chronos.backend import build_model, get_backend

    backend = get_backend()   # "mlx" | "cuda" | "cpu"
    model, engine = build_model(config, backend=backend)
"""
from __future__ import annotations
import importlib
from typing import Literal

BackendType = Literal["mlx", "cuda", "mps", "cpu"]


def get_backend() -> BackendType:
    """
    Auto-detect best available backend.
    Priority: MLX (Apple Silicon) > CUDA > MPS > CPU
    """
    # Try MLX first (Apple Silicon unified memory)
    if importlib.util.find_spec("mlx") is not None:
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return "mlx"
        except Exception:
            pass

    import torch
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_model(config, backend: BackendType = None, ssd_dir: str = "./expert_cache"):
    """
    Build ChronosModel + InferenceEngine for the given backend.

    Returns (model, inference_engine).
    """
    if backend is None:
        backend = get_backend()

    if backend == "mlx":
        from chronos.mlx.model import ChronosMLXModel
        from chronos.mlx.inference import ChronosMLXInferenceEngine
        model = ChronosMLXModel(config)
        engine = ChronosMLXInferenceEngine(model, config, ssd_dir=ssd_dir)
        return model, engine
    else:
        import torch
        from chronos.model.model_chronos import ChronosForCausalLM
        from chronos.runtime.inference_engine import ChronosInferenceEngine
        device_map = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}[backend]
        model = ChronosForCausalLM(config).to(device_map)
        engine = ChronosInferenceEngine(model, config, ssd_dir=ssd_dir)
        return model, engine
