"""
chronos/serving/vllm_adapter.py

Optional vLLM serving adapter. ``vllm`` is NOT a requirement of Chronos
and the module does the right thing whether or not it's installed.

Usage::

    from chronos.serving import register_chronos_with_vllm
    register_chronos_with_vllm()  # no-op if vllm not installed

When ``vllm`` is available, ``ChronosForCausalLM`` is registered in
``vllm.ModelRegistry`` so vLLM can find the architecture by name. The
actual worker-side glue (paged KV cache interop, per-layer expert mask
hook) is version-sensitive; see ``docs/vllm_integration.md`` for the
full integration steps and known gotchas.
"""
from __future__ import annotations

import importlib.util
from typing import Any, Optional


HAS_VLLM = importlib.util.find_spec("vllm") is not None


def register_chronos_with_vllm(verbose: bool = True) -> bool:
    """Register ``ChronosForCausalLM`` with vLLM's ModelRegistry.

    Returns True on success, False if vLLM is not installed or the API
    shape has changed. This function is idempotent.
    """
    if not HAS_VLLM:
        if verbose:
            print(
                "[chronos.serving] vllm not installed — skipping registration.\n"
                "  To enable vLLM serving: `pip install vllm`\n"
                "  (Linux + CUDA required; see docs/vllm_integration.md)"
            )
        return False

    try:
        import vllm  # noqa: F401
        from vllm import ModelRegistry

        from chronos.model.model_chronos import ChronosForCausalLM

        # ModelRegistry.register_model signature changes across vLLM
        # versions. Try the common shapes.
        arch = "ChronosForCausalLM"
        if hasattr(ModelRegistry, "register_model"):
            try:
                ModelRegistry.register_model(arch, ChronosForCausalLM)
            except TypeError:
                # Older API: register_model(name, module_path, class_name)
                ModelRegistry.register_model(
                    arch, "chronos.model.model_chronos", "ChronosForCausalLM",
                )
        else:
            if verbose:
                print("[chronos.serving] vllm.ModelRegistry API not recognized; skipping.")
            return False

        if verbose:
            print(f"[chronos.serving] Registered {arch} with vLLM ModelRegistry.")
        return True
    except Exception as e:
        if verbose:
            print(f"[chronos.serving] vLLM registration failed: {e}")
        return False


def set_available_expert_masks(model, masks) -> None:
    """Hook point for vLLM worker to inject per-step expert availability
    masks. Sets ``model._chronos_available_expert_masks`` so the next
    forward picks it up (if the model is wired to check there).

    This is a best-effort placeholder; the actual decode-loop glue depends
    on the vLLM version and is documented in docs/vllm_integration.md.
    """
    setattr(model, "_chronos_available_expert_masks", masks)


def is_available() -> bool:
    return HAS_VLLM
