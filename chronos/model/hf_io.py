"""
chronos/model/hf_io.py

HuggingFace save/load helpers for ChronosForCausalLM.

``ChronosForCausalLM`` already inherits ``transformers.PreTrainedModel`` so
``.from_pretrained()`` / ``.save_pretrained()`` technically work. The pieces
this module adds:

1. Register ``ChronosConfig`` with ``AutoConfig`` and ``ChronosForCausalLM``
   with ``AutoModelForCausalLM`` so users can do
   ``AutoModelForCausalLM.from_pretrained("path/to/chronos")`` without
   ``trust_remote_code=True``.

2. ``save_chronos_pretrained`` / ``load_chronos_pretrained`` helpers that
   also preserve the Chronos expert-cache layout (``cluster_manifest.json``
   alongside the ``.ctsr`` cluster files) so a round-trip keeps the
   offloaded SSD setup intact.

Importing this module is side-effect-ful: registration happens at import
time. ``chronos/__init__.py`` imports it so ordinary usage just works.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

import torch


def _register_auto_classes():
    """Register Chronos classes with Transformers AutoXXX factories."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception:
        return
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM

    # Idempotent: Transformers raises if we try to register the same config
    # twice. We catch and ignore.
    try:
        AutoConfig.register("chronos", ChronosConfig)
    except (ValueError, KeyError):
        pass
    try:
        AutoModelForCausalLM.register(ChronosConfig, ChronosForCausalLM)
    except (ValueError, KeyError):
        pass


_register_auto_classes()


def save_chronos_pretrained(
    model,
    save_directory: str,
    expert_cache_dir: Optional[str] = None,
    safe_serialization: bool = True,
) -> None:
    """Save a Chronos model in HF format (``model.safetensors`` + ``config.json``),
    and copy the cluster manifest + .ctsr files into ``save_directory/expert_cache/``
    if ``expert_cache_dir`` is provided so the offloaded layout travels with
    the checkpoint."""
    os.makedirs(save_directory, exist_ok=True)

    # MiniMind/Chronos tie ``lm_head.weight`` to ``model.embed_tokens.weight``
    # but doesn't declare it via ``_tied_weights_keys``, which trips
    # safetensors' shared-tensor detector. Clone lm_head once before saving;
    # restore the tie after.
    lm_head = getattr(model, "lm_head", None)
    embed = getattr(getattr(model, "model", model), "embed_tokens", None)
    re_tie = (
        lm_head is not None and embed is not None
        and lm_head.weight.data_ptr() == embed.weight.data_ptr()
    )
    if re_tie:
        original = lm_head.weight
        lm_head.weight = torch.nn.Parameter(original.detach().clone())

    try:
        model.save_pretrained(save_directory, safe_serialization=safe_serialization)
    finally:
        if re_tie:
            lm_head.weight = embed.weight  # restore the tie

    if expert_cache_dir and os.path.isdir(expert_cache_dir):
        dst = os.path.join(save_directory, "expert_cache")
        os.makedirs(dst, exist_ok=True)
        for name in os.listdir(expert_cache_dir):
            if name.endswith((".ctsr", ".json")):
                shutil.copy2(
                    os.path.join(expert_cache_dir, name),
                    os.path.join(dst, name),
                )


def load_chronos_pretrained(
    save_directory: str,
    device: Optional[str] = None,
    **from_pretrained_kwargs,
):
    """Load a Chronos model saved via ``save_chronos_pretrained``.

    If ``save_directory/expert_cache/cluster_manifest.json`` is present,
    the path is returned alongside the model so callers can wire it into
    ``CacheManager(ssd_dir=...)`` or ``ExpertStore.attach_cluster_manifest(...)``.

    Returns ``(model, expert_cache_dir_or_None)``.
    """
    from chronos.model.model_chronos import ChronosForCausalLM
    model = ChronosForCausalLM.from_pretrained(save_directory, **from_pretrained_kwargs)

    # MiniMind/Chronos precompute RoPE frequencies as *persistent* buffers.
    # When the checkpoint is saved as fp16 and reloaded, these deterministic
    # buffers get truncated to zeros/NaNs. Recompute them from scratch —
    # they are a pure function of config.
    _rehydrate_rope_buffers(model)

    if device is not None:
        model = model.to(device)

    cache_dir = os.path.join(save_directory, "expert_cache")
    if os.path.exists(os.path.join(cache_dir, "cluster_manifest.json")):
        return model, cache_dir
    return model, None


def _rehydrate_rope_buffers(model) -> None:
    """Recompute ``freqs_cos`` and ``freqs_sin`` on the inner ChronosModel.
    These are deterministic from config; the precompute logic lives inside
    ``ChronosModel.__init__``. We invoke it via a fresh ChronosModel and
    copy the buffers across. Cheap — only runs once per load."""
    try:
        inner = getattr(model, "model", None)
        if inner is None:
            return
        cfg = model.config
        from chronos.model.model_chronos import ChronosModel
        fresh = ChronosModel(cfg)
        if hasattr(inner, "freqs_cos") and hasattr(fresh, "freqs_cos"):
            inner.freqs_cos = fresh.freqs_cos
        if hasattr(inner, "freqs_sin") and hasattr(fresh, "freqs_sin"):
            inner.freqs_sin = fresh.freqs_sin
    except Exception:
        # Rehydration is best-effort; fall through silently so an unexpected
        # config shape doesn't block the load.
        pass
