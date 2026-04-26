"""Checkpoint metadata and controlled loading helpers for Chronos.

Chronos stores raw PyTorch ``.pth`` state dicts for compatibility, but a raw
state dict cannot encode all topology choices. In particular
``num_experts_per_tok`` is a runtime/router choice and cannot be recovered
from tensor shapes. New checkpoints therefore get a sibling
``*.config.json`` sidecar, and old checkpoints can be loaded only after the
missing unsniffable fields are supplied by a project config or CLI overrides.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Any, Iterable

import torch

from .config import ChronosConfig


CHECKPOINT_CONFIG_VERSION = 1

CHECKPOINT_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_experts",
    "num_experts_per_tok",
    "num_shared_experts",
    "intermediate_size",
    "moe_intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "rope_dim",
    "kv_latent_dim",
    "lookahead_steps",
    "vocab_size",
    "max_position_embeddings",
    "sliding_window_size",
    "use_hybrid_attention",
    "use_moe",
    "tie_word_embeddings",
    "norm_topk_prob",
    "router_aux_loss_coef",
    "lambda_balance",
    "lambda_temporal",
    "lambda_lookahead",
    "lambda_lookahead_topk",
    "lambda_router_anchor",
    "fallback_mask_prob",
    "offload_miss_policy_default",
    "recommended_resident_experts",
    "vram_budget_gb",
    "pinned_memory_max_fraction",
    "storage_format",
    "cluster_manifest_path",
    "dropout",
    "rms_norm_eps",
    "rope_theta",
    "flash_attn",
    "bos_token_id",
    "eos_token_id",
)
VALID_CHRONOS_CONFIG_KEYS = set(CHECKPOINT_CONFIG_FIELDS) | {"max_seq_len"}

SNIFFABLE_TOPOLOGY_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_experts",
    "moe_intermediate_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "rope_dim",
    "kv_latent_dim",
    "lookahead_steps",
    "vocab_size",
)

UNSNIFFABLE_REQUIRED_FIELDS = (
    "num_experts_per_tok",
    "num_shared_experts",
)


def checkpoint_config_path(checkpoint_path: str) -> str:
    if checkpoint_path.endswith(".pth"):
        return checkpoint_path[:-4] + ".config.json"
    return checkpoint_path + ".config.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _sha256_file(path: str) -> str | None:
    if not path or not os.path.exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_metadata(tokenizer_path: str | None = None) -> dict:
    if not tokenizer_path:
        try:
            import chronos.deps

            tokenizer_path = chronos.deps.get_tokenizer_path()
        except Exception:
            tokenizer_path = None
    meta = {"path": tokenizer_path}
    if tokenizer_path and os.path.isdir(tokenizer_path):
        for name in ("tokenizer.json", "tokenizer_config.json"):
            f = os.path.join(tokenizer_path, name)
            digest = _sha256_file(f)
            if digest:
                meta[f"{name}_sha256"] = digest
    elif tokenizer_path:
        digest = _sha256_file(tokenizer_path)
        if digest:
            meta["sha256"] = digest
    return meta


def config_to_dict(config: ChronosConfig | dict) -> dict:
    if isinstance(config, dict):
        src = dict(config)
        # UI config uses max_seq_len; model config uses max_position_embeddings.
        if "max_seq_len" in src and "max_position_embeddings" not in src:
            src["max_position_embeddings"] = src["max_seq_len"]
        return {k: _jsonable(v) for k, v in src.items() if v is not None}

    out = {}
    for key in CHECKPOINT_CONFIG_FIELDS:
        if hasattr(config, key):
            out[key] = _jsonable(getattr(config, key))
    return out


def save_checkpoint_config(
    checkpoint_path: str,
    config: ChronosConfig | dict,
    tokenizer_path: str | None = None,
    stage: str | None = None,
    extra: dict | None = None,
) -> str:
    cfg = config_to_dict(config)
    cfg.setdefault("use_moe", True)
    meta = {
        "checkpoint_format": "chronos_pth",
        "checkpoint_config_version": CHECKPOINT_CONFIG_VERSION,
        "stage": stage,
        "checkpoint": os.path.basename(checkpoint_path),
        "config": cfg,
        "tokenizer": tokenizer_metadata(tokenizer_path),
    }
    if extra:
        meta["extra"] = _jsonable(extra)
    path = checkpoint_config_path(checkpoint_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return path


def save_state_dict_with_config(
    model,
    checkpoint_path: str,
    config: ChronosConfig | dict,
    tokenizer_path: str | None = None,
    stage: str | None = None,
    half: bool = True,
    extra: dict | None = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)) or ".", exist_ok=True)
    state = model.state_dict()
    if half:
        state = {k: v.half().cpu() for k, v in state.items()}
    torch.save(state, checkpoint_path)
    save_checkpoint_config(checkpoint_path, config, tokenizer_path, stage=stage, extra=extra)
    return checkpoint_path


def read_checkpoint_config(checkpoint_path: str) -> tuple[dict, str | None]:
    path = checkpoint_config_path(checkpoint_path)
    if not os.path.exists(path):
        return {}, None
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta.get("config", meta)
    if not isinstance(cfg, dict):
        return {}, path
    return dict(cfg), path


def read_project_config(config_path: str | None = None) -> tuple[dict, str | None]:
    candidates = []
    if config_path:
        candidates.append(config_path)
    candidates.append("chronos_config.json")
    for path in candidates:
        if not path:
            continue
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            if "max_seq_len" in cfg and "max_position_embeddings" not in cfg:
                cfg["max_position_embeddings"] = cfg["max_seq_len"]
            return cfg, path
    return {}, None


def _state_dict_from_checkpoint(checkpoint_path: str, map_location="cpu") -> dict:
    sd = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if not isinstance(sd, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint_path}")
    return sd


def sniff_checkpoint_config(checkpoint_path: str, map_location="cpu") -> dict:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    sd = _state_dict_from_checkpoint(checkpoint_path, map_location=map_location)
    out: dict[str, Any] = {}

    embed = sd.get("model.embed_tokens.weight")
    if embed is not None:
        out["vocab_size"], out["hidden_size"] = int(embed.shape[0]), int(embed.shape[1])

    layer_idxs = set()
    for k in sd.keys():
        if k.startswith("model.layers."):
            try:
                layer_idxs.add(int(k.split(".")[2]))
            except (ValueError, IndexError):
                pass
    if layer_idxs:
        out["num_hidden_layers"] = max(layer_idxs) + 1

    expert_idxs = set()
    for k in sd.keys():
        if k.startswith("model.layers.0.mlp.experts."):
            try:
                expert_idxs.add(int(k.split(".")[5]))
            except (ValueError, IndexError):
                pass
    if expert_idxs:
        out["num_experts"] = max(expert_idxs) + 1

    shared_idxs = set()
    for k in sd.keys():
        if k.startswith("model.layers.0.mlp.shared_experts."):
            try:
                shared_idxs.add(int(k.split(".")[5]))
            except (ValueError, IndexError):
                pass
    if shared_idxs:
        out["num_shared_experts"] = max(shared_idxs) + 1

    gate = sd.get("model.layers.0.mlp.experts.0.gate_proj.weight")
    if gate is not None:
        out["moe_intermediate_size"] = int(gate.shape[0])
        out["intermediate_size"] = int(gate.shape[0])

    lookahead_proj = sd.get("model.lookahead_router.proj.2.weight")
    if lookahead_proj is not None and out.get("num_experts"):
        total = int(lookahead_proj.shape[0])
        n_exp = int(out["num_experts"])
        if n_exp > 0 and total % n_exp == 0:
            out["lookahead_steps"] = max(0, total // n_exp - 1)

    qnope = sd.get("model.layers.0.self_attn.q_nope_proj.weight")
    qrope = sd.get("model.layers.0.self_attn.q_rope_proj.weight")
    kvdown = sd.get("model.layers.0.self_attn.kv_down_proj.weight")
    vproj = sd.get("model.layers.0.self_attn.v_proj.weight")
    if qnope is not None and qrope is not None and out.get("hidden_size"):
        hidden = int(out["hidden_size"])
        candidates = [
            n for n in range(1, hidden + 1)
            if hidden % n == 0
            and int(qrope.shape[0]) % n == 0
            and int(qnope.shape[0]) % n == 0
        ]
        if candidates:
            preferred = 8 if 8 in candidates else max(candidates)
            head_dim = hidden // preferred
            rope_dim = int(qrope.shape[0]) // preferred
            nope_dim = int(qnope.shape[0]) // preferred
            if rope_dim + nope_dim == head_dim:
                out["num_attention_heads"] = preferred
                out["rope_dim"] = rope_dim
                out["head_dim"] = head_dim
    if kvdown is not None:
        out["kv_latent_dim"] = int(kvdown.shape[0])
    if vproj is not None and out.get("hidden_size"):
        n_heads = max(int(out.get("num_attention_heads", 8)), 1)
        head_dim = int(out.get("head_dim", int(out["hidden_size"]) // n_heads))
        if head_dim > 0:
            out["num_key_value_heads"] = max(1, int(vproj.shape[0]) // head_dim)

    sw_q = sd.get("model.layers.1.self_attn.q_proj.weight")
    sw_k = sd.get("model.layers.1.self_attn.k_proj.weight")
    if sw_q is not None and sw_k is not None and out.get("hidden_size"):
        hidden = int(out["hidden_size"])
        # Prefer existing head_dim when MLA sniffed it.
        head_dim = int(out.get("head_dim", hidden // max(int(out.get("num_attention_heads", 8)), 1)))
        if head_dim > 0:
            out["num_attention_heads"] = max(1, int(sw_q.shape[0]) // head_dim)
            out["num_key_value_heads"] = max(1, int(sw_k.shape[0]) // head_dim)

    return out


def resolve_checkpoint_path(
    save_dir: str,
    stem: str,
    hidden_size: int | None = None,
    explicit_path: str | None = None,
) -> str:
    if explicit_path:
        return explicit_path
    if hidden_size:
        exact = os.path.join(save_dir, f"{stem}_{hidden_size}_moe.pth")
        if os.path.exists(exact):
            return exact
    pattern = os.path.join(save_dir, f"{stem}_*_moe.pth")
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if candidates:
        return max(candidates, key=os.path.getmtime)
    if hidden_size:
        return os.path.join(save_dir, f"{stem}_{hidden_size}_moe.pth")
    return os.path.join(save_dir, f"{stem}_<H>_moe.pth")


def config_dict_for_checkpoint(
    checkpoint_path: str | None = None,
    project_config_path: str | None = None,
    overrides: dict | None = None,
    require_unsniffable: bool = True,
) -> tuple[dict, list[str]]:
    cfg: dict[str, Any] = {}
    sources: list[str] = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        sidecar_cfg, sidecar_path = read_checkpoint_config(checkpoint_path)
        if sidecar_cfg:
            cfg.update(sidecar_cfg)
            sources.append(sidecar_path or checkpoint_config_path(checkpoint_path))
        else:
            sniffed = sniff_checkpoint_config(checkpoint_path)
            if sniffed:
                cfg.update(sniffed)
                sources.append(f"{checkpoint_path} (tensor shapes)")

    project_cfg, project_path = read_project_config(project_config_path)
    if project_cfg:
        for key, value in project_cfg.items():
            if key not in VALID_CHRONOS_CONFIG_KEYS:
                continue
            if key not in cfg or cfg.get(key) in (None, "", 0):
                cfg[key] = value
        sources.append(project_path or project_config_path or "chronos_config.json")

    if overrides:
        for key, value in overrides.items():
            if value is None or value == "":
                continue
            if key not in VALID_CHRONOS_CONFIG_KEYS:
                continue
            if key == "max_seq_len":
                cfg["max_position_embeddings"] = value
            else:
                cfg[key] = value
        sources.append("explicit overrides")

    cfg.setdefault("use_moe", True)
    cfg.setdefault("use_hybrid_attention", True)

    if require_unsniffable:
        missing = [k for k in UNSNIFFABLE_REQUIRED_FIELDS if k not in cfg or cfg[k] in (None, "")]
        if missing:
            src = ", ".join(sources) if sources else "no config sources"
            raise ValueError(
                "Checkpoint is missing unsniffable topology fields "
                f"{missing}. Provide a .config.json sidecar, project config, or CLI overrides. "
                f"Sources used: {src}"
            )
    return cfg, sources


def chronos_config_from_checkpoint(
    checkpoint_path: str | None = None,
    project_config_path: str | None = None,
    overrides: dict | None = None,
    require_unsniffable: bool = True,
) -> tuple[ChronosConfig, list[str]]:
    cfg, sources = config_dict_for_checkpoint(
        checkpoint_path,
        project_config_path=project_config_path,
        overrides=overrides,
        require_unsniffable=require_unsniffable,
    )
    for auto_key in ("intermediate_size", "moe_intermediate_size"):
        if cfg.get(auto_key) == 0:
            cfg.pop(auto_key, None)
    cfg.pop("max_seq_len", None)
    return ChronosConfig(**cfg), sources


def load_checkpoint_state_dict(checkpoint_path: str, map_location="cpu") -> dict:
    return _state_dict_from_checkpoint(checkpoint_path, map_location=map_location)


def load_state_dict_controlled(
    model,
    state_dict: dict,
    *,
    allow_missing_prefixes: Iterable[str] = (),
    allow_missing_substrings: Iterable[str] = (),
    allow_unexpected_prefixes: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    result = model.load_state_dict(state_dict, strict=False)
    if result is None:
        missing, unexpected = [], []
    else:
        missing = getattr(result, "missing_keys", None)
        unexpected = getattr(result, "unexpected_keys", None)
        if missing is None or unexpected is None:
            missing, unexpected = result

    def allowed(key: str, prefixes: Iterable[str], substrings: Iterable[str] = ()) -> bool:
        return any(key.startswith(p) for p in prefixes) or any(s in key for s in substrings)

    bad_missing = [
        k for k in missing
        if not allowed(k, allow_missing_prefixes, allow_missing_substrings)
    ]
    bad_unexpected = [
        k for k in unexpected
        if not allowed(k, allow_unexpected_prefixes)
    ]
    if bad_missing or bad_unexpected:
        details = []
        if bad_missing:
            details.append("missing: " + ", ".join(bad_missing[:20]))
        if bad_unexpected:
            details.append("unexpected: " + ", ".join(bad_unexpected[:20]))
        raise RuntimeError("checkpoint load mismatch; " + " | ".join(details))
    return list(missing), list(unexpected)


def load_checkpoint_into_model(
    model,
    checkpoint_path: str,
    map_location="cpu",
    **kwargs,
) -> dict:
    state = load_checkpoint_state_dict(checkpoint_path, map_location=map_location)
    load_state_dict_controlled(model, state, **kwargs)
    return state
