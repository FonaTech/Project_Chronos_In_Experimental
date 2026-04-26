"""Chronos full-checkpoint export utilities.

This module exports trained Chronos ``.pth`` checkpoints into deployment
formats while preserving the Chronos-specific topology that makes offloaded MoE
inference possible: top-k routing, shared fallback experts, lookahead router
settings, hybrid attention, and optional expert-cache cluster metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch

from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    config_to_dict,
    load_checkpoint_state_dict,
    read_checkpoint_config,
)
from chronos.model.config import ChronosConfig


EXPORT_FORMATS = (
    "fp16-safetensors",
    "q8_0-safetensors",
    "fp16-gguf",
    "q8_0-gguf",
)


GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9


@dataclass
class ExportResult:
    format: str
    path: str
    bytes: int
    tensors: int
    metadata: dict = field(default_factory=dict)


def _read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def _read_string(f) -> str:
    n = _read_u64(f)
    return f.read(n).decode("utf-8")


def _dequantize_q8_0(blocks: torch.Tensor, scales: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    x = blocks.to(torch.float32) * scales.to(torch.float32).view(-1, 1)
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return x.reshape(-1)[:numel].reshape(tuple(int(d) for d in shape)).to(torch.float16)


def _metadata_to_config_dict(metadata: dict | None) -> dict:
    if not metadata:
        return {}
    raw = metadata.get("chronos_export") or metadata.get("chronos.export.metadata_json")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    cfg = payload.get("resolved_config") or payload.get("checkpoint_config") or {}
    if not isinstance(cfg, dict):
        return {}
    if "max_seq_len" in cfg and "max_position_embeddings" not in cfg:
        cfg["max_position_embeddings"] = cfg["max_seq_len"]
    return cfg


def resolve_export_artifact(path: str) -> str:
    """Resolve an export directory or file path to a concrete model artifact."""
    if os.path.isdir(path):
        for name in (
            "model.fp16.safetensors",
            "model.q8_0.safetensors",
            "model.fp16.gguf",
            "model.q8_0.gguf",
        ):
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
    return path


class SafetensorsStateReader:
    """Random-access reader for Chronos safetensors exports.

    FP16 exports expose the original state-dict keys. Q8_0 exports expose the
    same logical keys by dequantizing one tensor at a time, which is what the
    lazy expert loader needs.
    """

    def __init__(self, path: str):
        from safetensors import safe_open

        self.path = path
        with safe_open(path, framework="pt", device="cpu") as f:
            self.metadata = dict(f.metadata() or {})
            physical_keys = list(f.keys())
        payload = {}
        raw = self.metadata.get("chronos_export")
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
        self.quantized = (payload.get("q8_0") or {}).get("tensors", {})
        q_sidecars = set()
        for info in self.quantized.values():
            q_sidecars.add(info.get("blocks"))
            q_sidecars.add(info.get("scales"))
        self._physical_keys = physical_keys
        self._logical_keys = [
            k for k in physical_keys
            if k not in q_sidecars and ".q8_0." not in k
        ]
        for key in self.quantized:
            if key not in self._logical_keys:
                self._logical_keys.append(key)

    def keys(self) -> list[str]:
        return list(self._logical_keys)

    def get_tensor(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(self.path, framework="pt", device="cpu") as f:
            if name in self._physical_keys:
                return f.get_tensor(name).clone()
            info = self.quantized.get(name)
            if not info:
                raise KeyError(name)
            blocks = f.get_tensor(info["blocks"])
            scales = f.get_tensor(info["scales"])
            return _dequantize_q8_0(blocks, scales, info["shape"])

    def load_state_dict(self, include_experts: bool = True) -> dict[str, torch.Tensor]:
        return {
            key: self.get_tensor(key)
            for key in self.keys()
            if include_experts or ".mlp.experts." not in key
        }

    def config_dict(self) -> dict:
        return _metadata_to_config_dict(self.metadata)


class GGUFStateReader:
    """Minimal GGUF reader for GGUF files produced by this module."""

    def __init__(self, path: str):
        self.path = path
        self.kv: dict[str, object] = {}
        self.tensor_infos: dict[str, dict] = {}
        self.data_start = 0
        self._parse_header()

    def _parse_header(self) -> None:
        with open(self.path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"not a GGUF file: {self.path}")
            self.version = _read_u32(f)
            tensor_count = _read_u64(f)
            kv_count = _read_u64(f)

            for _ in range(kv_count):
                key = _read_string(f)
                value_type = _read_u32(f)
                if value_type == GGUF_VALUE_STRING:
                    self.kv[key] = _read_string(f)
                elif value_type == GGUF_VALUE_UINT32:
                    self.kv[key] = _read_u32(f)
                elif value_type == GGUF_VALUE_ARRAY:
                    elem_type = _read_u32(f)
                    count = _read_u64(f)
                    if elem_type != GGUF_VALUE_UINT32:
                        raise ValueError(f"unsupported GGUF array type {elem_type} for {key}")
                    self.kv[key] = [_read_u32(f) for _ in range(count)]
                else:
                    raise ValueError(f"unsupported GGUF metadata value type {value_type} for {key}")

            for _ in range(tensor_count):
                name = _read_string(f)
                n_dims = _read_u32(f)
                dims = [_read_u64(f) for _ in range(n_dims)]
                ggml_type = _read_u32(f)
                offset = _read_u64(f)
                shape = list(reversed(dims))
                self.tensor_infos[name] = {
                    "shape": shape,
                    "type": ggml_type,
                    "offset": offset,
                }
            self.data_start = _align_offset(f.tell(), 32)

    def keys(self) -> list[str]:
        return list(self.tensor_infos.keys())

    def _tensor_nbytes(self, info: dict) -> int:
        numel = 1
        for dim in info["shape"]:
            numel *= int(dim)
        if info["type"] == GGML_TYPE_F16:
            return numel * 2
        if info["type"] == GGML_TYPE_F32:
            return numel * 4
        if info["type"] == GGML_TYPE_Q8_0:
            return ((numel + 31) // 32) * 34
        raise ValueError(f"unsupported GGUF tensor type {info['type']}")

    def get_tensor(self, name: str) -> torch.Tensor:
        info = self.tensor_infos[name]
        nbytes = self._tensor_nbytes(info)
        with open(self.path, "rb") as f:
            f.seek(self.data_start + int(info["offset"]))
            data = f.read(nbytes)
        shape = tuple(int(d) for d in info["shape"])
        if info["type"] == GGML_TYPE_F16:
            return torch.frombuffer(bytearray(data), dtype=torch.float16).clone().reshape(shape)
        if info["type"] == GGML_TYPE_F32:
            return torch.frombuffer(bytearray(data), dtype=torch.float32).clone().reshape(shape)
        if info["type"] == GGML_TYPE_Q8_0:
            numel = 1
            for dim in shape:
                numel *= int(dim)
            blocks = (numel + 31) // 32
            scale_bits = torch.empty(blocks, dtype=torch.int16)
            q = torch.empty((blocks, 32), dtype=torch.int8)
            pos = 0
            for i in range(blocks):
                scale_bits[i] = struct.unpack("<h", data[pos:pos + 2])[0]
                pos += 2
                q[i] = torch.frombuffer(bytearray(data[pos:pos + 32]), dtype=torch.int8).clone()
                pos += 32
            scales = scale_bits.view(torch.float16)
            return _dequantize_q8_0(q, scales, shape)
        raise ValueError(f"unsupported GGUF tensor type {info['type']}")

    def load_state_dict(self, include_experts: bool = True) -> dict[str, torch.Tensor]:
        return {
            key: self.get_tensor(key)
            for key in self.keys()
            if include_experts or ".mlp.experts." not in key
        }

    def config_dict(self) -> dict:
        metadata = {"chronos.export.metadata_json": self.kv.get("chronos.export.metadata_json")}
        cfg = _metadata_to_config_dict(metadata)
        for key, value in self.kv.items():
            if key.startswith("chronos.") and isinstance(value, int):
                cfg.setdefault(key[len("chronos."):], value)
        return cfg


def open_export_reader(path: str):
    artifact = resolve_export_artifact(path)
    if artifact.endswith(".safetensors"):
        return SafetensorsStateReader(artifact)
    if artifact.endswith(".gguf"):
        return GGUFStateReader(artifact)
    raise ValueError(f"unsupported export artifact: {path}")


def is_export_artifact(path: str) -> bool:
    artifact = resolve_export_artifact(path or "")
    return artifact.endswith((".safetensors", ".gguf")) and os.path.exists(artifact)


def config_dict_from_export(path: str) -> dict:
    reader = open_export_reader(path)
    return reader.config_dict()


def chronos_config_from_export(path: str) -> ChronosConfig:
    cfg = config_dict_from_export(path)
    if not cfg:
        raise ValueError(f"no Chronos export metadata found in {path}")
    for auto_key in ("intermediate_size", "moe_intermediate_size"):
        if cfg.get(auto_key) == 0:
            cfg.pop(auto_key, None)
    cfg.pop("max_seq_len", None)
    return ChronosConfig(**cfg)


def _normalize_formats(formats: Iterable[str] | str | None) -> list[str]:
    if formats is None:
        return ["fp16-safetensors"]
    if isinstance(formats, str):
        raw = [p.strip() for p in formats.replace(",", " ").split()]
    else:
        raw = [str(p).strip() for p in formats]
    out = []
    for fmt in raw:
        if not fmt:
            continue
        fmt = fmt.lower().replace("fp16_safetensors", "fp16-safetensors")
        fmt = fmt.replace("q8_0_safetensors", "q8_0-safetensors")
        fmt = fmt.replace("fp16_gguf", "fp16-gguf").replace("q8_0_gguf", "q8_0-gguf")
        if fmt == "all":
            for known in EXPORT_FORMATS:
                if known not in out:
                    out.append(known)
            continue
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"unsupported export format {fmt!r}; expected one of {EXPORT_FORMATS}")
        if fmt not in out:
            out.append(fmt)
    return out or ["fp16-safetensors"]


def _tensor_state_dict(state_dict: dict) -> dict[str, torch.Tensor]:
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    return {k: v for k, v in state_dict.items() if torch.is_tensor(v)}


def _load_state_and_config(
    checkpoint_path: str,
    config_path: str | None = None,
) -> tuple[dict[str, torch.Tensor], object, list[str]]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    state = _tensor_state_dict(load_checkpoint_state_dict(checkpoint_path, map_location="cpu"))
    config, sources = chronos_config_from_checkpoint(
        checkpoint_path,
        project_config_path=config_path,
        require_unsniffable=True,
    )
    return state, config, sources


def _clone_state_for_safetensors(state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    seen_ptrs: dict[tuple[int, int, int], str] = {}
    for name, tensor in state.items():
        t = tensor.detach().cpu()
        if t.is_floating_point():
            t = t.to(dtype=dtype)
        else:
            t = t.contiguous()
        key = (int(t.untyped_storage().data_ptr()), int(t.storage_offset()), int(t.numel()))
        if key in seen_ptrs:
            t = t.clone()
        else:
            t = t.clone() if not t.is_contiguous() else t.contiguous()
            seen_ptrs[key] = name
        tensors[name] = t
    return tensors


def _q8_0_quantize_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = tensor.detach().cpu().to(torch.float32).contiguous().view(-1)
    original_shape = torch.tensor(list(tensor.shape), dtype=torch.int32)
    pad = (-x.numel()) % 32
    if pad:
        x = torch.cat([x, torch.zeros(pad, dtype=x.dtype)], dim=0)
    blocks = x.view(-1, 32)
    scales = blocks.abs().amax(dim=1) / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    qs = torch.round(blocks / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return qs.contiguous(), scales.to(torch.float16).contiguous(), original_shape


def _q8_0_safetensors_state(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict]:
    tensors: dict[str, torch.Tensor] = {}
    quantized: dict[str, dict] = {}
    for name, tensor in state.items():
        if tensor.is_floating_point() and tensor.ndim >= 2:
            q, scales, shape = _q8_0_quantize_tensor(tensor)
            tensors[f"{name}.q8_0.blocks"] = q
            tensors[f"{name}.q8_0.scales"] = scales
            tensors[f"{name}.q8_0.shape"] = shape
            quantized[name] = {
                "blocks": f"{name}.q8_0.blocks",
                "scales": f"{name}.q8_0.scales",
                "shape": list(tensor.shape),
                "block_size": 32,
            }
        else:
            tensors[name] = tensor.detach().cpu().contiguous()
    return tensors, quantized


def _sidecar_config_for_export(checkpoint_path: str, config) -> dict:
    sidecar, sidecar_path = read_checkpoint_config(checkpoint_path)
    return {
        "checkpoint_config_path": sidecar_path,
        "checkpoint_config": sidecar,
        "resolved_config": config_to_dict(config),
    }


def _cluster_manifest_summary(expert_cache_dir: str | None) -> dict | None:
    if not expert_cache_dir:
        return None
    manifest_path = os.path.join(expert_cache_dir, "cluster_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "path": manifest_path,
        "num_clusters": len(data.get("clusters", [])),
        "storage_format": data.get("storage_format", "safetensors"),
        "num_experts": data.get("num_experts"),
        "num_layers": data.get("num_layers"),
    }


def _build_auto_cluster_cache(
    checkpoint_path: str,
    output_dir: str,
    *,
    config_path: str | None = None,
    calibration_data_path: str | None = None,
    cluster_output_dir: str | None = None,
    cluster_max_batches: int = 50,
    cluster_batch_size: int = 4,
    cluster_max_seq_len: int = 256,
    cluster_device: str = "cpu",
    strict: bool = False,
) -> tuple[str | None, dict | None]:
    if not calibration_data_path:
        note = {
            "enabled": True,
            "status": "skipped",
            "reason": "calibration_data_path not provided",
        }
        if strict:
            raise ValueError("auto_cluster requires calibration_data_path")
        return None, note
    if not os.path.exists(calibration_data_path):
        note = {
            "enabled": True,
            "status": "skipped",
            "reason": f"calibration data not found: {calibration_data_path}",
        }
        if strict:
            raise FileNotFoundError(note["reason"])
        return None, note

    from chronos.io.cluster_layout import build_clustered_expert_cache_from_checkpoint

    cache_dir = cluster_output_dir or os.path.join(output_dir, "_auto_cluster_expert_cache")
    try:
        summary = build_clustered_expert_cache_from_checkpoint(
            checkpoint_path,
            calibration_data_path,
            cache_dir,
            config_path=config_path,
            device=cluster_device,
            max_batches=cluster_max_batches,
            batch_size=cluster_batch_size,
            max_seq_len=cluster_max_seq_len,
            dtype=torch.float16,
        )
        summary["enabled"] = True
        summary["status"] = "built"
        return cache_dir, summary
    except Exception as exc:
        if strict:
            raise
        return None, {
            "enabled": True,
            "status": "failed",
            "reason": str(exc),
        }


def _export_metadata(
    checkpoint_path: str,
    config,
    *,
    fmt: str,
    sources: Sequence[str],
    quantized_tensors: dict | None = None,
    expert_cache_dir: str | None = None,
    cluster_build: dict | None = None,
) -> dict[str, str]:
    cfg = config_to_dict(config)
    payload = {
        "format": fmt,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "config_sources": list(sources),
        "architecture": "chronos",
        "chronos": {
            "hidden_size": cfg.get("hidden_size"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "num_experts": cfg.get("num_experts"),
            "num_experts_per_tok": cfg.get("num_experts_per_tok"),
            "num_shared_experts": cfg.get("num_shared_experts"),
            "moe_intermediate_size": cfg.get("moe_intermediate_size"),
            "lookahead_steps": cfg.get("lookahead_steps"),
            "use_hybrid_attention": cfg.get("use_hybrid_attention"),
            "kv_latent_dim": cfg.get("kv_latent_dim"),
            "rope_dim": cfg.get("rope_dim"),
            "sliding_window_size": cfg.get("sliding_window_size"),
            "storage_format": cfg.get("storage_format"),
            "cluster_manifest_path": cfg.get("cluster_manifest_path"),
            "fallback_mask_prob": cfg.get("fallback_mask_prob"),
            "lambda_lookahead_topk": cfg.get("lambda_lookahead_topk"),
            "offload_miss_policy_default": cfg.get("offload_miss_policy_default", "on_demand"),
            "recommended_resident_experts": cfg.get("recommended_resident_experts"),
        },
        "export_contract": "predictive_lazy_routed_experts_or_shared_fallback + shared_residual",
        "offload_features": {
            "lazy_expert_loading": True,
            "lookahead_router": True,
            "shared_fallback_experts": True,
            "predictive_on_demand": True,
            "sync_on_demand_diagnostic": True,
            "miss_policy_default": cfg.get("offload_miss_policy_default", "on_demand"),
            "cluster_aware_safetensors": bool(_cluster_manifest_summary(expert_cache_dir)),
        },
    }
    if cluster_build:
        payload["cluster_build"] = cluster_build
    payload.update(_sidecar_config_for_export(checkpoint_path, config))
    cluster = _cluster_manifest_summary(expert_cache_dir)
    if cluster:
        payload["expert_cache"] = cluster
    if quantized_tensors is not None:
        payload["q8_0"] = {
            "scheme": "symmetric block quantization",
            "block_size": 32,
            "quantized_tensor_count": len(quantized_tensors),
            "tensors": quantized_tensors,
        }
    return {
        "format": fmt,
        "architecture": "chronos",
        "chronos_export": json.dumps(payload, sort_keys=True),
    }


def export_safetensors(
    state: dict[str, torch.Tensor],
    checkpoint_path: str,
    config,
    output_dir: str,
    *,
    fmt: str,
    sources: Sequence[str],
    expert_cache_dir: str | None = None,
    cluster_build: dict | None = None,
) -> ExportResult:
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    if fmt == "fp16-safetensors":
        tensors = _clone_state_for_safetensors(state, torch.float16)
        metadata = _export_metadata(
            checkpoint_path,
            config,
            fmt=fmt,
            sources=sources,
            expert_cache_dir=expert_cache_dir,
            cluster_build=cluster_build,
        )
        filename = "model.fp16.safetensors"
    elif fmt == "q8_0-safetensors":
        tensors, qmeta = _q8_0_safetensors_state(state)
        metadata = _export_metadata(
            checkpoint_path,
            config,
            fmt=fmt,
            sources=sources,
            quantized_tensors=qmeta,
            expert_cache_dir=expert_cache_dir,
            cluster_build=cluster_build,
        )
        filename = "model.q8_0.safetensors"
    else:
        raise ValueError(f"not a safetensors export format: {fmt}")

    path = os.path.join(output_dir, filename)
    save_file(tensors, path, metadata=metadata)
    return ExportResult(fmt, path, os.path.getsize(path), len(tensors), metadata)


def _pack_u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def _pack_u64(value: int) -> bytes:
    return struct.pack("<Q", int(value))


def _pack_string(value: str) -> bytes:
    data = str(value).encode("utf-8")
    return _pack_u64(len(data)) + data


def _gguf_kv_string(key: str, value: str) -> bytes:
    return _pack_string(key) + _pack_u32(GGUF_VALUE_STRING) + _pack_string(value)


def _gguf_kv_u32(key: str, value: int) -> bytes:
    return _pack_string(key) + _pack_u32(GGUF_VALUE_UINT32) + _pack_u32(int(value))


def _gguf_kv_u32_array(key: str, values: Sequence[int]) -> bytes:
    out = _pack_string(key)
    out += _pack_u32(GGUF_VALUE_ARRAY)
    out += _pack_u32(GGUF_VALUE_UINT32)
    out += _pack_u64(len(values))
    for value in values:
        out += _pack_u32(int(value))
    return out


def _align_offset(offset: int, alignment: int = 32) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _gguf_fp16_payload(tensor: torch.Tensor) -> tuple[bytes, int, list[int]]:
    t = tensor.detach().cpu()
    if t.is_floating_point():
        t = t.to(torch.float16)
        ggml_type = GGML_TYPE_F16
    else:
        t = t.to(torch.float32)
        ggml_type = GGML_TYPE_F32
    t = t.contiguous()
    return t.numpy().tobytes(order="C"), ggml_type, list(t.shape)


def _gguf_q8_0_payload(tensor: torch.Tensor) -> tuple[bytes, int, list[int]]:
    if not tensor.is_floating_point() or tensor.ndim < 2:
        return _gguf_fp16_payload(tensor)
    q, scales, _shape = _q8_0_quantize_tensor(tensor)
    blocks = q.shape[0]
    buf = bytearray()
    scales_i16 = scales.view(torch.int16).contiguous()
    q_cpu = q.contiguous()
    for i in range(blocks):
        buf += struct.pack("<h", int(scales_i16[i].item()))
        buf += q_cpu[i].numpy().tobytes(order="C")
    return bytes(buf), GGML_TYPE_Q8_0, list(tensor.shape)


def export_gguf(
    state: dict[str, torch.Tensor],
    checkpoint_path: str,
    config,
    output_dir: str,
    *,
    fmt: str,
    sources: Sequence[str],
    expert_cache_dir: str | None = None,
    cluster_build: dict | None = None,
) -> ExportResult:
    os.makedirs(output_dir, exist_ok=True)
    quantized = fmt == "q8_0-gguf"
    filename = "model.q8_0.gguf" if quantized else "model.fp16.gguf"
    path = os.path.join(output_dir, filename)

    cfg = config_to_dict(config)
    metadata_payload = json.loads(
        _export_metadata(
            checkpoint_path,
            config,
            fmt=fmt,
            sources=sources,
            expert_cache_dir=expert_cache_dir,
            cluster_build=cluster_build,
        )["chronos_export"]
    )
    kvs = [
        _gguf_kv_string("general.architecture", "chronos"),
        _gguf_kv_string("general.name", "Project Chronos"),
        _gguf_kv_string("chronos.export.format", fmt),
        _gguf_kv_string("chronos.export.metadata_json", json.dumps(metadata_payload, sort_keys=True)),
    ]
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "num_shared_experts",
        "moe_intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "lookahead_steps",
        "lambda_lookahead_topk",
        "kv_latent_dim",
        "rope_dim",
        "sliding_window_size",
        "vocab_size",
        "recommended_resident_experts",
    ):
        value = cfg.get(key)
        if isinstance(value, int):
            kvs.append(_gguf_kv_u32(f"chronos.{key}", value))

    tensor_infos = []
    payloads: list[bytes] = []
    running_offset = 0
    for name, tensor in state.items():
        payload, ggml_type, shape = (
            _gguf_q8_0_payload(tensor) if quantized else _gguf_fp16_payload(tensor)
        )
        running_offset = _align_offset(running_offset, 32)
        tensor_infos.append((name, shape, ggml_type, running_offset))
        payloads.append(payload)
        running_offset += len(payload)

    header = bytearray()
    header += b"GGUF"
    header += _pack_u32(3)
    header += _pack_u64(len(tensor_infos))
    header += _pack_u64(len(kvs))
    for kv in kvs:
        header += kv
    for name, shape, ggml_type, offset in tensor_infos:
        header += _pack_string(name)
        header += _pack_u32(len(shape))
        for dim in reversed(shape):
            header += _pack_u64(int(dim))
        header += _pack_u32(ggml_type)
        header += _pack_u64(offset)

    with open(path, "wb") as f:
        f.write(header)
        pos = f.tell()
        data_start = _align_offset(pos, 32)
        f.write(b"\x00" * (data_start - pos))
        written = 0
        for payload in payloads:
            aligned = _align_offset(written, 32)
            f.write(b"\x00" * (aligned - written))
            written = aligned
            f.write(payload)
            written += len(payload)

    return ExportResult(fmt, path, os.path.getsize(path), len(tensor_infos), metadata_payload)


def _copy_expert_cache(expert_cache_dir: str | None, output_dir: str) -> str | None:
    if not expert_cache_dir or not os.path.isdir(expert_cache_dir):
        return None
    manifest = os.path.join(expert_cache_dir, "cluster_manifest.json")
    if not os.path.exists(manifest):
        return None
    dst = os.path.join(output_dir, "expert_cache")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(expert_cache_dir):
        if name.endswith((".ctsr", ".json")):
            shutil.copy2(os.path.join(expert_cache_dir, name), os.path.join(dst, name))
    return dst


def _write_config_json(output_dir: str, config) -> str:
    cfg = dict(config.to_dict() if hasattr(config, "to_dict") else config_to_dict(config))
    cfg["model_type"] = "chronos"
    cfg.setdefault("architectures", ["ChronosForCausalLM"])
    path = os.path.join(output_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return path


def _write_deployment_notes(output_dir: str, results: Sequence[ExportResult]) -> None:
    gguf_files = [os.path.basename(r.path) for r in results if r.path.endswith(".gguf")]
    if not gguf_files:
        return
    notes = [
        "# Project Chronos GGUF deployment",
        "",
        "These files are standard GGUF containers with `general.architecture=chronos`.",
        "They are intended for Chronos-aware runtimes. Stock Ollama/llama.cpp builds",
        "that do not implement the Chronos architecture cannot execute the model",
        "correctly because Chronos uses hybrid MLA/sliding-window attention,",
        "lookahead expert prediction, and lazy MoE expert loading.",
        "",
        "For an Ollama build with Chronos architecture support, use a Modelfile like:",
        "",
        "```",
        f"FROM ./{gguf_files[0]}",
        "PARAMETER temperature 0.7",
        "```",
        "",
    ]
    with open(os.path.join(output_dir, "OLLAMA_CHRONOS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(notes))
    for name in gguf_files:
        stem = name.replace("model.", "").replace(".gguf", "")
        with open(os.path.join(output_dir, f"Modelfile.{stem}"), "w", encoding="utf-8") as f:
            f.write(f"FROM ./{name}\nPARAMETER temperature 0.7\n")


def export_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    *,
    formats: Iterable[str] | str | None = None,
    config_path: str | None = None,
    expert_cache_dir: str | None = None,
    copy_expert_cache: bool = True,
    auto_cluster: bool = False,
    calibration_data_path: str | None = None,
    cluster_output_dir: str | None = None,
    cluster_max_batches: int = 50,
    cluster_batch_size: int = 4,
    cluster_max_seq_len: int = 256,
    cluster_device: str = "cpu",
    strict_auto_cluster: bool = False,
) -> list[ExportResult]:
    fmt_list = _normalize_formats(formats)
    state, config, sources = _load_state_and_config(checkpoint_path, config_path)
    os.makedirs(output_dir, exist_ok=True)
    _write_config_json(output_dir, config)

    cluster_build = None
    if auto_cluster and not expert_cache_dir:
        expert_cache_dir, cluster_build = _build_auto_cluster_cache(
            checkpoint_path,
            output_dir,
            config_path=config_path,
            calibration_data_path=calibration_data_path,
            cluster_output_dir=cluster_output_dir,
            cluster_max_batches=cluster_max_batches,
            cluster_batch_size=cluster_batch_size,
            cluster_max_seq_len=cluster_max_seq_len,
            cluster_device=cluster_device,
            strict=strict_auto_cluster,
        )
    elif auto_cluster:
        cluster_build = {
            "enabled": True,
            "status": "skipped",
            "reason": "expert_cache_dir already provided",
            "expert_cache_dir": expert_cache_dir,
        }

    copied_cache_dir = _copy_expert_cache(expert_cache_dir, output_dir) if copy_expert_cache else None
    cache_for_meta = copied_cache_dir or expert_cache_dir
    if cluster_build and cache_for_meta and _cluster_manifest_summary(cache_for_meta):
        cluster_build = dict(cluster_build)
        cluster_build["exported_expert_cache_dir"] = cache_for_meta
    results = []
    for fmt in fmt_list:
        if fmt.endswith("safetensors"):
            result = export_safetensors(
                state,
                checkpoint_path,
                config,
                output_dir,
                fmt=fmt,
                sources=sources,
                expert_cache_dir=cache_for_meta,
                cluster_build=cluster_build,
            )
        else:
            result = export_gguf(
                state,
                checkpoint_path,
                config,
                output_dir,
                fmt=fmt,
                sources=sources,
                expert_cache_dir=cache_for_meta,
                cluster_build=cluster_build,
            )
        results.append(result)

    manifest = {
        "source_checkpoint": os.path.abspath(checkpoint_path),
        "formats": [r.__dict__ for r in results],
        "config_sources": list(sources),
        "expert_cache_dir": cache_for_meta,
        "cluster_build": cluster_build,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(output_dir, "chronos_export_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    _write_deployment_notes(output_dir, results)
    return results


def format_export_report(results: Sequence[ExportResult]) -> str:
    if not results:
        return "No export produced."
    lines = ["Export complete:"]
    for result in results:
        size_mb = result.bytes / (1024 * 1024)
        lines.append(f"- {result.format}: {result.path} ({size_mb:.2f} MB, {result.tensors} tensors)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a Chronos checkpoint to safetensors/GGUF")
    parser.add_argument("--model_path", "--checkpoint_path", dest="model_path", required=True)
    parser.add_argument("--output_dir", default="./exports/chronos")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["fp16-safetensors"],
        help="Any of: fp16-safetensors q8_0-safetensors fp16-gguf q8_0-gguf all",
    )
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--expert_cache_dir", default=None)
    parser.add_argument("--no_copy_expert_cache", action="store_true")
    parser.add_argument("--auto_cluster", action="store_true")
    parser.add_argument("--calibration_data_path", default=None)
    parser.add_argument("--cluster_output_dir", default=None)
    parser.add_argument("--cluster_max_batches", type=int, default=50)
    parser.add_argument("--cluster_batch_size", type=int, default=4)
    parser.add_argument("--cluster_max_seq_len", type=int, default=256)
    parser.add_argument("--cluster_device", default="cpu")
    parser.add_argument("--strict_auto_cluster", action="store_true")
    args = parser.parse_args(argv)

    results = export_checkpoint(
        args.model_path,
        args.output_dir,
        formats=args.formats,
        config_path=args.config_path,
        expert_cache_dir=args.expert_cache_dir,
        copy_expert_cache=not args.no_copy_expert_cache,
        auto_cluster=args.auto_cluster,
        calibration_data_path=args.calibration_data_path,
        cluster_output_dir=args.cluster_output_dir,
        cluster_max_batches=args.cluster_max_batches,
        cluster_batch_size=args.cluster_batch_size,
        cluster_max_seq_len=args.cluster_max_seq_len,
        cluster_device=args.cluster_device,
        strict_auto_cluster=args.strict_auto_cluster,
    )
    print(format_export_report(results))


if __name__ == "__main__":
    main()
