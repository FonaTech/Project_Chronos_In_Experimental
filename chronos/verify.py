"""Lightweight post-training verification for Chronos checkpoints."""
from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn.functional as F

import chronos.deps  # noqa: F401
from chronos.backend import available as available_backends
from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
from chronos.model.model_chronos import ChronosForCausalLM


def _make_probe_ids(vocab_size: int, seq_len: int = 8) -> torch.Tensor:
    ids = torch.arange(1, seq_len + 1, dtype=torch.long).reshape(1, seq_len)
    return ids.remainder(max(2, int(vocab_size) - 1)).clamp_min(1)


def _torch_prefill(config, checkpoint_path: str, ids: torch.Tensor, device: str = "cpu") -> dict[str, Any]:
    state = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    model = ChronosForCausalLM(config)
    load_state_dict_controlled(model, state)
    model = model.to(device).eval()
    with torch.no_grad():
        out, _lookahead = model(ids.to(device), use_cache=False)
    logits = out.logits[:, -1, :].detach().float().cpu()
    return {
        "logits": logits,
        "argmax": int(logits.argmax(dim=-1).item()),
        "finite": bool(torch.isfinite(logits).all().item()),
    }


def _masked_drift(config, checkpoint_path: str, ids: torch.Tensor) -> dict[str, Any]:
    state = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    model = ChronosForCausalLM(config)
    load_state_dict_controlled(model, state)
    model.eval()
    masks = [
        torch.ones(int(config.num_experts), dtype=torch.bool)
        for _ in range(int(config.num_hidden_layers))
    ]
    with torch.no_grad():
        no_mask, _ = model(ids, use_cache=False)
        all_avail, _ = model(ids, use_cache=False, available_expert_masks=masks)
    a = no_mask.logits[:, -1, :].float()
    b = all_avail.logits[:, -1, :].float()
    return {
        "cosine": float(F.cosine_similarity(a, b, dim=-1).mean().item()),
        "argmax_match": bool(a.argmax(dim=-1).eq(b.argmax(dim=-1)).all().item()),
        "max_abs": float((a - b).abs().max().item()),
    }


def _mlx_prefill(config, checkpoint_path: str, ids: torch.Tensor) -> dict[str, Any]:
    import mlx.core as mx
    from chronos.mlx.model import ChronosMLXModel

    state = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    pt = ChronosForCausalLM(config)
    load_state_dict_controlled(pt, state)
    model = ChronosMLXModel.from_chronos_pytorch(pt, config)
    logits, _lookahead, _cache = model(mx.array(ids.numpy()).astype(mx.int32))
    mx.eval(logits)
    last = torch.from_numpy(__import__("numpy").array(logits[:, -1, :], dtype="float32"))
    return {
        "logits": last,
        "argmax": int(last.argmax(dim=-1).item()),
        "finite": bool(torch.isfinite(last).all().item()),
    }


def verify_checkpoint(
    checkpoint_path: str,
    config,
    *,
    device: str = "cpu",
    include_mlx: bool = True,
    write_json: bool = True,
) -> dict[str, Any]:
    ids = _make_probe_ids(int(config.vocab_size), seq_len=min(8, int(getattr(config, "max_seq_len", 8) or 8)))
    report: dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "topology": {
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_experts": int(config.num_experts),
            "num_experts_per_tok": int(config.num_experts_per_tok),
            "num_shared_experts": int(config.num_shared_experts),
        },
        "torch_cpu": {},
        "masked_drift": {},
        "warnings": [],
    }

    try:
        torch_ref = _torch_prefill(config, checkpoint_path, ids, device="cpu")
        report["torch_cpu"] = {
            "argmax": torch_ref["argmax"],
            "finite": torch_ref["finite"],
        }
        if not torch_ref["finite"]:
            report["warnings"].append("torch_cpu logits are not finite")
    except Exception as exc:
        report["torch_cpu"] = {"error": str(exc)}
        report["warnings"].append(f"torch_cpu verify failed: {exc}")
        torch_ref = None

    try:
        drift = _masked_drift(config, checkpoint_path, ids)
        report["masked_drift"] = drift
        if drift["cosine"] < 0.999 or not drift["argmax_match"]:
            report["warnings"].append(
                "no_mask vs all_available drift is too high for safe offload parity"
            )
    except Exception as exc:
        report["masked_drift"] = {"error": str(exc)}
        report["warnings"].append(f"masked drift verify failed: {exc}")

    if include_mlx and "mlx" in available_backends():
        try:
            mlx_out = _mlx_prefill(config, checkpoint_path, ids)
            mlx_report = {
                "argmax": mlx_out["argmax"],
                "finite": mlx_out["finite"],
            }
            if torch_ref is not None:
                cosine = float(F.cosine_similarity(torch_ref["logits"], mlx_out["logits"], dim=-1).mean().item())
                mlx_report["cosine_vs_torch_cpu"] = cosine
                mlx_report["argmax_match_torch_cpu"] = bool(mlx_out["argmax"] == torch_ref["argmax"])
                if cosine < 0.995 or mlx_out["argmax"] != torch_ref["argmax"]:
                    report["warnings"].append("MLX prefill logits differ from PyTorch CPU baseline")
            report["mlx"] = mlx_report
        except Exception as exc:
            report["mlx"] = {"error": str(exc)}
            report["warnings"].append(f"MLX verify failed: {exc}")

    report["ok"] = not report["warnings"]
    if write_json:
        out_path = checkpoint_path.replace(".pth", ".verify.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["verify_json"] = out_path
    return report
