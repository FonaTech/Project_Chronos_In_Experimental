"""Quick CPU/MPS/MLX training dtype benchmark for Chronos.

This is intentionally small. It checks that each backend can run a real
forward/backward/update step with the requested dtype policy, reports speed,
and compares the last loss against a CPU float32 reference.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from chronos.backend import training_available, resolve_training_device
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.mlx.training.trainer import run_mlx_stage
from chronos.trainer.device_utils import (
    autocast_context,
    configure_cpu_threads,
    grad_scaler,
    optimizer_step_with_scaler,
    resolve_dtype_name,
    runtime_summary,
)


def _tiny_config() -> ChronosConfig:
    return ChronosConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_experts=2,
        num_experts_per_tok=1,
        num_shared_experts=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_latent_dim=8,
        rope_dim=4,
        max_seq_len=16,
        vocab_size=64,
        use_moe=True,
        lambda_balance=0.0,
        lambda_temporal=0.0,
        lambda_lookahead=0.0,
        lambda_lookahead_topk=0.0,
    )


def _make_loader(cfg: ChronosConfig, *, batch_size: int, seq_len: int, rows: int) -> DataLoader:
    g = torch.Generator().manual_seed(1234)
    ids = torch.randint(0, cfg.vocab_size, (rows, seq_len), dtype=torch.long, generator=g)
    return DataLoader(TensorDataset(ids, ids.clone()), batch_size=batch_size, shuffle=False)


def _run_torch_backend(backend: str, dtype: str, steps: int, cpu_budget_percent: float) -> dict:
    selected, device = resolve_training_device(backend)
    if selected == "mlx":
        raise ValueError("torch backend runner cannot run mlx")
    threads = configure_cpu_threads("auto", budget_percent=cpu_budget_percent)
    torch.manual_seed(2026)
    cfg = _tiny_config()
    loader = _make_loader(cfg, batch_size=2, seq_len=8, rows=max(steps * 2, 4))
    model = ChronosForCausalLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scaler = grad_scaler(device, dtype)
    last_loss = float("nan")
    t0 = time.monotonic()
    model.train()
    for step, (ids, labels) in enumerate(loader, start=1):
        if step > steps:
            break
        ids = ids.to(device)
        labels = labels.to(device)
        with autocast_context(device, dtype):
            out, _lookahead = model(ids, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        optimizer_step_with_scaler(scaler, opt, model.parameters(), 1.0)
        last_loss = float(loss.item())
    elapsed = time.monotonic() - t0
    rt = runtime_summary(device, dtype)
    return {
        "backend": selected,
        "device": device,
        "dtype": rt.dtype,
        "requested_dtype": dtype,
        "steps": steps,
        "steps_per_sec": round(steps / max(elapsed, 1e-9), 4),
        "elapsed_s": round(elapsed, 4),
        "last_loss": last_loss,
        "finite": bool(math.isfinite(last_loss)),
        "cpu_threads": threads,
        "autocast": rt.autocast,
        "scaler": rt.scaler,
    }


def _run_mlx_backend(dtype: str, steps: int, cpu_budget_percent: float) -> dict:
    configure_cpu_threads("auto", budget_percent=cpu_budget_percent)
    torch.manual_seed(2026)
    cfg = _tiny_config()
    loader = _make_loader(cfg, batch_size=2, seq_len=8, rows=max(steps * 2, 4))
    args = SimpleNamespace(
        dtype=dtype,
        learning_rate=1e-4,
        weight_decay=0.0,
        steps=steps,
        epochs=1,
        max_seq_len=8,
        grad_clip=1.0,
    )
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="chronos_mlx_bench_") as tmp:
        result = run_mlx_stage(
            stage="pretrain",
            config=cfg,
            checkpoint_path=None,
            save_dir=tmp,
            loader=loader,
            args=args,
        )
    elapsed = time.monotonic() - t0
    return {
        "backend": "mlx",
        "device": "mlx",
        "dtype": result.dtype,
        "requested_dtype": dtype,
        "steps": int(result.steps),
        "steps_per_sec": round(int(result.steps) / max(elapsed, 1e-9), 4),
        "elapsed_s": round(elapsed, 4),
        "last_loss": float(result.last_loss),
        "finite": bool(math.isfinite(float(result.last_loss))),
        "cpu_threads": int(torch.get_num_threads() or 1),
        "autocast": False,
        "scaler": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark Chronos CPU/MPS/MLX training dtype paths")
    p.add_argument("--backends", nargs="+", default=["cpu", "mps", "mlx"])
    p.add_argument("--dtypes", nargs="+", default=["auto", "float16"])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--cpu_budget_percent", type=float, default=75)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    available = set(training_available())
    rows = []
    cpu_fp32_ref = None
    for backend in args.backends:
        if backend != "cpu" and backend not in available:
            rows.append({"backend": backend, "error": "backend not available"})
            continue
        for dtype in args.dtypes:
            try:
                if backend == "mlx":
                    row = _run_mlx_backend(dtype, args.steps, args.cpu_budget_percent)
                else:
                    row = _run_torch_backend(backend, dtype, args.steps, args.cpu_budget_percent)
                if backend == "cpu" and resolve_dtype_name("cpu", dtype) == "float32":
                    cpu_fp32_ref = row["last_loss"]
                rows.append(row)
            except Exception as exc:
                rows.append({"backend": backend, "requested_dtype": dtype, "error": str(exc)})

    if cpu_fp32_ref is not None:
        for row in rows:
            if "last_loss" in row:
                row["loss_delta_vs_cpu_fp32"] = round(float(row["last_loss"]) - cpu_fp32_ref, 6)

    if args.json:
        print(json.dumps({"rows": rows}, indent=2, ensure_ascii=False))
        return

    headers = ["backend", "dtype", "steps/s", "loss", "delta", "finite", "threads", "autocast"]
    print("\t".join(headers))
    for row in rows:
        if "error" in row:
            print(f"{row.get('backend')}\t{row.get('requested_dtype','')}\tERROR\t{row['error']}")
            continue
        print(
            f"{row['backend']}\t{row['dtype']}\t{row['steps_per_sec']:.4f}\t"
            f"{row['last_loss']:.6f}\t{row.get('loss_delta_vs_cpu_fp32', 0.0):.6f}\t"
            f"{row['finite']}\t{row['cpu_threads']}\t{row['autocast']}"
        )


if __name__ == "__main__":
    main()
