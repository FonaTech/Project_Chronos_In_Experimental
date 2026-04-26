"""train_chronos_orpo.py — Stage 4 (ORPO) entry point."""
import argparse

import chronos.deps  # noqa
import torch
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.orpo_trainer import ChronosORPOTrainer, build_orpo_loader
from chronos.trainer.stage_utils import (
    add_topology_args,
    build_config_from_upstream,
    load_required_checkpoint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--from_weight", default="sft")
    p.add_argument("--save_dir", default="out")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--num_workers", default="auto")
    p.add_argument("--cpu_threads", default="auto")
    p.add_argument("--cpu_budget_percent", default=100, type=float)
    add_topology_args(p, defaults=False)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lambda_or", type=float, default=0.1)
    p.add_argument("--lambda_router_anchor", type=float, default=0.1)
    args = p.parse_args()
    selected_backend, resolved_device = resolve_training_device(args.device)
    args.device = resolved_device

    cfg, ckp_path, sources = build_config_from_upstream(
        args,
        default_stem="sft",
        max_positions=args.max_seq_len,
        lambda_router_anchor=args.lambda_router_anchor,
    )
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
    loader = build_orpo_loader(
        args.data_path,
        tokenizer,
        args.max_seq_len,
        args.batch_size,
        device=selected_backend if selected_backend == "mlx" else args.device,
        num_workers=args.num_workers,
    )
    if selected_backend == "mlx":
        from chronos.mlx.training import run_mlx_stage

        print(f"[ORPO][MLX] Loaded {ckp_path}")
        print(f"[ORPO][MLX] Topology sources: {', '.join(sources)}")
        run_mlx_stage(
            stage="orpo",
            config=cfg,
            checkpoint_path=ckp_path,
            save_dir=args.save_dir,
            loader=loader,
            args=args,
        )
        return

    model = ChronosForCausalLM(cfg)
    load_required_checkpoint(model, ckp_path, args.device)
    model = model.to(args.device)
    print(f"[ORPO] Loaded {ckp_path}")
    print(f"[ORPO] Topology sources: {', '.join(sources)}")
    print(f"[ORPO] Training backend: {selected_backend}  device={args.device}")

    trainer = ChronosORPOTrainer(model, cfg, args, tokenizer)

    if cfg.lambda_router_anchor > 0:
        first = next(iter(loader))
        calib = torch.cat([first["x_chosen"], first["x_rejected"]], dim=0)
        trainer.set_calibration_batch(calib)
        print("[ORPO] Router anchor reference captured.")

    iters = len(loader) if args.steps is None else min(args.steps, len(loader))
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, loader, iters, max_steps=args.steps)
        if args.steps is not None:
            break
    trainer._save(epoch=args.epochs - 1, step=iters)
    print(f"[ORPO] Done. Checkpoint: {args.save_dir}/orpo_{cfg.hidden_size}_moe.pth")


if __name__ == "__main__":
    main()
