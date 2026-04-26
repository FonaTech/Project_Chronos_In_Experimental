"""train_chronos_sft.py — Stage 2 (SFT) entry point."""
import argparse

import chronos.deps  # noqa
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.sft_trainer import ChronosSFTTrainer, build_sft_loader
from chronos.trainer.stage_utils import (
    add_topology_args,
    build_config_from_upstream,
    load_required_checkpoint,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--from_weight", default="chronos",
                   help="Pretrained checkpoint stem (loaded from save_dir/<stem>_<H>_moe.pth).")
    p.add_argument("--save_dir", default="out")
    p.add_argument("--steps", type=int, default=None,
                   help="If set, stop after N steps (smoke runs).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-5)
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
    p.add_argument("--lambda_router_anchor", type=float, default=0.01,
                   help="SFT default: weak anchor.")
    args = p.parse_args()
    selected_backend, resolved_device = resolve_training_device(args.device)
    args.device = resolved_device

    cfg, ckp_path, sources = build_config_from_upstream(
        args,
        default_stem="chronos",
        max_positions=args.max_seq_len,
        lambda_router_anchor=args.lambda_router_anchor,
    )
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
    loader = build_sft_loader(
        args.data_path,
        tokenizer,
        args.max_seq_len,
        args.batch_size,
        device=selected_backend if selected_backend == "mlx" else args.device,
        num_workers=args.num_workers,
    )
    if selected_backend == "mlx":
        from chronos.mlx.training import run_mlx_stage

        print(f"[SFT][MLX] Loaded pretrain weights from {ckp_path}")
        print(f"[SFT][MLX] Topology sources: {', '.join(sources)}")
        run_mlx_stage(
            stage="sft",
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
    print(f"[SFT] Loaded pretrain weights from {ckp_path}")
    print(f"[SFT] Topology sources: {', '.join(sources)}")
    print(f"[SFT] Training backend: {selected_backend}  device={args.device}")

    trainer = ChronosSFTTrainer(model, cfg, args, tokenizer)

    # Capture router reference on the first batch BEFORE training
    if cfg.lambda_router_anchor > 0:
        first_batch = next(iter(loader))
        trainer.set_calibration_batch(first_batch[0])
        print("[SFT] Router anchor reference captured.")

    iters = len(loader) if args.steps is None else min(args.steps, len(loader))
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, loader, iters, max_steps=args.steps)
        if args.steps is not None:
            break
    trainer._save(epoch=args.epochs - 1, step=iters)
    print(f"[SFT] Done. Checkpoint: {args.save_dir}/sft_{cfg.hidden_size}_moe.pth")


if __name__ == "__main__":
    main()
