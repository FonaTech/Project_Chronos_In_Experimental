"""
train_chronos.py — Phase 1 training entry point for Project Chronos.

Usage:
    python train_chronos.py \
        --data_path /path/to/pretrain.jsonl \
        --hidden_size 512 \
        --num_hidden_layers 8 \
        --num_experts 4 \
        --lambda_temporal 1e-3 \
        --epochs 2 \
        --device cuda:0
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from trainer.trainer_utils import setup_seed, Logger
from chronos.data.flexible_dataset import FlexibleDataset as PretrainDataset
from chronos.trainer.chronos_trainer import ChronosTrainer
from chronos.trainer.stage_utils import add_topology_args, build_pretrain_config
from chronos.model.checkpoint import save_state_dict_with_config
from chronos.trainer.device_utils import configure_cpu_threads, dataloader_kwargs, runtime_summary


def parse_args():
    p = argparse.ArgumentParser(description="Project Chronos Pretraining")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./out")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--num_workers", default="auto")
    p.add_argument("--cpu_threads", default="auto")
    p.add_argument("--cpu_budget_percent", default=100, type=float)
    add_topology_args(p, defaults=True)
    # Loss coefficients
    p.add_argument("--lambda_balance", type=float, default=5e-4)
    p.add_argument("--lambda_temporal", type=float, default=1e-3)
    p.add_argument("--lambda_lookahead", type=float, default=0.1)
    p.add_argument("--lambda_lookahead_topk", type=float, default=0.05)
    p.add_argument("--fallback_mask_prob", type=float, default=0.0,
                   help="Randomly mask routed experts during training to teach the shared offload fallback.")
    # VRAM budget
    p.add_argument("--vram_budget_gb", type=float, default=4.0)
    # Pipeline compat: accept but ignore args that other Stage scripts take.
    # The 6-stage Web UI invokes every stage with the same arg signature.
    p.add_argument("--steps", type=int, default=None,
                   help="If set, cap total steps (overrides epochs·len(loader)).")
    p.add_argument("--from_weight", type=str, default="",
                   help="Ignored for Stage 1 (pretrain). Present for pipeline parity.")
    return p.parse_args()


def main():
    args = parse_args()
    selected_backend, resolved_device = resolve_training_device(args.device)
    args.device = resolved_device
    threads = configure_cpu_threads(args.cpu_threads, budget_percent=args.cpu_budget_percent)
    setup_seed(42)

    config = build_pretrain_config(args)

    tokenizer = AutoTokenizer.from_pretrained(
        chronos.deps.get_tokenizer_path()
    )
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        **dataloader_kwargs(device=args.device, num_workers=args.num_workers, shuffle=True),
    )

    if selected_backend == "mlx":
        from chronos.mlx.training import run_mlx_stage

        rt = runtime_summary(args.device, args.dtype)
        Logger(
            f"Training backend: {selected_backend}  device={args.device} "
            f"dtype={rt.dtype} autocast={rt.autocast}"
        )
        Logger(f"CPU threads: {threads}  DataLoader workers: {loader.num_workers}")
        run_mlx_stage(
            stage="pretrain",
            config=config,
            checkpoint_path=None,
            save_dir=args.save_dir,
            loader=loader,
            args=args,
        )
        return

    trainer = ChronosTrainer(config, args)
    total_params = sum(p.numel() for p in trainer.model.parameters()) / 1e6
    Logger(f"Chronos model: {total_params:.2f}M params | "
           f"experts={config.num_experts} shared={config.num_shared_experts} "
           f"lookahead={config.lookahead_steps} "
           f"λ1={config.lambda_balance} λ2={config.lambda_temporal}")
    rt = runtime_summary(args.device, args.dtype)
    Logger(
        f"Training backend: {selected_backend}  device={args.device} "
        f"dtype={rt.dtype} autocast={rt.autocast}"
    )
    Logger(f"CPU threads: {threads}  DataLoader workers: {loader.num_workers}")

    for epoch in range(args.epochs):
        iters = len(loader) if args.steps is None else min(int(args.steps), len(loader))
        # Monkey-patch the trainer epoch to honor --steps as a hard cap.
        if args.steps is not None:
            _orig_step = trainer.train_step
            step_counter = {"n": 0}
            def _capped_step(*a, **kw):
                r = _orig_step(*a, **kw)
                step_counter["n"] += 1
                if step_counter["n"] >= int(args.steps):
                    raise StopIteration
                return r
            trainer.train_step = _capped_step
        try:
            trainer.train_epoch(epoch, loader, iters)
        except StopIteration:
            Logger(f"Reached --steps={args.steps}, stopping.")
            break

    Logger("Training complete.")
    # Always save a final checkpoint at the conventional path so that
    # downstream stages (sft/dpo/orpo/grpo/distill) can pick it up via
    # --from_weight chronos.
    import os
    os.makedirs(args.save_dir, exist_ok=True)
    ckp = os.path.join(args.save_dir, f"chronos_{config.hidden_size}_moe.pth")
    save_state_dict_with_config(trainer.model, ckp, config, stage="chronos")
    Logger(f"Saved → {ckp}")


if __name__ == "__main__":
    main()
