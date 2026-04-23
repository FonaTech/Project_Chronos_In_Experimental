"""train_chronos_sft.py — Stage 2 (SFT) entry point."""
import argparse
import os
import sys

import chronos.deps  # noqa
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.sft_trainer import ChronosSFTTrainer, build_sft_loader


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
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--lambda_router_anchor", type=float, default=0.01,
                   help="SFT default: weak anchor.")
    args = p.parse_args()
    selected_backend, resolved_device = resolve_training_device(args.device)
    args.device = resolved_device

    cfg = ChronosConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_experts=args.num_experts,
        num_experts_per_tok=1,
        max_position_embeddings=args.max_seq_len,
        flash_attn=False,
        lambda_router_anchor=args.lambda_router_anchor,
    )
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
    model = ChronosForCausalLM(cfg).to(args.device)

    # Load pretrain checkpoint if present
    ckp_path = os.path.join(args.save_dir, f"{args.from_weight}_{cfg.hidden_size}_moe.pth")
    if os.path.exists(ckp_path):
        state = torch.load(ckp_path, map_location=args.device)
        model.load_state_dict(state, strict=False)
        print(f"[SFT] Loaded pretrain weights from {ckp_path}")
    else:
        print(f"[SFT] No pretrain checkpoint at {ckp_path} — training from random init.")
    print(f"[SFT] Training backend: {selected_backend}  device={args.device}")

    loader = build_sft_loader(args.data_path, tokenizer, args.max_seq_len, args.batch_size)
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
