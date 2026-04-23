"""train_chronos_dpo.py — Stage 3 (DPO) entry point."""
import argparse
import os

import chronos.deps  # noqa
import torch
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.dpo_trainer import ChronosDPOTrainer, build_dpo_loader


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
    p.add_argument("--dtype", default="float16")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lambda_router_anchor", type=float, default=0.1,
                   help="DPO default: strong anchor.")
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

    ckp_path = os.path.join(args.save_dir, f"{args.from_weight}_{cfg.hidden_size}_moe.pth")
    if os.path.exists(ckp_path):
        state = torch.load(ckp_path, map_location=args.device)
        model.load_state_dict(state, strict=False)
        print(f"[DPO] Loaded {ckp_path}")
    else:
        print(f"[DPO] No checkpoint at {ckp_path} — training from random init.")
    print(f"[DPO] Training backend: {selected_backend}  device={args.device}")

    loader = build_dpo_loader(args.data_path, tokenizer, args.max_seq_len, args.batch_size)
    trainer = ChronosDPOTrainer(model, cfg, args, tokenizer)

    if cfg.lambda_router_anchor > 0:
        first = next(iter(loader))
        # Use chosen+rejected concatenated as calibration input
        calib = torch.cat([first["x_chosen"], first["x_rejected"]], dim=0)
        trainer.set_calibration_batch(calib)
        print("[DPO] Router anchor reference captured.")

    iters = len(loader) if args.steps is None else min(args.steps, len(loader))
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, loader, iters, max_steps=args.steps)
        if args.steps is not None:
            break
    trainer._save(epoch=args.epochs - 1, step=iters)
    print(f"[DPO] Done. Checkpoint: {args.save_dir}/dpo_{cfg.hidden_size}_moe.pth")


if __name__ == "__main__":
    main()
