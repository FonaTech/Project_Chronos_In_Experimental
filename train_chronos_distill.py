"""train_chronos_distill.py — Stage 6 distillation entry point."""
import argparse
import os

import chronos.deps  # noqa
import torch
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.distill_trainer import ChronosDistillTrainer, build_distill_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--teacher_path", required=True,
                   help="Path to a Chronos or MiniMind .pth teacher checkpoint.")
    p.add_argument("--from_weight", default="grpo",
                   help="Student starting checkpoint stem.")
    p.add_argument("--save_dir", default="out")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=96)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Weight on KD loss (vs. label CE).")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--lambda_router_anchor", type=float, default=0.05)
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
    student = ChronosForCausalLM(cfg).to(args.device)

    # Load student starting weights
    stu_ckp = os.path.join(args.save_dir, f"{args.from_weight}_{cfg.hidden_size}_moe.pth")
    if os.path.exists(stu_ckp):
        state = torch.load(stu_ckp, map_location=args.device)
        student.load_state_dict(state, strict=False)
        print(f"[Distill] Student: {stu_ckp}")
    else:
        print(f"[Distill] Student: random init ({stu_ckp} not found)")
    print(f"[Distill] Training backend: {selected_backend}  device={args.device}")

    # Load teacher — try Chronos first, then fall back to MiniMind
    teacher = ChronosForCausalLM(cfg).to(args.device)
    if os.path.exists(args.teacher_path):
        t_state = torch.load(args.teacher_path, map_location=args.device)
        teacher.load_state_dict(t_state, strict=False)
        print(f"[Distill] Teacher: {args.teacher_path}")
    else:
        raise FileNotFoundError(f"teacher not found: {args.teacher_path}")
    teacher.eval().requires_grad_(False)

    loader = build_distill_loader(args.data_path, tokenizer, args.max_seq_len, args.batch_size)
    trainer = ChronosDistillTrainer(student, teacher, cfg, args, tokenizer)

    if cfg.lambda_router_anchor > 0:
        first = next(iter(loader))
        trainer.set_calibration_batch(first[0])
        print("[Distill] Router anchor reference captured.")

    iters = len(loader) if args.steps is None else min(args.steps, len(loader))
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, loader, iters, max_steps=args.steps)
        if args.steps is not None:
            break
    trainer._save(epoch=args.epochs - 1, step=iters)
    print(f"[Distill] Done. Checkpoint: {args.save_dir}/distill_{cfg.hidden_size}_moe.pth")


if __name__ == "__main__":
    main()
