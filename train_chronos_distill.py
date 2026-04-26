"""train_chronos_distill.py — Stage 6 distillation entry point."""
import argparse
import os

import chronos.deps  # noqa
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.distill_trainer import ChronosDistillTrainer, build_distill_loader
from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_controlled,
)
from chronos.trainer.stage_utils import (
    add_topology_args,
    build_config_from_upstream,
    load_required_checkpoint,
)


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
    p.add_argument("--dtype", default="auto")
    p.add_argument("--num_workers", default="auto")
    p.add_argument("--cpu_threads", default="auto")
    p.add_argument("--cpu_budget_percent", default=100, type=float)
    add_topology_args(p, defaults=False)
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Weight on KD loss (vs. label CE).")
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--lambda_router_anchor", type=float, default=0.05)
    args = p.parse_args()
    selected_backend, resolved_device = resolve_training_device(args.device)
    args.device = resolved_device

    cfg, stu_ckp, sources = build_config_from_upstream(
        args,
        default_stem="grpo",
        max_positions=args.max_seq_len,
        lambda_router_anchor=args.lambda_router_anchor,
    )
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
    loader = build_distill_loader(
        args.data_path,
        tokenizer,
        args.max_seq_len,
        args.batch_size,
        device=selected_backend if selected_backend == "mlx" else args.device,
        num_workers=args.num_workers,
    )
    if selected_backend == "mlx":
        from chronos.mlx.training import run_mlx_stage

        if not os.path.exists(args.teacher_path):
            raise FileNotFoundError(f"teacher not found: {args.teacher_path}")
        print(f"[Distill][MLX] Student: {stu_ckp}")
        print(f"[Distill][MLX] Teacher: {args.teacher_path}")
        print(f"[Distill][MLX] Topology sources: {', '.join(sources)}")
        run_mlx_stage(
            stage="distill",
            config=cfg,
            checkpoint_path=stu_ckp,
            teacher_path=args.teacher_path,
            save_dir=args.save_dir,
            loader=loader,
            args=args,
        )
        return

    student = ChronosForCausalLM(cfg)
    load_required_checkpoint(student, stu_ckp, args.device)
    student = student.to(args.device)
    print(f"[Distill] Student: {stu_ckp}")
    print(f"[Distill] Topology sources: {', '.join(sources)}")
    print(f"[Distill] Training backend: {selected_backend}  device={args.device}")

    if not os.path.exists(args.teacher_path):
        raise FileNotFoundError(f"teacher not found: {args.teacher_path}")
    teacher_cfg, teacher_sources = chronos_config_from_checkpoint(
        args.teacher_path,
        overrides={"max_position_embeddings": args.max_seq_len},
        require_unsniffable=True,
    )
    teacher = ChronosForCausalLM(teacher_cfg)
    t_state = load_checkpoint_state_dict(args.teacher_path, map_location="cpu")
    load_state_dict_controlled(teacher, t_state)
    teacher = teacher.to(args.device)
    print(f"[Distill] Teacher: {args.teacher_path}")
    print(f"[Distill] Teacher topology sources: {', '.join(teacher_sources)}")
    teacher.eval().requires_grad_(False)

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
