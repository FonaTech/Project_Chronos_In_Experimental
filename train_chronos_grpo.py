"""train_chronos_grpo.py — Stage 5 (GRPO) entry point."""
import argparse
import os

import chronos.deps  # noqa
import torch
from transformers import AutoTokenizer

from chronos.backend import resolve_training_device
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.trainer.grpo_trainer import ChronosGRPOTrainer, load_grpo_prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--from_weight", default="orpo")
    p.add_argument("--save_dir", default="out")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_seq_len", type=int, default=96)
    p.add_argument("--max_gen_len", type=int, default=24)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10000)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty weight")
    p.add_argument("--lambda_router_anchor", type=float, default=0.1)
    p.add_argument("--reward", type=str, default="toy",
                   help="Reward spec: 'toy' (default) or 'lm:/path/to/reward-model'.")
    args = p.parse_args()
    selected_backend, resolved_device = resolve_training_device(args.device)
    args.device = resolved_device

    cfg = ChronosConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_experts=args.num_experts,
        num_experts_per_tok=1,
        max_position_embeddings=args.max_seq_len + args.max_gen_len,
        flash_attn=False,
        lambda_router_anchor=args.lambda_router_anchor,
    )
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
    model = ChronosForCausalLM(cfg).to(args.device)

    ckp_path = os.path.join(args.save_dir, f"{args.from_weight}_{cfg.hidden_size}_moe.pth")
    if os.path.exists(ckp_path):
        state = torch.load(ckp_path, map_location=args.device)
        model.load_state_dict(state, strict=False)
        print(f"[GRPO] Loaded {ckp_path}")
    else:
        print(f"[GRPO] No checkpoint at {ckp_path} — training from random init.")
    print(f"[GRPO] Training backend: {selected_backend}  device={args.device}")

    prompts = load_grpo_prompts(args.data_path, max_prompts=args.steps)
    from chronos.trainer.reward import build_reward_fn
    reward_fn = build_reward_fn(args.reward)
    trainer = ChronosGRPOTrainer(model, cfg, args, tokenizer, reward_fn=reward_fn)

    if cfg.lambda_router_anchor > 0 and prompts:
        # Build a calibration batch from the first prompt
        ptext = prompts[0]
        calib = tokenizer(ptext, return_tensors="pt", truncation=True,
                          max_length=args.max_seq_len).input_ids
        trainer.set_calibration_batch(calib)
        print("[GRPO] Router anchor reference captured.")

    iters = len(prompts) if args.steps is None else min(args.steps, len(prompts))
    for epoch in range(args.epochs):
        trainer.train_epoch(epoch, prompts, iters, max_steps=args.steps)
        if args.steps is not None:
            break
    trainer._save(epoch=args.epochs - 1, step=iters)
    print(f"[GRPO] Done. Checkpoint: {args.save_dir}/grpo_{cfg.hidden_size}_moe.pth")


if __name__ == "__main__":
    main()
