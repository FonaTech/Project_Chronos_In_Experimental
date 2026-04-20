"""
tools/compare_minimind_chronos_v3.py

Full 6-stage pipeline comparison + HF roundtrip + backend report +
M3 overlap benchmark. Extends v2 with Stage 6 (distillation) and the
M5 deliverables (HF safetensors I/O + BackendDispatcher).

Run::

    python tools/compare_minimind_chronos_v3.py \\
        --pretrain_steps 150 --align_steps 30 --distill_steps 30 \\
        --simulated_ssd_ms 30 --device cpu \\
        --output results/compare_results_v3.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import chronos.deps  # noqa: E402
from chronos.backend import BackendDispatcher  # noqa: E402

# Reuse the v2 harness entirely, then add stages.
from tools.compare_minimind_chronos_v2 import (  # noqa: E402
    _build_pretrain_loader, _build_sft_loader, _build_dpo_loader,
    train_minimind_pretrain, train_chronos_pretrain,
    train_minimind_sft, train_chronos_sft,
    train_minimind_dpo, train_chronos_dpo,
    train_chronos_orpo,
    measure_decode_tps,
    measure_activation_fraction_chronos, measure_activation_fraction_minimind,
    _expert_bytes_minimind, _chronos_vram_experts_bytes,
    _frozen_clone,
)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa
from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.hf_io import save_chronos_pretrained, load_chronos_pretrained
from chronos.trainer.distill_trainer import (
    ChronosDistillTrainer, build_distill_loader,
)
from chronos.trainer.loss_mixin import collect_router_probs, capture_reference_routing
from chronos.trainer.grpo_trainer import ChronosGRPOTrainer, load_grpo_prompts
from chronos.trainer.reward import ToyReward


def train_chronos_distill_short(student, teacher, cfg, loader, steps, lr, device):
    class _A:
        pass
    a = _A()
    a.device = device
    a.dtype = "float16"
    a.learning_rate = lr
    a.accumulation_steps = 1
    a.grad_clip = 1.0
    a.log_interval = 10
    a.epochs = 1
    a.alpha = 0.7
    a.temperature = 4.0
    a.save_dir = tempfile.mkdtemp()

    tr = ChronosDistillTrainer(student, teacher, cfg, a, tokenizer=None)
    # We skip capture_reference_routing for this short smoke — anchor
    # coefficient is 0 by default on ChronosConfig anyway.
    tr.train_epoch(0, loader, steps, max_steps=steps)
    # Return a proxy "final kd" by running one last forward
    student.eval()
    with torch.no_grad():
        for ids, labels in loader:
            student_logits = student(ids.to(device))[0].logits
            teacher_logits = teacher(ids.to(device))
            teacher_logits = (teacher_logits[0].logits if isinstance(teacher_logits, tuple)
                              else teacher_logits.logits)
            V = min(student_logits.shape[-1], teacher_logits.shape[-1])
            from chronos.trainer.distill_trainer import distillation_kl
            kd = distillation_kl(
                student_logits[..., :V], teacher_logits[..., :V], T=4.0,
            )
            student.train()
            return float(kd.item())
    student.train()
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_steps", type=int, default=100)
    p.add_argument("--align_steps", type=int, default=30)
    p.add_argument("--distill_steps", type=int, default=30)
    p.add_argument("--simulated_ssd_ms", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=96)
    p.add_argument("--hidden_size", type=int, default=192)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_experts", type=int, default=4)
    p.add_argument("--lr_pretrain", type=float, default=5e-4)
    p.add_argument("--lr_align", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str,
                   default=os.path.join(_REPO, "results/compare_results_v3.json"))
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--pretrain_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_pretrain.jsonl"))
    p.add_argument("--sft_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_sft.jsonl"))
    p.add_argument("--dpo_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_dpo.jsonl"))
    p.add_argument("--grpo_path", type=str,
                   default=os.path.join(_REPO, "tests/fixtures/tiny_grpo.jsonl"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())

    # Backend report
    bd = BackendDispatcher()
    print("\n== compare_minimind_chronos_v3 ==")
    print(bd.describe())
    print(f"   H={args.hidden_size} L={args.num_hidden_layers} E={args.num_experts}  device={args.device}\n")

    # Build both models
    mm_cfg = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=args.max_seq_len + 64,
        use_moe=True, num_experts=args.num_experts, num_experts_per_tok=1,
        flash_attn=False,
    )
    ch_cfg = ChronosConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=args.max_seq_len + 64,
        use_moe=True, num_experts=args.num_experts, num_experts_per_tok=1,
        flash_attn=False, lambda_router_anchor=0.0,
    )
    mm = MiniMindForCausalLM(mm_cfg).to(args.device)
    ch = ChronosForCausalLM(ch_cfg).to(args.device)

    results = {"backends": bd.available(), "selected_backend": bd.select()}

    # Stage 1
    print("[Stage 1] Pretrain")
    loader = _build_pretrain_loader(tokenizer, args.pretrain_path, args.batch_size, args.max_seq_len)
    mm_ce = train_minimind_pretrain(mm, loader, args.pretrain_steps, args.lr_pretrain, args.device)
    torch.manual_seed(args.seed)
    ch_ce = train_chronos_pretrain(ch, ch_cfg, loader, args.pretrain_steps, args.lr_pretrain, args.device)
    results["pretrain"] = {"minimind_ce": mm_ce, "chronos_ce": ch_ce}
    print(f"  minimind ce={mm_ce:.4f}  chronos ce={ch_ce:.4f}")

    # HF roundtrip check on Chronos pretrain checkpoint
    print("[M5e] HF safetensors roundtrip")
    with tempfile.TemporaryDirectory() as hf_dir:
        save_chronos_pretrained(ch, hf_dir)
        sample = next(iter(loader))[0][:1].to(args.device)
        ch.eval()
        with torch.no_grad():
            before = ch(sample)[0].logits.detach().clone()
        ch_loaded, _ = load_chronos_pretrained(hf_dir, device=args.device)
        ch_loaded.eval()
        with torch.no_grad():
            after = ch_loaded(sample)[0].logits.detach().clone()
        delta = float((before - after).abs().max().item())
        ch.train()
    results["hf_roundtrip"] = {"max_logit_delta": delta}
    print(f"  max logit delta: {delta:.2e}")

    # Capture pretrain reference for drift tracking
    calib_ids = next(iter(loader))[0][:2].to(args.device)
    pretrain_ref = capture_reference_routing(ch, calib_ids, args.device)

    # Stage 2 SFT
    print("[Stage 2] SFT")
    sft_loader = _build_sft_loader(tokenizer, args.sft_path, args.batch_size, args.max_seq_len)
    mm_ce = train_minimind_sft(mm, sft_loader, args.align_steps, args.lr_align, args.device)
    ch_cfg.lambda_router_anchor = 0.01
    ref = capture_reference_routing(ch, calib_ids, args.device)
    ch_ce = train_chronos_sft(ch, ch_cfg, sft_loader, args.align_steps, args.lr_align, args.device, router_ref=ref)
    results["sft"] = {"minimind_ce": mm_ce, "chronos_ce": ch_ce}
    print(f"  minimind ce={mm_ce:.4f}  chronos ce={ch_ce:.4f}")

    # Snapshot SFT weights to serve as distillation teacher later
    sft_teacher_state = {k: v.detach().clone() for k, v in ch.state_dict().items()}

    # Stage 3 DPO
    print("[Stage 3] DPO")
    dpo_loader = _build_dpo_loader(tokenizer, args.dpo_path, args.batch_size, args.max_seq_len)
    mm_dpo = train_minimind_dpo(mm, dpo_loader, args.align_steps, args.lr_align * 0.1, args.device)
    ch_cfg.lambda_router_anchor = 0.1
    ref = capture_reference_routing(ch, calib_ids, args.device)
    ch_dpo = train_chronos_dpo(ch, ch_cfg, dpo_loader, args.align_steps, args.lr_align * 0.1, args.device, router_ref=ref)
    results["dpo"] = {"minimind_dpo": mm_dpo, "chronos_dpo": ch_dpo}
    print(f"  minimind dpo={mm_dpo:.4f}  chronos dpo={ch_dpo:.4f}")

    # Router drift after DPO
    ch.eval()
    with torch.no_grad():
        _ = ch(calib_ids, use_cache=False)
        r4 = collect_router_probs(ch)
        if r4 is not None and pretrain_ref is not None:
            cur = r4.mean(dim=2)
            T = min(cur.shape[1], pretrain_ref.shape[1])
            B = min(cur.shape[0], pretrain_ref.shape[0])
            kl = (cur[:B, :T] * (
                cur[:B, :T].clamp_min(1e-9).log()
                - pretrain_ref[:B, :T].clamp_min(1e-9).log()
            )).sum(dim=-1).mean().item()
            results["dpo"]["router_kl_drift_from_pretrain"] = float(kl)
    ch.train()

    # Stage 4 ORPO (chronos only)
    print("[Stage 4] ORPO")
    ch_or = train_chronos_orpo(ch, ch_cfg, dpo_loader, args.align_steps, args.lr_align * 0.1, args.device)
    results["orpo"] = {"chronos_or": ch_or}
    print(f"  chronos or={ch_or:.4f}")

    # Stage 5 GRPO (chronos only)
    print("[Stage 5] GRPO")
    class _A: pass
    a = _A()
    a.device = args.device; a.dtype = "float16"
    a.max_seq_len = args.max_seq_len; a.max_gen_len = 16
    a.num_generations = 2; a.temperature = 1.0
    a.learning_rate = args.lr_align * 0.1
    a.accumulation_steps = 1; a.grad_clip = 1.0
    a.log_interval = 10; a.epochs = 1
    a.save_dir = tempfile.mkdtemp(); a.beta = 0.04
    ch_cfg.lambda_router_anchor = 0.1
    gt = ChronosGRPOTrainer(ch, ch_cfg, a, tokenizer, reward_fn=ToyReward())
    prompts = load_grpo_prompts(args.grpo_path, max_prompts=args.align_steps)
    gt.train_epoch(0, prompts, len(prompts), max_steps=args.align_steps)
    gt.model.eval()
    final_rewards = []
    for pt in prompts[:5]:
        ids = tokenizer(pt, return_tensors="pt", truncation=True,
                        max_length=args.max_seq_len).input_ids
        _, gen = gt._rollout(ids)
        final_rewards.append(ToyReward()(pt, gen, tokenizer))
    gt.model.train()
    results["grpo"] = {"chronos_mean_reward_post": sum(final_rewards) / max(len(final_rewards), 1)}
    print(f"  chronos mean_reward={results['grpo']['chronos_mean_reward_post']:.3f}")

    # Stage 6 Distill (chronos only; teacher = SFT-stage snapshot)
    print("[Stage 6] Distill")
    teacher = ChronosForCausalLM(ch_cfg).to(args.device)
    teacher.load_state_dict(sft_teacher_state, strict=False)
    teacher.eval().requires_grad_(False)
    # Use SFT loader as distillation data (same SFTDataset format)
    distill_loader = build_distill_loader(
        args.sft_path, tokenizer, args.max_seq_len, args.batch_size,
    )
    kd_final = train_chronos_distill_short(
        ch, teacher, ch_cfg, distill_loader,
        args.distill_steps, args.lr_align * 0.05, args.device,
    )
    results["distill"] = {"chronos_final_kd": kd_final}
    print(f"  chronos kd={kd_final:.4f}")

    # Final measurements
    print("\n[Measurements]")
    ch.eval(); mm.eval()
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, 16))
    tps_mm = measure_decode_tps(mm, prompt_ids, args.max_new_tokens, args.device)
    tps_ch = measure_decode_tps(ch, prompt_ids, args.max_new_tokens, args.device)
    frac_mm = measure_activation_fraction_minimind(mm, prompt_ids, 32, args.device)
    frac_ch = measure_activation_fraction_chronos(ch, prompt_ids, 32, args.device)
    results["measurements"] = {
        "tps_minimind": tps_mm, "tps_chronos": tps_ch,
        "activation_frac_minimind": frac_mm, "activation_frac_chronos": frac_ch,
        "resident_expert_bytes_minimind": _expert_bytes_minimind(mm),
        "resident_expert_bytes_chronos_cached": _chronos_vram_experts_bytes(ch, ch_cfg, 2),
    }

    # M3 overlap bench
    print("\n[M3 overlap bench]")
    from chronos.runtime.cache_manager import CacheManager
    ssd = tempfile.mkdtemp(prefix="chronos_v3_m3_")
    mgr = CacheManager(ch, ch_cfg, ssd_dir=ssd)
    mgr.expert_store.offload_all_to_ssd(
        clusters=[[i] for i in range(ch_cfg.num_experts)]
    )
    mgr.expert_store.attach_cluster_manifest()
    mgr.warm_up(initial_expert_ids=[0])
    mgr.start()
    E = ch_cfg.num_experts
    lp = torch.zeros(1, 1, ch_cfg.lookahead_steps + 1, E)
    lp[..., 0] = 1.0 / E
    lp[0, 0, 1, E - 1] = 1.0
    os.environ["CHRONOS_SIM_SSD_MS"] = str(args.simulated_ssd_ms)
    try:
        t0 = time.monotonic()
        mgr.prefetch_for_next_step(lp)
        async_ms = (time.monotonic() - t0) * 1000
        time.sleep(args.simulated_ssd_ms / 1000 + 0.1)
        lp2 = torch.zeros(1, 1, ch_cfg.lookahead_steps + 1, E)
        lp2[..., 0] = 1.0 / E
        fresh = (E // 2) if E >= 3 else 1
        t0 = time.monotonic()
        mgr.step(lp2, [fresh])
        legacy_ms = (time.monotonic() - t0) * 1000
    finally:
        os.environ.pop("CHRONOS_SIM_SSD_MS", None)
        mgr.stop()
    results["m3_overlap_benchmark"] = {
        "simulated_ssd_ms": args.simulated_ssd_ms,
        "prefetch_for_next_step_ms": async_ms,
        "legacy_step_ms": legacy_ms,
        "overlap_headroom_ms": legacy_ms - async_ms,
    }

    # Save + print
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("STAGE RESULTS  (v3)")
    print("-" * 78)
    for stage in ("pretrain", "sft", "dpo", "orpo", "grpo", "distill"):
        r = results[stage]
        line = f"  [{stage:<8}] " + "  ".join(
            f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, (float, int))
        )
        print(line)
    print(f"\n  [hf_roundtrip]  max_logit_delta={results['hf_roundtrip']['max_logit_delta']:.2e}")

    print("\nDECODE MEASUREMENTS")
    print("-" * 78)
    m = results["measurements"]
    print(f"  tokens/sec             minimind={m['tps_minimind']:.1f}  chronos={m['tps_chronos']:.1f}")
    print(f"  activation fraction    minimind={m['activation_frac_minimind']:.3f}  chronos={m['activation_frac_chronos']:.3f}")
    print(f"  resident expert MB     minimind={m['resident_expert_bytes_minimind']/1e6:.2f}  chronos(cached)={m['resident_expert_bytes_chronos_cached']/1e6:.2f}")

    print("\nM3 OVERLAP")
    print("-" * 78)
    m3 = results["m3_overlap_benchmark"]
    print(f"  simulated SSD latency:     {m3['simulated_ssd_ms']}ms")
    print(f"  prefetch_for_next_step:    {m3['prefetch_for_next_step_ms']:.2f}ms  (non-blocking)")
    print(f"  legacy step() with miss:   {m3['legacy_step_ms']:.2f}ms     (blocking)")
    print(f"  pipeline headroom / tok:   {m3['overlap_headroom_ms']:.2f}ms")

    print("\nBACKENDS AVAILABLE ON THIS HOST")
    print("-" * 78)
    print(f"  {results['backends']}  selected: {results['selected_backend']}")
    print("=" * 78)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
