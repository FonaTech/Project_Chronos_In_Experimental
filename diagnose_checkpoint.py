"""Diagnose a Chronos checkpoint with offload/predictive-routing checks.

The report focuses on the Chronos-specific behavior:
- checkpoint topology and metadata source
- chat-template generation
- no-mask vs all-available masked logit drift
- all-cold shared fallback behavior
- router entropy and expert usage
- LookaheadRouter t+1/t+2 accuracy
- SSD/RAM/VRAM offload pipeline stats from ChronosInferenceEngine
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import chronos.deps  # noqa: F401
from chronos.backend import resolve_training_device
from chronos.data.flexible_dataset import FlexibleDataset, StreamingSFTDataset
from chronos.eval.io_profiler import evaluate_lookahead_accuracy, estimate_cache_hit_rate
from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_controlled,
)
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.moe_chronos import ChronosMOEFeedForward, LazyExpertPlaceholder
from chronos.runtime.inference_engine import ChronosInferenceEngine


def _encode_chat(tokenizer, prompt: str, device: str, max_length: int = 256):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def _greedy_decode(model, tokenizer, input_ids, attention_mask=None, max_new_tokens=64, masks=None):
    gen = input_ids.clone()
    attn = attention_mask.clone() if attention_mask is not None else None
    prompt_len = gen.shape[1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out, _ = model(
                gen,
                attention_mask=attn,
                use_cache=False,
                available_expert_masks=masks,
            )
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nxt], dim=-1)
            if attn is not None:
                attn = torch.cat([attn, attn.new_ones(attn.shape[0], 1)], dim=-1)
            if int(nxt.item()) == (tokenizer.eos_token_id or -1):
                break
    return tokenizer.decode(gen[0, prompt_len:].tolist(), skip_special_tokens=True)


def _router_stats(model) -> dict:
    entropies = []
    usage = Counter()
    for layer in model.model.layers:
        moe = getattr(layer, "mlp", None)
        if not isinstance(moe, ChronosMOEFeedForward) or moe.last_router_probs is None:
            continue
        probs = moe.last_router_probs.detach().float()
        entropy = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        entropies.append(float(entropy.item()))
        top1 = probs.argmax(dim=-1).reshape(-1).cpu().tolist()
        usage.update(int(i) for i in top1)
    total = sum(usage.values()) or 1
    return {
        "router_entropy_mean": sum(entropies) / max(len(entropies), 1),
        "expert_usage_top1": {str(k): round(v / total, 4) for k, v in sorted(usage.items())},
    }


def _masked_drift(model, input_ids, attention_mask, config) -> dict:
    n_layers = int(getattr(config, "num_hidden_layers", 0))
    n_experts = int(getattr(config, "num_experts", 0))
    all_true = torch.ones(n_experts, dtype=torch.bool, device=input_ids.device)
    masks = [all_true] * n_layers
    with torch.no_grad():
        no_mask, _ = model(input_ids, attention_mask=attention_mask, use_cache=False)
        all_avail, _ = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            available_expert_masks=masks,
        )
    a = no_mask.logits[:, -1, :].float()
    b = all_avail.logits[:, -1, :].float()
    return {
        "no_mask_vs_all_avail_cosine": float(F.cosine_similarity(a, b, dim=-1).mean().item()),
        "no_mask_vs_all_avail_argmax_match": bool(torch.argmax(a, dim=-1).eq(torch.argmax(b, dim=-1)).all().item()),
        "no_mask_vs_all_avail_max_abs": float((a - b).abs().max().item()),
    }


def _make_lazy_model_for_engine(config, state, device):
    model = ChronosForCausalLM(config)
    base_only = {k: v for k, v in state.items() if ".mlp.experts." not in k}
    load_state_dict_controlled(
        model,
        base_only,
        allow_missing_substrings=(".mlp.experts.",),
    )
    for layer in model.model.layers:
        moe = getattr(layer, "mlp", None)
        if not isinstance(moe, ChronosMOEFeedForward):
            continue
        for idx in range(len(moe.experts)):
            moe.experts[idx] = LazyExpertPlaceholder(
                config.hidden_size, config.moe_intermediate_size, torch.float16
            )
    return model.to(device).eval()


def _maybe_loader(data_path: str, tokenizer, max_seq_len: int, dataset_kind: str, batch_size: int):
    if not data_path or not os.path.exists(data_path):
        return None
    if dataset_kind == "sft":
        ds = StreamingSFTDataset(data_path, tokenizer, max_length=max_seq_len)
    else:
        ds = FlexibleDataset(data_path, tokenizer, max_length=max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _ce(model, loader, device, max_batches=20):
    if loader is None:
        return None
    total_loss, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, (ids, labels) in enumerate(loader):
            if i >= max_batches:
                break
            ids = ids.to(device)
            labels = labels.to(device)
            out, _ = model(ids, labels=labels)
            n = int((labels != -100).sum().item())
            if n == 0:
                continue
            total_loss += float(out.loss.item()) * n
            total_tokens += n
    if total_tokens == 0:
        return None
    return total_loss / total_tokens


def diagnose(args) -> dict:
    _backend, device = resolve_training_device(args.device)
    overrides = {
        "max_position_embeddings": max(args.max_seq_len, len(args.prompt) + args.max_new_tokens + 32),
    }
    config, sources = chronos_config_from_checkpoint(
        args.model_path,
        project_config_path=args.config_path,
        overrides=overrides,
        require_unsniffable=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
    state = load_checkpoint_state_dict(args.model_path, map_location="cpu")
    model = ChronosForCausalLM(config)
    load_state_dict_controlled(model, state)
    model = model.to(device).eval()

    input_ids, attention_mask = _encode_chat(
        tokenizer, args.prompt, device, max_length=args.max_seq_len
    )
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask, use_cache=False)
    report = {
        "checkpoint": args.model_path,
        "config_sources": sources,
        "topology": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "num_shared_experts": config.num_shared_experts,
            "moe_intermediate_size": config.moe_intermediate_size,
            "lookahead_steps": config.lookahead_steps,
            "kv_latent_dim": config.kv_latent_dim,
            "rope_dim": config.rope_dim,
        },
        **_router_stats(model),
        **_masked_drift(model, input_ids, attention_mask, config),
    }
    try:
        from chronos.verify import verify_checkpoint

        report["post_training_verify"] = verify_checkpoint(
            args.model_path,
            config,
            device="cpu",
            include_mlx=bool(args.mlx_parity),
            write_json=False,
        )
    except Exception as exc:
        report["post_training_verify_error"] = str(exc)

    n_layers = int(config.num_hidden_layers)
    cold = torch.zeros(int(config.num_experts), dtype=torch.bool, device=device)
    cold_masks = [cold] * n_layers
    report["chat_sample"] = _greedy_decode(
        model,
        tokenizer,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
    )
    report["all_cold_fallback_sample"] = _greedy_decode(
        model,
        tokenizer,
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=min(args.max_new_tokens, 64),
        masks=cold_masks,
    )

    pre_loader = _maybe_loader(args.pretrain_data, tokenizer, args.max_seq_len, "pretrain", args.batch_size)
    sft_loader = _maybe_loader(args.sft_data, tokenizer, args.max_seq_len, "sft", args.batch_size)
    report["pretrain_ce"] = _ce(model, pre_loader, device, args.max_batches)
    report["sft_ce"] = _ce(model, sft_loader, device, args.max_batches)

    eval_loader = sft_loader or pre_loader
    if eval_loader is not None:
        try:
            report["lookahead"] = evaluate_lookahead_accuracy(
                model, eval_loader, device, lookahead_steps=int(config.lookahead_steps)
            )
            report["cache_hit_estimate"] = estimate_cache_hit_rate(
                model, eval_loader, device, vram_budget_gb=float(config.vram_budget_gb)
            )
        except Exception as exc:
            report["lookahead_error"] = str(exc)

    lazy = _make_lazy_model_for_engine(config, state, device)
    engine = ChronosInferenceEngine(lazy, config, ssd_dir=args.ssd_dir)
    try:
        engine.setup_from_state_dict(state, warm_expert_ids=[])
        out = engine.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=min(args.max_new_tokens, 64),
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        report["offload_engine_sample"] = tokenizer.decode(
            out[0, input_ids.shape[1]:].tolist(),
            skip_special_tokens=True,
        )
        report["offload_engine_stats"] = engine.last_stats
    finally:
        engine.teardown()

    return report


def main():
    p = argparse.ArgumentParser(description="Diagnose Chronos checkpoint topology and offload behavior")
    p.add_argument("--model_path", required=True)
    p.add_argument("--config_path", default=None)
    p.add_argument("--pretrain_data", default="")
    p.add_argument("--sft_data", default="")
    p.add_argument("--prompt", default="请用中文简要介绍一下 Project Chronos 的 MoE 专家预取机制。")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_batches", type=int, default=20)
    p.add_argument("--ssd_dir", default="./diagnose_expert_cache")
    p.add_argument("--mlx_parity", action="store_true", help="Also compare MLX full prefill logits against PyTorch CPU")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = p.parse_args()

    report = diagnose(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    print("\n=== Chronos Checkpoint Diagnostic ===")
    print(f"checkpoint: {report['checkpoint']}")
    print(f"config sources: {', '.join(report['config_sources'])}")
    print("topology:", json.dumps(report["topology"], ensure_ascii=False))
    print(f"router_entropy_mean: {report['router_entropy_mean']:.4f}")
    print(f"expert_usage_top1: {report['expert_usage_top1']}")
    print(
        "no_mask vs all_avail:",
        f"cos={report['no_mask_vs_all_avail_cosine']:.6f}",
        f"argmax_match={report['no_mask_vs_all_avail_argmax_match']}",
        f"max_abs={report['no_mask_vs_all_avail_max_abs']:.6g}",
    )
    if report.get("pretrain_ce") is not None:
        print(f"pretrain_ce: {report['pretrain_ce']:.4f}")
    if report.get("sft_ce") is not None:
        print(f"sft_ce: {report['sft_ce']:.4f}")
    if "lookahead" in report:
        print("lookahead:", report["lookahead"])
    if "cache_hit_estimate" in report:
        print("cache_hit_estimate:", report["cache_hit_estimate"])
    if "post_training_verify" in report:
        verify = report["post_training_verify"]
        print("verify:", json.dumps({
            "ok": verify.get("ok"),
            "warnings": verify.get("warnings", []),
            "mlx": verify.get("mlx", {}),
            "masked_drift": verify.get("masked_drift", {}),
        }, ensure_ascii=False))
    print("\nchat_sample:", report["chat_sample"][:500])
    print("\nall_cold_fallback_sample:", report["all_cold_fallback_sample"][:500])
    print("\noffload_engine_sample:", report.get("offload_engine_sample", "")[:500])
    print("offload_engine_stats:", report.get("offload_engine_stats", {}))


if __name__ == "__main__":
    main()
