"""Minimal but real MLX-native six-stage training support."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import torch
from mlx.utils import tree_map

from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    load_checkpoint_state_dict,
    load_state_dict_controlled,
    save_state_dict_with_config,
)
from chronos.mlx.model import ChronosMLXModel
from chronos.mlx.moe import ChronosMLXMOE
from chronos.mlx.training.io import mlx_state_to_torch
from chronos.trainer.optim_utils import get_lr


def _ids_to_mx(x) -> mx.array:
    if isinstance(x, mx.array):
        return x.astype(mx.int32)
    if isinstance(x, torch.Tensor):
        return mx.array(x.detach().cpu().numpy()).astype(mx.int32)
    return mx.array(x).astype(mx.int32)


def _masked_lm_loss(logits: mx.array, labels: mx.array) -> mx.array:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1]).astype(mx.float32)
    flat_labels = shift_labels.reshape(-1)
    mask = flat_labels != -100
    safe_labels = mx.where(mask, flat_labels, mx.zeros_like(flat_labels))
    losses = nn.losses.cross_entropy(flat_logits, safe_labels, reduction="none")
    return (losses * mask.astype(losses.dtype)).sum() / mx.maximum(mask.astype(mx.float32).sum(), 1.0)


def _gather_token_logprobs(logits: mx.array, labels: mx.array) -> mx.array:
    labels = labels.astype(mx.int32)
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    flat = log_probs.reshape(-1, log_probs.shape[-1])
    idx = labels.reshape(-1)
    rows = mx.arange(idx.shape[0])
    return flat[rows, idx].reshape(labels.shape)


def _aligned_sequence_logprob(
    logits: mx.array,
    labels: mx.array,
    mask: mx.array,
    *,
    average: bool = False,
) -> mx.array:
    tok_lp = _gather_token_logprobs(logits, labels)
    mask = mask.astype(mx.float32)
    total = (tok_lp * mask).sum(axis=1)
    if average:
        return total / mx.maximum(mask.sum(axis=1), 1.0)
    return total


def _shifted_token_logprobs(logits: mx.array, ids: mx.array) -> mx.array:
    return _gather_token_logprobs(logits[:, :-1, :], ids[:, 1:])


def _collect_router_probs(model) -> mx.array | None:
    probs = []
    for layer in getattr(model, "layers", []):
        moe = getattr(layer, "mlp", None)
        if isinstance(moe, ChronosMLXMOE) and moe.last_router_probs is not None:
            probs.append(moe.last_router_probs)
    if not probs:
        return None
    return mx.stack(probs, axis=2)  # [B, S, L, E]


def _temporal_locality_loss(router_mean: mx.array) -> mx.array:
    if router_mean.shape[1] <= 1:
        return mx.zeros(())
    diff = router_mean[:, 1:, :] - router_mean[:, :-1, :]
    return (diff * diff).sum(axis=-1).mean()


def _load_balance_loss(router_4d: mx.array, num_experts_per_tok: int) -> mx.array:
    B, S, L, E = router_4d.shape
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    total = mx.zeros(())
    for li in range(L):
        scores = router_4d[:, :, li, :].reshape(B * S, E)
        topk_idx = mx.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k]
        selected = (topk_idx[..., None] == mx.arange(E)).astype(scores.dtype)
        load = selected.mean(axis=0)  # [K, E], matching the PyTorch trainer
        total = total + (load * scores.mean(axis=0)).sum() * E
    return total


def _lookahead_supervision_loss(
    lookahead_probs: mx.array | None,
    teacher_router_probs: mx.array | None,
    lookahead_steps: int,
) -> mx.array:
    if lookahead_probs is None:
        return mx.zeros(())
    if teacher_router_probs is None:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    _B, S, Kp1, _E = lookahead_probs.shape
    K = min(int(lookahead_steps or 0), Kp1 - 1)
    if K <= 0 or S <= 1:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    teacher = mx.stop_gradient(teacher_router_probs.astype(mx.float32))
    total = mx.zeros((), dtype=mx.float32)
    terms = 0
    for k in range(1, K + 1):
        if S - k <= 0:
            continue
        teacher_k = teacher[:, k:, :]
        pred_k = lookahead_probs[:, :-k, k, :].astype(mx.float32)
        total = total + (-(teacher_k * mx.log(mx.maximum(pred_k, 1e-9))).sum(axis=-1)).mean()
        terms += 1
    return total / max(terms, 1)


def _lookahead_topk_hit_loss(
    lookahead_probs: mx.array | None,
    teacher_router_probs: mx.array | None,
    lookahead_steps: int,
    num_experts_per_tok: int,
) -> mx.array:
    if lookahead_probs is None:
        return mx.zeros(())
    if teacher_router_probs is None:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    _B, S, Kp1, E = lookahead_probs.shape
    K = min(int(lookahead_steps or 0), Kp1 - 1)
    top_k = max(1, min(int(num_experts_per_tok or 1), E))
    if K <= 0 or S <= 1:
        return mx.zeros((), dtype=lookahead_probs.dtype)
    teacher = mx.stop_gradient(teacher_router_probs.astype(mx.float32))
    total = mx.zeros((), dtype=mx.float32)
    terms = 0
    for k in range(1, K + 1):
        if S - k <= 0:
            continue
        teacher_k = teacher[:, k:, :]
        pred_k = lookahead_probs[:, :-k, k, :].astype(mx.float32)
        target_ids = mx.argpartition(-teacher_k, kth=top_k - 1, axis=-1)[:, :, :top_k]
        target_mask = (target_ids[..., None] == mx.arange(E)).astype(pred_k.dtype).sum(axis=-2)
        hit_mass = (pred_k * target_mask).sum(axis=-1)
        total = total + (-mx.log(mx.maximum(hit_mass / float(top_k), 1e-9))).mean()
        terms += 1
    return total / max(terms, 1)


def _router_kl_anchor(current: mx.array, reference: mx.array | None, lambda_anchor: float) -> mx.array:
    if reference is None or lambda_anchor <= 0.0:
        return mx.zeros((), dtype=current.dtype)
    B = min(current.shape[0], reference.shape[0])
    S = min(current.shape[1], reference.shape[1])
    cur = current[:B, :S, :].astype(mx.float32)
    ref = mx.stop_gradient(reference[:B, :S, :].astype(mx.float32))
    eps = 1e-9
    kl = (cur * (mx.log(mx.maximum(cur, eps)) - mx.log(mx.maximum(ref, eps)))).sum(axis=-1).mean()
    return float(lambda_anchor) * kl


def _chronos_regularized_loss(model, base_loss: mx.array, lookahead_probs: mx.array | None, config, router_ref=None) -> mx.array:
    loss = base_loss.astype(mx.float32)
    router_4d = _collect_router_probs(model)
    if router_4d is None:
        return loss
    router_4d = router_4d.astype(mx.float32)

    lambda_balance = float(getattr(config, "lambda_balance", 0.0) or 0.0)
    if lambda_balance > 0.0:
        loss = loss + lambda_balance * _load_balance_loss(
            router_4d,
            int(getattr(config, "num_experts_per_tok", 1) or 1),
        )

    if router_4d.shape[1] > 1:
        router_mean = router_4d.mean(axis=2)
        lambda_temporal = float(getattr(config, "lambda_temporal", 0.0) or 0.0)
        if lambda_temporal > 0.0:
            loss = loss + lambda_temporal * _temporal_locality_loss(router_mean)

        lookahead_steps = int(getattr(config, "lookahead_steps", 0) or 0)
        lambda_lookahead = float(getattr(config, "lambda_lookahead", 0.0) or 0.0)
        if lookahead_probs is not None and lambda_lookahead > 0.0 and lookahead_steps > 0:
            teacher = mx.stop_gradient(router_mean)
            loss = loss + lambda_lookahead * _lookahead_supervision_loss(
                lookahead_probs,
                teacher,
                lookahead_steps,
            )
            lambda_topk = float(getattr(config, "lambda_lookahead_topk", 0.0) or 0.0)
            if lambda_topk > 0.0:
                loss = loss + lambda_topk * _lookahead_topk_hit_loss(
                    lookahead_probs,
                    teacher,
                    lookahead_steps,
                    int(getattr(config, "num_experts_per_tok", 1) or 1),
                )

        loss = loss + _router_kl_anchor(
            router_mean,
            router_ref,
            float(getattr(config, "lambda_router_anchor", 0.0) or 0.0),
        )
    return loss


def _ce_stage_loss(model, input_ids, labels):
    logits, lookahead, _cache = model(input_ids)
    base = _masked_lm_loss(logits, labels)
    return _chronos_regularized_loss(model, base, lookahead, model.config)


def _dpo_stage_loss(model, reference, batch, beta: float = 0.1, router_ref=None):
    x_c = _ids_to_mx(batch["x_chosen"])
    x_r = _ids_to_mx(batch["x_rejected"])
    y_c = _ids_to_mx(batch["y_chosen"])
    y_r = _ids_to_mx(batch["y_rejected"])
    m_c = _ids_to_mx(batch["mask_chosen"])
    m_r = _ids_to_mx(batch["mask_rejected"])
    x = mx.concatenate([x_c, x_r], axis=0)
    logits, lookahead, _cache = model(x)
    B = x_c.shape[0]
    lp_c = _aligned_sequence_logprob(logits[:B], y_c, m_c)
    lp_r = _aligned_sequence_logprob(logits[B:], y_r, m_r)
    if reference is not None:
        ref_logits, _ref_lp, _ref_cache = reference(x)
        ref_c = mx.stop_gradient(_aligned_sequence_logprob(ref_logits[:B], y_c, m_c))
        ref_r = mx.stop_gradient(_aligned_sequence_logprob(ref_logits[B:], y_r, m_r))
        base = -mx.mean(nn.log_sigmoid(beta * ((lp_c - lp_r) - (ref_c - ref_r))))
    else:
        base = -mx.mean(nn.log_sigmoid(beta * (lp_c - lp_r)))
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref)


def _orpo_stage_loss(model, batch, beta: float = 0.1, lambda_or: float = 0.1, router_ref=None):
    x_c = _ids_to_mx(batch["x_chosen"])
    x_r = _ids_to_mx(batch["x_rejected"])
    y_c = _ids_to_mx(batch["y_chosen"])
    y_r = _ids_to_mx(batch["y_rejected"])
    m_c = _ids_to_mx(batch["mask_chosen"])
    m_r = _ids_to_mx(batch["mask_rejected"])
    x = mx.concatenate([x_c, x_r], axis=0)
    logits, lookahead, _cache = model(x)
    B = x_c.shape[0]
    lp_c = _aligned_sequence_logprob(logits[:B], y_c, m_c, average=True)
    lp_r = _aligned_sequence_logprob(logits[B:], y_r, m_r, average=True)
    nll = -mx.mean(lp_c)
    p_c = mx.clip(mx.exp(lp_c), 1e-6, 1.0 - 1e-6)
    p_r = mx.clip(mx.exp(lp_r), 1e-6, 1.0 - 1e-6)
    odds = (mx.log(p_c) - mx.log1p(-p_c)) - (
        mx.log(p_r) - mx.log1p(-p_r)
    )
    base = nll - lambda_or * mx.mean(nn.log_sigmoid(beta * odds))
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref)


def _distill_stage_loss(
    model,
    teacher,
    input_ids,
    labels,
    teacher_input_ids=None,
    alpha: float = 0.7,
    temperature: float = 4.0,
    router_ref=None,
):
    logits, lookahead, _cache = model(input_ids)
    t_logits, _tlp, _tcache = teacher(teacher_input_ids if teacher_input_ids is not None else input_ids)
    V = min(logits.shape[-1], t_logits.shape[-1])
    S = min(logits.shape[1], t_logits.shape[1])
    logits_kd = logits[:, :S, :V]
    t_logits = mx.stop_gradient(t_logits[:, :S, :V])
    logits_kd = logits_kd.astype(mx.float32)
    t_logits = t_logits.astype(mx.float32)
    s_logp = logits_kd / temperature - mx.logsumexp(logits_kd / temperature, axis=-1, keepdims=True)
    t_prob = mx.softmax(t_logits / temperature, axis=-1)
    kd_tok = mx.sum(t_prob * (mx.log(mx.maximum(t_prob, 1e-12)) - s_logp), axis=-1)
    mask = (labels[:, :S] != -100).astype(mx.float32)
    kd = (temperature * temperature) * (kd_tok * mask).sum() / mx.maximum(mask.sum(), 1.0)
    ce = _masked_lm_loss(logits, labels)
    base = alpha * kd + (1.0 - alpha) * ce
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref)


def _fit_teacher_input(ids: mx.array, teacher_vocab_size: int) -> mx.array:
    return mx.where(ids >= int(teacher_vocab_size), ids % int(teacher_vocab_size), ids)


def _grpo_stage_loss(model, reference, batch, beta: float = 0.04, router_ref=None):
    ids = _ids_to_mx(batch["ids"])
    response_mask = mx.array(batch["response_mask"]).astype(mx.float32)
    advantages = mx.array(batch["advantages"]).astype(mx.float32)
    logits, lookahead, _cache = model(ids)
    lp_tok = _shifted_token_logprobs(logits, ids)
    mask = response_mask[:, 1:]
    seq_lp = (lp_tok * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
    pg_loss = -mx.mean(mx.stop_gradient(advantages) * seq_lp)

    kl = mx.zeros((), dtype=logits.dtype)
    if reference is not None:
        ref_logits, _ref_lp, _ref_cache = reference(ids)
        ref_lp_tok = mx.stop_gradient(_shifted_token_logprobs(ref_logits, ids))
        log_ratio = ref_lp_tok - lp_tok
        kl = ((mx.exp(log_ratio) - log_ratio - 1.0) * mask).sum() / mx.maximum(mask.sum(), 1.0)
    base = pg_loss + float(beta) * kl
    return _chronos_regularized_loss(model, base, lookahead, model.config, router_ref=router_ref)


def _capture_router_ref(model, batch, stage: str) -> mx.array | None:
    if stage in {"pretrain", "grpo"}:
        return None
    if stage in {"sft", "distill"}:
        ids = _ids_to_mx(batch[0])
    elif stage in {"dpo", "orpo"}:
        ids = mx.concatenate([_ids_to_mx(batch["x_chosen"]), _ids_to_mx(batch["x_rejected"])], axis=0)
    else:
        return None
    _logits, _lookahead, _cache = model(ids)
    router_4d = _collect_router_probs(model)
    if router_4d is None:
        return None
    ref = mx.stop_gradient(router_4d.mean(axis=2))
    mx.eval(ref)
    return ref


def _sample_next(logits: mx.array, temperature: float) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    probs = mx.softmax(logits / max(float(temperature), 1e-6), axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-9)).astype(mx.int32)


def _rollout_one(model, prompt_ids: mx.array, max_gen_len: int, temperature: float, eos_token_id: int | None) -> tuple[mx.array, list[int]]:
    ids = prompt_ids
    cache = None
    generated: list[int] = []
    for _ in range(max(1, int(max_gen_len))):
        token_in = ids if cache is None else ids[:, -1:]
        logits, _lookahead, cache = model(token_in, cache=cache)
        next_id = _sample_next(logits[:, -1, :], temperature).reshape(1, 1)
        mx.eval(next_id)
        tok = int(next_id.item())
        generated.append(tok)
        ids = mx.concatenate([ids, next_id], axis=1)
        if eos_token_id is not None and tok == int(eos_token_id):
            break
    mx.eval(ids)
    return ids, generated


def _build_grpo_batch(model, prompt: str, tokenizer, reward_fn, args) -> dict:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    max_seq_len = int(getattr(args, "max_seq_len", 96) if args is not None else 96)
    max_gen_len = int(getattr(args, "max_gen_len", 24) if args is not None else 24)
    num_generations = int(getattr(args, "num_generations", 4) if args is not None else 4)
    temperature = float(getattr(args, "temperature", 1.0) if args is not None else 1.0)
    enc = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_seq_len)
    prompt_ids = _ids_to_mx(enc.input_ids)
    prompt_len = int(prompt_ids.shape[1])
    trajectories = []
    rewards = []
    for _ in range(max(1, num_generations)):
        full_ids, response_tokens = _rollout_one(
            model,
            prompt_ids,
            max_gen_len=max_gen_len,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
        trajectories.append(full_ids)
        rewards.append(float(reward_fn(prompt, response_tokens, tokenizer)))

    max_len = max(int(t.shape[1]) for t in trajectories)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ids_rows = []
    mask_rows = []
    for full_ids in trajectories:
        row = [int(v) for v in full_ids[0].tolist()]
        pad = [pad_id] * (max_len - len(row))
        ids_rows.append(row + pad)
        mask_rows.append(([0] * prompt_len + [1] * max(0, len(row) - prompt_len)) + [0] * len(pad))
    rewards_mx = mx.array(rewards, dtype=mx.float32)
    adv = rewards_mx - rewards_mx.mean()
    denom = mx.maximum(mx.sqrt(((adv * adv).mean())), 1e-6)
    advantages = adv / denom
    mx.eval(advantages)
    return {
        "ids": ids_rows,
        "response_mask": mask_rows,
        "advantages": advantages.tolist(),
    }


@dataclass
class MLXStageResult:
    checkpoint_path: str
    steps: int
    last_loss: float
    dtype: str
    elapsed_s: float = 0.0
    total_steps: int = 0
    stopped: bool = False
    checkpoint_saved: bool = False
    rollbacks: int = 0


def _is_mx_array(value) -> bool:
    return isinstance(value, mx.array)


def _tree_copy(tree):
    if isinstance(tree, dict):
        return {k: _tree_copy(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_tree_copy(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_copy(v) for v in tree)
    if _is_mx_array(tree):
        copied = mx.array(tree)
        mx.eval(copied)
        return copied
    return tree


def _tree_arrays(tree) -> list[mx.array]:
    arrays: list[mx.array] = []
    if isinstance(tree, dict):
        for value in tree.values():
            arrays.extend(_tree_arrays(value))
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            arrays.extend(_tree_arrays(value))
    elif _is_mx_array(tree):
        arrays.append(tree)
    return arrays


def _tree_all_finite(tree) -> bool:
    arrays = _tree_arrays(tree)
    if not arrays:
        return True
    checks = [mx.all(mx.isfinite(arr)) for arr in arrays]
    mx.eval(*checks)
    return all(bool(check.item()) for check in checks)


def _scalar_is_finite(value: mx.array) -> bool:
    check = mx.all(mx.isfinite(value))
    mx.eval(check)
    return bool(check.item())


def _add_grads(a, b):
    if a is None:
        return _tree_copy(b)
    return tree_map(lambda x, y: x + y, a, b)


def _scale_grads(grads, scale: float):
    return tree_map(lambda g: g * float(scale), grads)


def _planned_total_steps(data_iter, epochs: int, max_steps) -> int:
    planned = 0
    try:
        planned = max(1, int(len(data_iter)) * max(1, int(epochs)))
    except Exception:
        planned = 0
    if max_steps is not None:
        cap = max(1, int(max_steps))
        return min(planned, cap) if planned else cap
    return planned


def _normalize_mlx_dtype_name(dtype_name: str | None) -> str:
    value = (dtype_name or "auto").strip().lower()
    if value in {"auto", ""}:
        return "bfloat16" if hasattr(mx, "bfloat16") else "float32"
    if value in {"fp16", "float16", "half"}:
        return "float16"
    if value in {"bf16", "bfloat16"}:
        return "bfloat16"
    return "float32"


def _mlx_dtype_from_name(dtype_name: str | None):
    name = _normalize_mlx_dtype_name(dtype_name)
    if name == "float16":
        return mx.float16
    if name == "bfloat16":
        return mx.bfloat16 if hasattr(mx, "bfloat16") else mx.float32
    return mx.float32


def _load_mlx_from_checkpoint(config, checkpoint_path: str | None, dtype_name: str | None = None):
    pt = ChronosForCausalLM(config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
        load_state_dict_controlled(pt, state)
    model = ChronosMLXModel.from_chronos_pytorch(pt, config)
    dtype = _mlx_dtype_from_name(dtype_name)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    model.config = config
    return model


def _save_mlx_checkpoint(model, config, out_path: str, stage: str) -> None:
    pt = ChronosForCausalLM(config)
    pt.load_state_dict(mlx_state_to_torch(model, config), strict=True)
    save_state_dict_with_config(pt, out_path, config, stage=stage)


def run_mlx_stage(
    *,
    stage: str,
    config,
    checkpoint_path: str | None,
    save_dir: str,
    loader=None,
    prompts=None,
    teacher_path: str | None = None,
    args=None,
    progress_callback=None,
    stop_event=None,
) -> MLXStageResult:
    dtype_name = _normalize_mlx_dtype_name(getattr(args, "dtype", "auto") if args is not None else "auto")
    model = _load_mlx_from_checkpoint(config, checkpoint_path, dtype_name=dtype_name)
    reference = None
    if stage in {"dpo", "grpo"} and checkpoint_path:
        reference = _load_mlx_from_checkpoint(config, checkpoint_path, dtype_name=dtype_name)
    teacher = None
    if stage == "distill":
        if not teacher_path:
            raise ValueError("MLX distill requires teacher_path")
        try:
            teacher_config, _teacher_sources = chronos_config_from_checkpoint(
                teacher_path,
                require_unsniffable=True,
            )
        except Exception:
            teacher_config = config
        teacher = _load_mlx_from_checkpoint(teacher_config, teacher_path, dtype_name=dtype_name)

    base_lr = float(getattr(args, "learning_rate", 5e-5) if args is not None else 5e-5)
    default_eps = 1e-6 if dtype_name in {"float16", "bfloat16"} else 1e-8
    optimizer = optim.AdamW(
        learning_rate=base_lr,
        weight_decay=float(getattr(args, "weight_decay", 0.01) if args is not None else 0.01),
        eps=float(getattr(args, "adam_eps", default_eps) if args is not None else default_eps),
    )
    max_steps = getattr(args, "steps", None) if args is not None else None
    epochs = int(getattr(args, "epochs", 1) if args is not None else 1)
    accum = max(1, int(getattr(args, "accumulation_steps", 1) if args is not None else 1))
    grad_clip = float(getattr(args, "grad_clip", 0.0) if args is not None else 0.0)
    save_interval = max(1, int(getattr(args, "save_interval", 1000) if args is not None else 1000))
    rollback_limit = max(1, int(getattr(args, "mlx_rollback_limit", 3) if args is not None else 3))
    beta = float(getattr(args, "beta", 0.1) if args is not None else 0.1)
    lambda_or = float(getattr(args, "lambda_or", 0.1) if args is not None else 0.1)
    alpha = float(getattr(args, "alpha", 0.7) if args is not None else 0.7)
    temperature = float(getattr(args, "temperature", 4.0) if args is not None else 4.0)
    log_interval = max(1, int(getattr(args, "log_interval", 10) if args is not None else 10))
    steps = 0
    last_loss = 0.0
    t0 = time.monotonic()
    last_log_t = t0
    last_log_step = 0
    router_ref = None

    def _should_stop() -> bool:
        return bool(stop_event is not None and stop_event.is_set())

    if stage == "grpo":
        from transformers import AutoTokenizer
        import chronos.deps
        from chronos.trainer.reward import build_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
        reward_fn = build_reward_fn(getattr(args, "reward", "toy") if args is not None else "toy")
        data_iter = prompts or []
    else:
        data_iter = loader
    if data_iter is None:
        raise ValueError(f"MLX {stage} requires loader/prompts")
    total_steps = _planned_total_steps(data_iter, epochs, max_steps)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(
        save_dir,
        f"{stage}_{config.hidden_size}_moe.pth" if stage != "pretrain" else f"chronos_{config.hidden_size}_moe.pth",
    )
    checkpoint_stage = "chronos" if stage == "pretrain" else stage
    checkpoint_saved = False
    rollback_count = 0
    consecutive_rollbacks = 0
    lr_scale = 1.0
    pending_grads = None
    pending_count = 0
    rollback_model_state = _tree_copy(model.parameters())
    rollback_optimizer_state = _tree_copy(optimizer.state)

    def _emit(event: dict) -> None:
        if progress_callback is not None:
            progress_callback(event)

    def _discard_pending() -> None:
        nonlocal pending_grads, pending_count
        pending_grads = None
        pending_count = 0

    def _restore_rollback_state() -> None:
        _discard_pending()
        model.update(_tree_copy(rollback_model_state))
        optimizer.state = _tree_copy(rollback_optimizer_state)
        mx.eval(model.parameters(), optimizer.state)

    def _save_checkpoint(reason: str) -> None:
        nonlocal checkpoint_saved
        _save_mlx_checkpoint(model, config, out_path, stage=checkpoint_stage)
        checkpoint_saved = True
        _emit({
            "event": "save",
            "stage": stage,
            "steps": total_steps,
            "step": steps,
            "loss": last_loss,
            "elapsed_s": time.monotonic() - t0,
            "dtype": dtype_name,
            "checkpoint_path": out_path,
            "reason": reason,
        })

    def loss_fn(m, batch):
        if stage in {"pretrain", "sft"}:
            ids, labels = batch
            return _ce_stage_loss(m, _ids_to_mx(ids), _ids_to_mx(labels))
        if stage == "dpo":
            return _dpo_stage_loss(m, reference, batch, beta=beta, router_ref=router_ref)
        if stage == "orpo":
            return _orpo_stage_loss(m, batch, beta=beta, lambda_or=lambda_or, router_ref=router_ref)
        if stage == "distill":
            ids, labels = batch
            ids = _ids_to_mx(ids)
            labels = _ids_to_mx(labels)
            teacher_vocab = int(getattr(getattr(teacher, "config", config), "vocab_size", config.vocab_size))
            return _distill_stage_loss(
                m,
                teacher,
                ids,
                labels,
                teacher_input_ids=_fit_teacher_input(ids, teacher_vocab),
                alpha=alpha,
                temperature=temperature,
                router_ref=router_ref,
            )
        if stage == "grpo":
            return _grpo_stage_loss(m, reference, batch, beta=beta, router_ref=router_ref)
        raise ValueError(f"unknown MLX stage: {stage}")

    grad_fn = nn.value_and_grad(model, loss_fn)
    _emit({
        "event": "start",
        "stage": stage,
        "dtype": dtype_name,
        "step": 0,
        "steps": total_steps,
        "loss": 0.0,
        "elapsed_s": 0.0,
        "lr": base_lr,
        "accumulation_steps": accum,
        "save_interval": save_interval,
    })

    stop_outer = False
    for epoch_idx in range(epochs):
        for batch in data_iter:
            if _should_stop():
                break
            if router_ref is None and stage in {"sft", "dpo", "orpo", "distill"}:
                router_ref = _capture_router_ref(model, batch, stage)
            train_batch = batch
            if stage == "grpo":
                if _should_stop():
                    break
                train_batch = _build_grpo_batch(
                    model,
                    str(batch),
                    tokenizer,
                    reward_fn,
                    args,
                )
            loss, grads = grad_fn(model, train_batch)
            mx.eval(loss, grads)
            steps += 1
            finite = _scalar_is_finite(loss) and _tree_all_finite(grads)
            if not finite:
                rollback_count += 1
                consecutive_rollbacks += 1
                lr_scale *= 0.5
                _discard_pending()
                _emit({
                    "event": "rollback",
                    "stage": stage,
                    "epoch": epoch_idx + 1,
                    "epochs": epochs,
                    "step": steps,
                    "steps": total_steps,
                    "loss": last_loss,
                    "elapsed_s": time.monotonic() - t0,
                    "dtype": dtype_name,
                    "lr": base_lr * lr_scale,
                    "rollbacks": rollback_count,
                    "reason": "non_finite_loss_or_grad",
                })
                if consecutive_rollbacks >= rollback_limit:
                    stop_outer = True
                    break
                if max_steps is not None and steps >= int(max_steps):
                    break
                continue

            last_loss = float(loss.item())
            pending_grads = _add_grads(pending_grads, grads)
            pending_count += 1
            should_update = pending_count >= accum
            if max_steps is not None and steps >= int(max_steps):
                should_update = True
            if should_update and pending_grads is not None:
                current_lr = get_lr(steps, total_steps or steps, base_lr * lr_scale)
                optimizer.learning_rate = current_lr
                update_grads = _scale_grads(pending_grads, 1.0 / max(1, pending_count))
                if grad_clip > 0.0:
                    update_grads, _norm = optim.clip_grad_norm(update_grads, grad_clip)
                    mx.eval(update_grads)
                if not _tree_all_finite(update_grads):
                    rollback_count += 1
                    consecutive_rollbacks += 1
                    lr_scale *= 0.5
                    _discard_pending()
                    _emit({
                        "event": "rollback",
                        "stage": stage,
                        "epoch": epoch_idx + 1,
                        "epochs": epochs,
                        "step": steps,
                        "steps": total_steps,
                        "loss": last_loss,
                        "elapsed_s": time.monotonic() - t0,
                        "dtype": dtype_name,
                        "lr": base_lr * lr_scale,
                        "rollbacks": rollback_count,
                        "reason": "non_finite_accumulated_grad",
                    })
                    if consecutive_rollbacks >= rollback_limit:
                        stop_outer = True
                        break
                else:
                    rollback_model_state = _tree_copy(model.parameters())
                    rollback_optimizer_state = _tree_copy(optimizer.state)
                    optimizer.update(model, update_grads)
                    mx.eval(model.parameters(), optimizer.state)
                    if not _tree_all_finite(model.parameters()):
                        rollback_count += 1
                        consecutive_rollbacks += 1
                        lr_scale *= 0.5
                        _restore_rollback_state()
                        _emit({
                            "event": "rollback",
                            "stage": stage,
                            "epoch": epoch_idx + 1,
                            "epochs": epochs,
                            "step": steps,
                            "steps": total_steps,
                            "loss": last_loss,
                            "elapsed_s": time.monotonic() - t0,
                            "dtype": dtype_name,
                            "lr": base_lr * lr_scale,
                            "rollbacks": rollback_count,
                            "reason": "non_finite_parameters",
                        })
                        if consecutive_rollbacks >= rollback_limit:
                            stop_outer = True
                            break
                    else:
                        consecutive_rollbacks = 0
                        pending_grads = None
                        pending_count = 0

            if steps % save_interval == 0:
                _save_checkpoint("interval")

            if steps == 1 or steps % log_interval == 0:
                now = time.monotonic()
                delta_steps = steps - last_log_step
                tps = delta_steps / max(now - last_log_t, 1e-6)
                last_log_t = now
                last_log_step = steps
                _emit({
                    "event": "step",
                    "stage": stage,
                    "epoch": epoch_idx + 1,
                    "epochs": epochs,
                    "step": steps,
                    "steps": total_steps,
                    "loss": last_loss,
                    "ce": last_loss,
                    "aux": 0.0,
                    "temporal": 0.0,
                    "tps": tps,
                    "elapsed_s": now - t0,
                    "dtype": dtype_name,
                    "lr": float(optimizer.learning_rate.item()) if "learning_rate" in optimizer.state else base_lr * lr_scale,
                    "rollbacks": rollback_count,
                })
            if max_steps is not None and steps >= int(max_steps):
                break
        if stop_outer or _should_stop() or (max_steps is not None and steps >= int(max_steps)):
            break

    stopped = _should_stop()
    if pending_grads is not None and pending_count > 0 and not stop_outer:
        current_lr = get_lr(steps, total_steps or steps, base_lr * lr_scale)
        optimizer.learning_rate = current_lr
        update_grads = _scale_grads(pending_grads, 1.0 / max(1, pending_count))
        if grad_clip > 0.0:
            update_grads, _norm = optim.clip_grad_norm(update_grads, grad_clip)
            mx.eval(update_grads)
        if _tree_all_finite(update_grads):
            rollback_model_state = _tree_copy(model.parameters())
            rollback_optimizer_state = _tree_copy(optimizer.state)
            optimizer.update(model, update_grads)
            mx.eval(model.parameters(), optimizer.state)
            if not _tree_all_finite(model.parameters()):
                _restore_rollback_state()
        else:
            _discard_pending()
    _save_checkpoint("stop" if stopped else ("rollback_limit" if stop_outer else "final"))
    elapsed = time.monotonic() - t0
    _emit({
        "event": "stopped" if stopped else ("failed" if stop_outer else "complete"),
        "stage": stage,
        "step": steps,
        "steps": total_steps,
        "loss": last_loss,
        "elapsed_s": elapsed,
        "dtype": dtype_name,
        "checkpoint_path": out_path if checkpoint_saved else "",
        "checkpoint_saved": checkpoint_saved,
        "rollbacks": rollback_count,
    })
    if stopped:
        print(
            f"[MLX {stage}] stopped steps={steps} loss={last_loss:.4f} "
            f"elapsed={elapsed:.2f}s saved={out_path}"
        )
    elif stop_outer:
        print(
            f"[MLX {stage}] stopped after rollback limit steps={steps} loss={last_loss:.4f} "
            f"elapsed={elapsed:.2f}s saved={out_path if checkpoint_saved else 'n/a'}"
        )
    else:
        print(
            f"[MLX {stage}] dtype={dtype_name} steps={steps} loss={last_loss:.4f} "
            f"elapsed={elapsed:.2f}s saved={out_path}"
        )
    return MLXStageResult(
        out_path,
        steps,
        last_loss,
        dtype_name,
        elapsed,
        total_steps=total_steps,
        stopped=stopped or stop_outer,
        checkpoint_saved=checkpoint_saved,
        rollbacks=rollback_count,
    )
