"""
ui/tabs/inference_tab.py — Real-time generation with tokens/s display
"""
import time
import math

from ui.gradio_compat import gr
import pandas as pd

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable
from chronos.trainer.device_utils import configure_cpu_threads, cpu_thread_snapshot


INFERENCE_BACKEND_CHOICES = ("mlx", "cuda", "xpu", "mps", "cpu")
TORCH_INFERENCE_PRIORITY = ("cuda", "xpu", "mps", "cpu")
INFERENCE_MODE_CHOICES = ("compare", "offload", "fullload")
MISS_POLICY_CHOICES = ("on_demand", "sync_on_demand", "fallback_diagnostic")
RAM_LOAD_RATIO_CHOICES = (
    "0.10", "0.20", "0.25", "0.33", "0.50", "0.67",
    "0.75", "0.90", "1.00", "1.10", "1.25",
)
RAM_LOAD_SWEEP_RATIOS = (0.10, 0.20, 0.25, 0.33, 0.50, 0.67, 0.75, 0.90, 1.00, 1.10, 1.25)
INFERENCE_CHART_METRICS = (
    ("response_time_s", "Response time", "s"),
    ("tokens_per_sec", "Decode speed", "tokens/s"),
    ("rss_delta_gb", "RSS delta", "GB"),
    ("setup_rss_delta_gb", "Setup RSS delta", "GB"),
    ("prefill_rss_delta_gb", "Prefill RSS delta", "GB"),
    ("decode_rss_delta_gb", "Decode RSS delta", "GB"),
    ("rss_after_setup_gb", "Setup RSS actual", "GB"),
    ("rss_after_prefill_gb", "Prefill RSS actual", "GB"),
    ("rss_after_decode_gb", "Decode RSS actual", "GB"),
    ("mps_allocated_after_setup_gb", "Setup MPS allocated", "GB"),
    ("mps_allocated_after_prefill_gb", "Prefill MPS allocated", "GB"),
    ("mps_allocated_after_decode_gb", "Decode MPS allocated", "GB"),
    ("mlx_active_after_setup_gb", "Setup MLX active", "GB"),
    ("mlx_active_after_prefill_gb", "Prefill MLX active", "GB"),
    ("mlx_active_after_decode_gb", "Decode MLX active", "GB"),
    ("prefill_time_s", "Prefill time", "s"),
    ("decode_time_s", "Decode time", "s"),
    ("cache_hit_rate", "Cache hit rate", "ratio"),
    ("resident_hit_rate", "Resident hit rate", "ratio"),
    ("prediction_hit_rate", "Prediction hit rate", "ratio"),
    ("fallback_weight_rate", "Fallback weight", "ratio"),
    ("on_demand_loads", "On-demand loads", "count"),
    ("on_demand_load_time_s", "On-demand load time", "s"),
)
INFERENCE_SWEEP_CHART_METRICS = (
    ("tokens_per_sec", "Decode speed", "tokens/s"),
    ("response_time_s", "Response time", "s"),
    ("cache_hit_rate", "Cache hit rate", "ratio"),
    ("prediction_hit_rate", "Prediction hit rate", "ratio"),
    ("fallback_weight_rate", "Fallback weight", "ratio"),
    ("rss_delta_gb", "RSS delta", "GB"),
    ("prefill_rss_delta_gb", "Prefill RSS delta", "GB"),
    ("decode_rss_delta_gb", "Decode RSS delta", "GB"),
)


def _best_torch_inference_backend() -> str:
    from chronos.backend import BackendDispatcher

    d = BackendDispatcher()
    for name in TORCH_INFERENCE_PRIORITY:
        if d.info(name).available:
            return name
    return "cpu"


def _available_inference_backend_choices() -> list[str]:
    from chronos.backend import available

    detected = set(available())
    return ["auto"] + [name for name in INFERENCE_BACKEND_CHOICES if name in detected]


def _default_inference_backend_value() -> str:
    return "auto"


def _resolve_inference_backend(requested_backend: str, model_path_val: str, sniffed: dict) -> tuple[str, str]:
    from chronos.backend import select

    requested = (requested_backend or "auto").strip().lower() or "auto"
    backend = select(None if requested == "auto" else requested)
    note = ""

    if requested not in {"", "auto"} and backend != requested:
        note = (
            f"[backend fallback] Requested {requested}, but it is not available; "
            f"using {backend}.\n"
        )

    return backend, note


def _checkpoint_expert_cache_dir(model_path_val: str, model_cfg) -> str:
    """Keep lazy expert shards isolated per checkpoint.

    Reusing a single ./expert_cache across different .pth files can silently
    load stale expert shards. The cache key uses path + stat metadata + core
    topology so repeated runs reuse the same shards, while changed checkpoints
    get a clean cache namespace.
    """
    import hashlib
    import os

    base = os.path.join(os.getcwd(), "expert_cache")
    if not model_path_val or not os.path.exists(model_path_val):
        return base
    try:
        st = os.stat(model_path_val)
    except OSError:
        return base
    topology = (
        getattr(model_cfg, "vocab_size", ""),
        getattr(model_cfg, "hidden_size", ""),
        getattr(model_cfg, "num_hidden_layers", ""),
        getattr(model_cfg, "num_experts", ""),
        getattr(model_cfg, "moe_intermediate_size", ""),
    )
    raw = f"{os.path.abspath(model_path_val)}:{st.st_size}:{st.st_mtime_ns}:{topology}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return os.path.join(base, f"ckpt_{key}")


def _memory_snapshot_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def _backend_memory_snapshot(backend: str) -> dict:
    try:
        from chronos.backend.mac_diagnostics import mps_memory_snapshot, mlx_memory_snapshot
    except Exception:
        return {}
    if backend == "mps":
        return mps_memory_snapshot()
    if backend == "mlx":
        return mlx_memory_snapshot()
    return {}


def _configure_inference_cpu_threads_if_needed(
    backend: str,
    requested: int | str | None = "auto",
    budget_percent: int | float | str | None = 75,
) -> dict:
    """Keep WebUI CPU inference from inheriting a single-thread shell env."""
    if backend != "cpu":
        return {}
    threads = configure_cpu_threads(requested, budget_percent=budget_percent)
    snap = cpu_thread_snapshot()
    return {
        "cpu_threads": int(threads),
        "cpu_thread_env": snap,
    }


def _memory_delta(after: dict, before: dict, key: str) -> float:
    try:
        return round(float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0), 6)
    except Exception:
        return 0.0


def _actual_active_expert_count(model_cfg) -> int:
    num_experts = int(getattr(model_cfg, "num_experts", 1) or 1)
    for attr in ("observed_active_experts", "offload_working_set_experts", "recommended_working_set_experts"):
        value = getattr(model_cfg, attr, None)
        if value in (None, "", 0):
            continue
        try:
            return max(1, min(num_experts, int(value)))
        except (TypeError, ValueError):
            pass
    top_k = int(getattr(model_cfg, "num_experts_per_tok", 1) or 1)
    num_layers = int(getattr(model_cfg, "num_hidden_layers", 1) or 1)
    # top_k is per layer. The cache is keyed by global expert id, and a
    # single generated token can therefore touch up to top_k * layers distinct
    # expert ids before repeats. Capping at top_k alone makes the lazy path
    # replace a large part of the routed MoE with shared fallback.
    return max(1, min(num_experts, top_k * max(1, num_layers)))


def _bounded_offload_expert_budget(model_cfg, ram_load_ratio: float | str | None = 1.0) -> dict:
    """Derive the offload cache budget from the trained MoE topology.

    The ideal active count is the routed working set, not the checkpoint's
    recommended resident cache size. The user-controlled ratio must produce
    real lower-budget runs for sweeps; exact quality is preserved by
    on-demand materialisation rather than by silently clamping to top-k.
    """
    active = _actual_active_expert_count(model_cfg)
    num_experts = int(getattr(model_cfg, "num_experts", active) or active)
    try:
        ratio = float(ram_load_ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    ratio = max(0.01, ratio)
    routing_top_k = int(getattr(model_cfg, "num_experts_per_tok", active) or active)
    min_vram = 1
    hard_cap = max(active, int(math.floor(active * 1.25)))
    requested = max(1, int(math.ceil(active * ratio)))
    effective = max(min_vram, min(num_experts, requested, hard_cap))
    ram_buffer = effective
    return {
        "ideal_active_experts": active,
        "routing_top_k": routing_top_k,
        "num_moe_layers": int(getattr(model_cfg, "num_hidden_layers", 1) or 1),
        "requested_ram_load_ratio": round(ratio, 3),
        "requested_expert_budget": requested,
        "max_allowed_expert_budget": min(num_experts, hard_cap),
        "effective_expert_budget": effective,
        "effective_vram_expert_budget": effective,
        "effective_ram_expert_budget": ram_buffer,
        "min_vram_expert_budget": min_vram,
        "budget_clamped": effective != requested,
    }


def _clone_model_cfg(model_cfg):
    from chronos.model.config import ChronosConfig

    try:
        data = dict(model_cfg.to_dict())
    except Exception:
        data = dict(getattr(model_cfg, "__dict__", {}))
    data.pop("architectures", None)
    data.pop("transformers_version", None)
    data.pop("_name_or_path", None)
    data.pop("_commit_hash", None)
    return ChronosConfig(**data)


def _apply_offload_expert_budget(engine, model_cfg, ram_load_ratio: float | str | None = 1.0) -> dict:
    budget = _bounded_offload_expert_budget(model_cfg, ram_load_ratio)
    vram_cap = int(budget["effective_vram_expert_budget"])
    ram_cap = int(budget["effective_ram_expert_budget"])
    cache_manager = getattr(engine, "cache_manager", None)
    store = getattr(cache_manager, "expert_store", None)
    if store is None:
        return budget
    store.vram_capacity = vram_cap
    store.ram_capacity = ram_cap
    store.vram_lru.capacity = vram_cap
    store.ram_lru.capacity = ram_cap
    return budget


def _replace_torch_experts_with_placeholders(model, model_cfg) -> None:
    """Remove randomly initialised live experts before model.to(device).

    For .pth inference the real expert tensors are sharded from the checkpoint
    to SSD by ChronosInferenceEngine.setup_from_state_dict(). Keeping the
    constructor's random experts alive until model.to(mps/cuda) defeats the
    SSD+DRAM lazy-load design by moving full dummy experts onto the device.
    """
    import torch
    from chronos.model.moe_chronos import ChronosMOEFeedForward, LazyExpertPlaceholder

    layers = getattr(getattr(model, "model", None), "layers", [])
    hidden = int(getattr(model_cfg, "hidden_size", 0) or 0)
    intermediate = int(getattr(model_cfg, "moe_intermediate_size", 0) or 0)
    if not hidden or not intermediate:
        return
    for layer in layers:
        moe = getattr(layer, "mlp", None)
        if not isinstance(moe, ChronosMOEFeedForward):
            continue
        for idx in range(len(moe.experts)):
            moe.experts[idx] = LazyExpertPlaceholder(hidden, intermediate, torch.float16)


def _replace_mlx_experts_with_placeholders(model) -> None:
    import gc
    import mlx.core as mx
    from chronos.mlx.moe import ChronosMLXMOE, LazyFeedForwardMLX

    for layer in getattr(model, "layers", []):
        moe = getattr(layer, "mlp", None)
        if not isinstance(moe, ChronosMLXMOE):
            continue
        for idx in range(len(moe.experts)):
            moe.experts[idx] = LazyFeedForwardMLX()
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def _populate_mlx_cache_from_state_dict(store, state_dict: dict) -> None:
    import os
    import json
    import torch
    from safetensors.numpy import save_file as np_save_file
    from chronos.io.storage import ClusterManifest, KEY_SEP, MANIFEST_FILENAME

    os.makedirs(store.ssd_dir, exist_ok=True)
    manifest_clusters = {}
    expert_to_cluster = {}
    for eid in range(int(store.num_experts)):
        tensors = {}
        for li in range(int(store.num_layers)):
            prefix = f"model.layers.{li}.mlp.experts.{eid}."
            for pname in store._PARAM_NAMES:
                key = prefix + pname
                if key in state_dict:
                    tensors[f"l{li}_e{eid}{KEY_SEP}{pname}"] = (
                        state_dict[key].detach().to(dtype=torch.float16, device="cpu").numpy()
                    )
        if tensors:
            file_name = f"cluster_{eid}.ctsr"
            np_save_file(tensors, os.path.join(store.ssd_dir, file_name))
            manifest_clusters[eid] = (file_name, [eid])
            expert_to_cluster[eid] = eid
    if len(expert_to_cluster) != int(store.num_experts):
        raise RuntimeError(
            f"MLX lazy cache is incomplete: wrote {len(expert_to_cluster)}/{int(store.num_experts)} experts"
        )
    manifest = ClusterManifest(
        version=1,
        num_experts=int(store.num_experts),
        num_layers=int(store.num_layers),
        storage_format="safetensors",
        clusters=manifest_clusters,
        expert_to_cluster=expert_to_cluster,
    )
    with open(os.path.join(store.ssd_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def _populate_mlx_cache_from_reader(store, reader) -> None:
    import os
    import json
    import torch
    from safetensors.numpy import save_file as np_save_file
    from chronos.io.storage import ClusterManifest, KEY_SEP, MANIFEST_FILENAME

    os.makedirs(store.ssd_dir, exist_ok=True)
    keys = set(reader.keys())
    manifest_clusters = {}
    expert_to_cluster = {}
    for eid in range(int(store.num_experts)):
        tensors = {}
        for li in range(int(store.num_layers)):
            prefix = f"model.layers.{li}.mlp.experts.{eid}."
            for pname in store._PARAM_NAMES:
                key = prefix + pname
                if key in keys:
                    tensors[f"l{li}_e{eid}{KEY_SEP}{pname}"] = (
                        reader.get_tensor(key).detach().to(dtype=torch.float16, device="cpu").numpy()
                    )
        if tensors:
            file_name = f"cluster_{eid}.ctsr"
            np_save_file(tensors, os.path.join(store.ssd_dir, file_name))
            manifest_clusters[eid] = (file_name, [eid])
            expert_to_cluster[eid] = eid
    if len(expert_to_cluster) != int(store.num_experts):
        raise RuntimeError(
            f"MLX lazy cache is incomplete: wrote {len(expert_to_cluster)}/{int(store.num_experts)} experts"
        )
    manifest = ClusterManifest(
        version=1,
        num_experts=int(store.num_experts),
        num_layers=int(store.num_layers),
        storage_format="safetensors",
        clusters=manifest_clusters,
        expert_to_cluster=expert_to_cluster,
    )
    with open(os.path.join(store.ssd_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def _rebuild_mlx_lazy_cache(store, cache_dir: str, reader, weights) -> None:
    import os
    import shutil

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    store.ssd_dir = cache_dir
    store._cluster_manifest = None
    store._loaded_clusters.clear()
    store._warm.clear()
    store._warm_lru.clear()
    if reader is not None:
        _populate_mlx_cache_from_reader(store, reader)
    elif weights is not None:
        _populate_mlx_cache_from_state_dict(store, weights)
    else:
        store.offload_all_to_ssd(clusters=[[i] for i in range(int(store.num_experts))])
    if not store.attach_cluster_manifest(cache_dir):
        raise RuntimeError(f"failed to build a complete MLX lazy expert cache at {cache_dir}")


def _encode_prompt(tokenizer, prompt_val: str, raw_prompt: bool = False) -> tuple[list[int], list[int]]:
    if raw_prompt:
        enc = tokenizer(prompt_val, add_special_tokens=False, return_attention_mask=True)
    else:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_val}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(rendered, add_special_tokens=False, return_attention_mask=True)
    return list(enc["input_ids"]), list(enc.get("attention_mask") or [1] * len(enc["input_ids"]))


def _format_offload_stats(stats: dict) -> str:
    if not stats:
        return ""
    prefill = stats.get("prefill_expert_ids", [])
    return (
        "[offload/predict] "
        f"cache_hit_rate={stats.get('cache_hit_rate', 0):.2%}  "
        f"prefill={prefill}  "
        f"predicted={stats.get('prefetch_submitted_ids', [])}  "
        f"prefetch_requests={stats.get('total_requests', 0)}  "
        f"promoted={stats.get('promoted_expert_ids', [])}  "
        f"vram={stats.get('vram_experts', 0)}/{stats.get('vram_capacity', 0)}  "
        f"ram={stats.get('ram_experts', 0)}/{stats.get('ram_capacity_dynamic', 0)}  "
        f"storage={stats.get('storage_format', 'n/a')}  "
        f"cluster_aware={stats.get('cluster_aware', False)}\n"
    )


def _run_torch_inference(backend: str, model_cfg, model_path_val: str, token_ids: list[int],
                         attention_mask: list[int], max_tok: int, temp: float,
                         eos_token_id: int | None, do_sample: bool = True,
                         ram_load_ratio: float | str | None = 1.0,
                         miss_policy: str = "on_demand",
                         cpu_threads: int | str | None = "auto",
                         cpu_budget_percent: int | float | str | None = 100) -> tuple[list[int], dict]:
    import os
    import importlib
    import gc
    import torch
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
    from chronos.export import is_export_artifact, open_export_reader
    inference_engine_mod = importlib.import_module("chronos.runtime.inference_engine")

    cpu_thread_stats = _configure_inference_cpu_threads_if_needed(
        backend,
        requested=cpu_threads,
        budget_percent=cpu_budget_percent,
    )
    setup_t0 = time.monotonic()
    setup_mem0 = _memory_snapshot_gb()
    backend_mem0 = _backend_memory_snapshot(backend)
    model = ChronosForCausalLM(model_cfg)
    model_init_mem = _memory_snapshot_gb()
    weights = None
    reader = None
    if model_path_val and os.path.exists(model_path_val):
        if is_export_artifact(model_path_val):
            reader = open_export_reader(model_path_val)
            base_only = reader.load_state_dict(include_experts=False)
        else:
            weights = load_checkpoint_state_dict(model_path_val, map_location="cpu")
            base_only = {
                k: v for k, v in weights.items()
                if ".mlp.experts." not in k
            }
        load_state_dict_controlled(
            model,
            base_only,
            allow_missing_substrings=(".mlp.experts.",),
        )
        _replace_torch_experts_with_placeholders(model, model_cfg)
    base_load_mem = _memory_snapshot_gb()

    device_map = {"cuda": "cuda", "xpu": "xpu", "mps": "mps", "cpu": "cpu"}[backend]
    model = model.to(device_map).eval()
    device_mem = _memory_snapshot_gb()
    input_ids = torch.tensor([token_ids]).to(device_map)
    attn = torch.tensor([attention_mask]).to(device_map)
    engine = inference_engine_mod.ChronosInferenceEngine(
        model,
        model_cfg,
        ssd_dir=_checkpoint_expert_cache_dir(model_path_val, model_cfg),
    )
    expert_budget = _apply_offload_expert_budget(engine, model_cfg, ram_load_ratio)
    miss_policy = _normalize_miss_policy(miss_policy)
    if weights is not None:
        engine.setup_from_state_dict(weights, warm_expert_ids=[])
        del weights
        weights = None
        gc.collect()
    elif reader is not None:
        engine.setup_from_state_reader(reader, warm_expert_ids=[])
        del reader
        gc.collect()
    else:
        engine.setup(warm_expert_ids=[])
    setup_mem1 = _memory_snapshot_gb()
    backend_mem_setup = _backend_memory_snapshot(backend)
    setup_stats = {
        "setup_time_s": round(time.monotonic() - setup_t0, 3),
        "setup_rss_delta_gb": round(setup_mem1 - setup_mem0, 3),
        "model_init_rss_delta_gb": round(model_init_mem - setup_mem0, 3),
        "base_load_rss_delta_gb": round(base_load_mem - model_init_mem, 3),
        "device_move_rss_delta_gb": round(device_mem - base_load_mem, 3),
        "expert_setup_rss_delta_gb": round(setup_mem1 - device_mem, 3),
        "rss_before_setup_gb": round(setup_mem0, 3),
        "rss_after_model_init_gb": round(model_init_mem, 3),
        "rss_after_base_load_gb": round(base_load_mem, 3),
        "rss_after_device_move_gb": round(device_mem, 3),
        "rss_after_setup_gb": round(setup_mem1, 3),
        **cpu_thread_stats,
    }
    for snapshot, suffix in [
        (backend_mem0, "before_setup"),
        (backend_mem_setup, "after_setup"),
    ]:
        for key in ("mps_allocated_gb", "mps_driver_allocated_gb", "mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb"):
            if key in snapshot:
                prefix = key[:-3] if key.endswith("_gb") else key
                setup_stats[f"{prefix}_{suffix}_gb"] = snapshot[key]
    for key in ("mps_allocated_gb", "mps_driver_allocated_gb", "mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb"):
        if key in backend_mem_setup:
            prefix = key[:-3] if key.endswith("_gb") else key
            setup_stats[f"{prefix}_setup_delta_gb"] = _memory_delta(backend_mem_setup, backend_mem0, key)

    try:
        out = engine.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=int(max_tok),
            temperature=float(temp),
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            miss_policy=miss_policy,
        )
        stats = dict(getattr(engine, "last_stats", {}))
    finally:
        engine.teardown()

    prompt_len = input_ids.shape[1]
    generated = out[0, prompt_len:].tolist()
    stats.update(setup_stats)
    stats.update(expert_budget)

    return generated, stats


def _next_token_from_logits(logits, input_ids, temperature: float, top_p: float = 0.85,
                            top_k: int = 50, do_sample: bool = True,
                            repetition_penalty: float = 1.0):
    import torch
    import torch.nn.functional as F

    temperature = max(float(temperature), 1e-6)
    logits = logits / temperature
    if repetition_penalty != 1.0:
        for i in range(input_ids.shape[0]):
            logits[i, torch.unique(input_ids[i])] /= repetition_penalty
    if top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        logits[logits < torch.topk(logits, k)[0][..., -1, None]] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        mask = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
    return (
        torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
    )


def _run_torch_full_dram(backend: str, model_cfg, model_path_val: str, token_ids: list[int],
                         attention_mask: list[int], max_tok: int, temp: float,
                         eos_token_id: int | None, do_sample: bool = True,
                         cpu_threads: int | str | None = "auto",
                         cpu_budget_percent: int | float | str | None = 100) -> tuple[list[int], dict]:
    import os
    import time as _time
    import gc
    import torch
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
    from chronos.export import is_export_artifact, open_export_reader

    device_map = {"cuda": "cuda", "xpu": "xpu", "mps": "mps", "cpu": "cpu"}[backend]
    cpu_thread_stats = _configure_inference_cpu_threads_if_needed(
        backend,
        requested=cpu_threads,
        budget_percent=cpu_budget_percent,
    )
    setup_t0 = _time.monotonic()
    setup_mem0 = _memory_snapshot_gb()
    backend_mem0 = _backend_memory_snapshot(backend)
    model = ChronosForCausalLM(model_cfg)
    model_init_mem = _memory_snapshot_gb()
    if model_path_val and os.path.exists(model_path_val):
        if is_export_artifact(model_path_val):
            weights = open_export_reader(model_path_val).load_state_dict(include_experts=True)
        else:
            weights = load_checkpoint_state_dict(model_path_val, map_location="cpu")
        load_state_dict_controlled(model, weights)
        del weights
        gc.collect()
    weight_load_mem = _memory_snapshot_gb()
    model = model.to(device_map).eval()
    setup_mem1 = _memory_snapshot_gb()
    backend_mem_setup = _backend_memory_snapshot(backend)
    input_ids = torch.tensor([token_ids], device=device_map)
    attn = torch.tensor([attention_mask], device=device_map)
    prompt_len = input_ids.shape[1]
    past_key_values = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    tokens_generated = 0
    prefill_t0 = _time.monotonic()
    prefill_mem0 = _memory_snapshot_gb()
    backend_mem_prefill0 = _backend_memory_snapshot(backend)
    prefill_time_s = 0.0
    prefill_mem1 = prefill_mem0
    backend_mem_prefill1 = backend_mem_prefill0
    prefill_done_t = prefill_t0

    with torch.inference_mode():
        for step in range(int(max_tok)):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs, _ = model(
                input_ids[:, past_len:],
                attention_mask=attn,
                past_key_values=past_key_values,
                use_cache=True,
            )
            if attn is not None:
                attn = torch.cat([attn, attn.new_ones(attn.shape[0], 1)], -1)
            next_token = _next_token_from_logits(
                outputs.logits[:, -1, :],
                input_ids,
                temperature=float(temp),
                do_sample=do_sample,
            )
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values
            tokens_generated += 1

            if step == 0:
                prefill_done_t = _time.monotonic()
                prefill_mem1 = _memory_snapshot_gb()
                backend_mem_prefill1 = _backend_memory_snapshot(backend)
                prefill_time_s = prefill_done_t - prefill_t0

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

    end_t = _time.monotonic()
    end_mem = _memory_snapshot_gb()
    backend_mem_end = _backend_memory_snapshot(backend)
    if tokens_generated == 0:
        prefill_done_t = end_t
        prefill_mem1 = end_mem
        backend_mem_prefill1 = backend_mem_end
        prefill_time_s = end_t - prefill_t0
    elapsed = end_t - prefill_t0
    generated = input_ids[0, prompt_len:].tolist()
    stats = {
        "tokens_generated": len(generated),
        "elapsed_s": round(elapsed, 3),
        "tokens_per_sec": round(len(generated) / max(elapsed, 1e-6), 2),
        "setup_time_s": round(_time.monotonic() - setup_t0 - elapsed, 3),
        "setup_rss_delta_gb": round(setup_mem1 - setup_mem0, 3),
        "model_init_rss_delta_gb": round(model_init_mem - setup_mem0, 3),
        "base_load_rss_delta_gb": round(weight_load_mem - model_init_mem, 3),
        "device_move_rss_delta_gb": round(setup_mem1 - weight_load_mem, 3),
        "expert_setup_rss_delta_gb": 0.0,
        "rss_before_setup_gb": round(setup_mem0, 3),
        "rss_after_model_init_gb": round(model_init_mem, 3),
        "rss_after_base_load_gb": round(weight_load_mem, 3),
        "rss_after_device_move_gb": round(setup_mem1, 3),
        "rss_after_setup_gb": round(setup_mem1, 3),
        "prefill_time_s": round(prefill_time_s, 3),
        "decode_time_s": round(max(0.0, end_t - prefill_done_t), 3),
        "prefill_rss_delta_gb": round(prefill_mem1 - prefill_mem0, 3),
        "decode_rss_delta_gb": round(end_mem - prefill_mem1, 3),
        "rss_before_prefill_gb": round(prefill_mem0, 3),
        "rss_after_prefill_gb": round(prefill_mem1, 3),
        "rss_after_decode_gb": round(end_mem, 3),
        "storage_format": "full_dram",
        "cache_hit_rate": 1.0,
        "cache_hits": 0,
        "cache_misses": 0,
        **cpu_thread_stats,
    }
    for snapshot, suffix in [
        (backend_mem0, "before_setup"),
        (backend_mem_setup, "after_setup"),
        (backend_mem_prefill0, "before_prefill"),
        (backend_mem_prefill1, "after_prefill"),
        (backend_mem_end, "after_decode"),
    ]:
        for key in ("mps_allocated_gb", "mps_driver_allocated_gb", "mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb"):
            if key in snapshot:
                prefix = key[:-3] if key.endswith("_gb") else key
                stats[f"{prefix}_{suffix}_gb"] = snapshot[key]
    return generated, stats


def _run_mlx_full_dram(model_cfg, model_path_val: str, token_ids: list[int],
                       max_tok: int, temp: float, do_sample: bool = True) -> tuple[list[int], dict]:
    import os
    import time as _time
    import gc
    import torch
    import mlx.core as mx
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
    from chronos.export import is_export_artifact, open_export_reader
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.inference import ChronosMLXInferenceEngine

    setup_t0 = _time.monotonic()
    setup_mem0 = _memory_snapshot_gb()
    backend_mem0 = _backend_memory_snapshot("mlx")
    if model_path_val and is_export_artifact(model_path_val):
        weights = open_export_reader(model_path_val).load_state_dict(include_experts=True)
    elif model_path_val and os.path.exists(model_path_val):
        weights = load_checkpoint_state_dict(model_path_val, map_location="cpu")
    else:
        weights = {}

    pt_model = ChronosForCausalLM(model_cfg)
    model_init_mem = _memory_snapshot_gb()
    if weights:
        load_state_dict_controlled(pt_model, weights)
        del weights
        weights = {}
        gc.collect()
    weight_load_mem = _memory_snapshot_gb()
    mlx_model = ChronosMLXModel.from_chronos_pytorch(pt_model, model_cfg)
    del pt_model
    gc.collect()
    setup_mem1 = _memory_snapshot_gb()
    backend_mem_setup = _backend_memory_snapshot("mlx")
    engine = ChronosMLXInferenceEngine(mlx_model, model_cfg)
    engine.store.storage_format = "full_dram"
    try:
        ids = mx.array([token_ids])
        run_t0 = _time.monotonic()
        tokens = list(engine.generate(
            ids,
            max_new_tokens=int(max_tok),
            temperature=float(temp) if do_sample else 0.0,
            top_p=0.85,
        ))
        run_elapsed = _time.monotonic() - run_t0
        end_mem = _memory_snapshot_gb()
        backend_mem_end = _backend_memory_snapshot("mlx")
        stats = dict(getattr(engine, "last_stats", {}))
        prefill_mem = float(stats.get("rss_after_prefill_gb", setup_mem1) or setup_mem1)
        prefill_time_s = float(stats.get("prefill_time_s", 0.0) or 0.0)
        stats.update({
            "setup_time_s": round(_time.monotonic() - setup_t0 - run_elapsed, 3),
            "setup_rss_delta_gb": round(setup_mem1 - setup_mem0, 3),
            "model_init_rss_delta_gb": round(model_init_mem - setup_mem0, 3),
            "base_load_rss_delta_gb": round(weight_load_mem - model_init_mem, 3),
            "device_move_rss_delta_gb": round(setup_mem1 - weight_load_mem, 3),
            "expert_setup_rss_delta_gb": 0.0,
            "rss_before_setup_gb": round(setup_mem0, 3),
            "rss_after_model_init_gb": round(model_init_mem, 3),
            "rss_after_base_load_gb": round(weight_load_mem, 3),
            "rss_after_device_move_gb": round(setup_mem1, 3),
            "rss_after_setup_gb": round(setup_mem1, 3),
            "prefill_rss_delta_gb": round(prefill_mem - setup_mem1, 3),
            "decode_rss_delta_gb": round(end_mem - prefill_mem, 3),
            "rss_before_prefill_gb": round(setup_mem1, 3),
            "rss_after_prefill_gb": round(prefill_mem, 3),
            "rss_after_decode_gb": round(end_mem, 3),
            "prefill_time_s": round(prefill_time_s, 3),
            "decode_time_s": round(max(0.0, run_elapsed - prefill_time_s), 3),
            "storage_format": "mlx_full_dram",
            "cache_hit_rate": 1.0,
            "resident_hit_rate": 1.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "miss_policy": "mlx_full_dram",
        })
        for snapshot, suffix in [
            (backend_mem0, "before_setup"),
            (backend_mem_setup, "after_setup"),
            (backend_mem_end, "after_decode"),
        ]:
            for key in ("mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb"):
                if key in snapshot:
                    prefix = key[:-3] if key.endswith("_gb") else key
                    stats[f"{prefix}_{suffix}_gb"] = snapshot[key]
        return tokens, stats
    finally:
        engine.stop()


def _run_mlx_lazy_inference(model_cfg, model_path_val: str, token_ids: list[int],
                            max_tok: int, temp: float, do_sample: bool = True,
                            ram_load_ratio: float | str | None = 1.0,
                            quality_safe: bool = True) -> tuple[list[int], dict]:
    import os
    import gc
    import torch
    import mlx.core as mx
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled
    from chronos.export import is_export_artifact, open_export_reader
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.inference import ChronosMLXInferenceEngine

    budget = _bounded_offload_expert_budget(model_cfg, ram_load_ratio)
    model_cfg.recommended_resident_experts = int(budget["effective_vram_expert_budget"])
    model_cfg.recommended_ram_experts = int(budget["effective_ram_expert_budget"])
    cache_dir = _checkpoint_expert_cache_dir(model_path_val, model_cfg) + "_mlx"

    setup_mem0 = _memory_snapshot_gb()
    backend_mem0 = _backend_memory_snapshot("mlx")
    reader = None
    weights = None
    if model_path_val and is_export_artifact(model_path_val):
        reader = open_export_reader(model_path_val)
        base_only = reader.load_state_dict(include_experts=False)
    elif model_path_val and os.path.exists(model_path_val):
        weights = load_checkpoint_state_dict(model_path_val, map_location="cpu")
        base_only = {k: v for k, v in weights.items() if ".mlp.experts." not in k}
    else:
        raise FileNotFoundError("MLX lazy/offload inference requires a checkpoint or export artifact.")

    pt_model = ChronosForCausalLM(model_cfg)
    if base_only:
        load_state_dict_controlled(
            pt_model,
            base_only,
            allow_missing_substrings=(".mlp.experts.",),
        )
    _replace_torch_experts_with_placeholders(pt_model, model_cfg)
    model_init_mem = _memory_snapshot_gb()
    mlx_model = ChronosMLXModel.from_chronos_pytorch(pt_model, model_cfg, include_experts=False)
    _replace_mlx_experts_with_placeholders(mlx_model)
    del pt_model
    gc.collect()

    engine = ChronosMLXInferenceEngine(mlx_model, model_cfg, ssd_dir=cache_dir)
    try:
        # MLX lazy mode must keep the same expert budget as the offload path:
        # warming all experts hides the architecture and can leave stale LRU
        # state when the budget evicts placeholders. Warm only the execution
        # budget; selected misses are synchronously materialized expert-by-expert.
        warm_count = int(budget["effective_vram_expert_budget"])
        warm_ids = list(range(min(int(model_cfg.num_experts), warm_count)))
        if not engine.store.attach_cluster_manifest(cache_dir):
            _rebuild_mlx_lazy_cache(engine.store, cache_dir, reader, weights)
        if reader is not None:
            del reader
            reader = None
        if weights is not None:
            del weights
            weights = None
            gc.collect()
        engine.store.warm_up(warm_ids)
        setup_mem1 = _memory_snapshot_gb()
        backend_mem_setup = _backend_memory_snapshot("mlx")
        ids = mx.array([token_ids])
        tokens = list(engine.generate(
            ids,
            max_new_tokens=int(max_tok),
            temperature=float(temp) if do_sample else 0.0,
            top_p=0.85,
        ))
        stats = dict(getattr(engine, "last_stats", {}))
        store_stats = engine.store.stats()
        stats.update({
            "setup_rss_delta_gb": round(setup_mem1 - setup_mem0, 3),
            "model_init_rss_delta_gb": round(model_init_mem - setup_mem0, 3),
            "expert_setup_rss_delta_gb": round(setup_mem1 - model_init_mem, 3),
            "rss_before_setup_gb": round(setup_mem0, 3),
            "rss_after_model_init_gb": round(model_init_mem, 3),
            "rss_after_setup_gb": round(setup_mem1, 3),
            "rss_after_prefill_gb": stats.get("rss_after_prefill_gb", round(setup_mem1, 3)),
            "rss_after_decode_gb": stats.get("rss_after_decode_gb", round(setup_mem1, 3)),
            **budget,
            "storage_format": store_stats.get("storage_format", "safetensors"),
            "cluster_aware": store_stats.get("cluster_aware", False),
            "num_clusters": store_stats.get("num_clusters", 0),
            "vram_experts": store_stats.get("hot_experts", 0),
            "vram_capacity": store_stats.get("hot_capacity", 0),
            "ram_experts": store_stats.get("warm_experts", 0),
            "ram_capacity_dynamic": store_stats.get("warm_capacity", int(model_cfg.num_experts)),
            "miss_policy": "mlx_lazy_quality",
            "quality_safe": quality_safe,
        })
        for snapshot, suffix in [
            (backend_mem0, "before_setup"),
            (backend_mem_setup, "after_setup"),
        ]:
            for key in ("mlx_active_gb", "mlx_cache_gb", "mlx_peak_gb"):
                if key in snapshot:
                    prefix = key[:-3] if key.endswith("_gb") else key
                    stats[f"{prefix}_{suffix}_gb"] = snapshot[key]
        return tokens, stats
    finally:
        if reader is not None:
            del reader
        if weights is not None:
            del weights
        engine.stop()


def _text_diff_summary(a: str, b: str) -> dict:
    import difflib

    ratio = difflib.SequenceMatcher(a=a, b=b).ratio() if (a or b) else 1.0
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix += 1
    return {
        "same": a == b,
        "similarity": round(ratio, 4),
        "common_prefix_chars": prefix,
        "len_lazy": len(a),
        "len_full_dram": len(b),
    }


def _empty_inference_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["metric", "mode", "x", "value", "normalized_value", "unit"])


def _rows_to_chart_df(rows: list[dict]) -> pd.DataFrame:
    data = _rows_to_chart_records(rows)
    return pd.DataFrame(data) if data else _empty_inference_df()


def _rows_to_chart_records(rows: list[dict]) -> list[dict]:
    has_sweep = any(row.get("ram_load_ratio") is not None for row in rows)
    metrics = INFERENCE_SWEEP_CHART_METRICS if has_sweep else INFERENCE_CHART_METRICS
    metric_max: dict[str, float] = {}
    for _key, label, _unit in metrics:
        vals = []
        for row in rows:
            value = row.get(_key)
            if value is None:
                continue
            try:
                vals.append(abs(float(value)))
            except (TypeError, ValueError):
                continue
        metric_max[label] = max(vals) if vals else 0.0

    data = []
    for row in rows:
        mode = str(row.get("mode", ""))
        ratio = row.get("ram_load_ratio")
        x_value = f"{float(ratio):.2g}x" if ratio is not None else mode
        for key, label, unit in metrics:
            value = row.get(key)
            if value is None:
                continue
            value = float(value)
            denom = metric_max.get(label, 0.0)
            data.append({
                "metric": label,
                "mode": mode,
                "x": x_value,
                "value": value,
                "normalized_value": value / denom if denom > 0 else 0.0,
                "unit": unit,
            })
    return data


def _normalize_inference_mode(selected_mode: str | None) -> str:
    mode = (selected_mode or "compare").strip().lower()
    return mode if mode in INFERENCE_MODE_CHOICES else "compare"


def _normalize_ram_load_ratio(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _normalize_miss_policy(value) -> str:
    policy = (value or "on_demand").strip().lower().replace("-", "_")
    aliases = {
        "quality_safe": "sync_on_demand",
        "quality": "sync_on_demand",
        "safe": "sync_on_demand",
        "strict": "sync_on_demand",
        "sync": "sync_on_demand",
        "blocking": "sync_on_demand",
        "fallback": "fallback_diagnostic",
        "diagnostic": "fallback_diagnostic",
    }
    policy = aliases.get(policy, policy)
    return policy if policy in MISS_POLICY_CHOICES else "on_demand"


def _format_inference_stats(rows: list[dict], diff: dict | None = None) -> str:
    if not rows:
        return "_No inference stats yet._"
    mode_labels = [str(row.get("mode", f"run_{idx + 1}")) for idx, row in enumerate(rows)]

    def fmt_num(row: dict, key: str, suffix: str = "", precision: int = 3) -> str:
        try:
            return f"{float(row.get(key, 0.0) or 0.0):.{precision}f}{suffix}"
        except (TypeError, ValueError):
            return ""

    def fmt_pct(row: dict, key: str, fallback_key: str | None = None) -> str:
        value = row.get(key, row.get(fallback_key, 0.0) if fallback_key else 0.0)
        try:
            return f"{float(value or 0.0):.2%}"
        except (TypeError, ValueError):
            return ""

    def fmt_int(row: dict, key: str) -> str:
        try:
            return str(int(row.get(key, 0) or 0))
        except (TypeError, ValueError):
            return ""

    def fmt_hits(row: dict) -> str:
        return f"{fmt_int(row, 'cache_hits')}/{fmt_int(row, 'cache_misses')}"

    metric_rows = [
        ("Backend", lambda r: str(r.get("backend", ""))),
        ("Policy", lambda r: str(r.get("miss_policy", "full"))),
        ("CPU threads", lambda r: fmt_int(r, "cpu_threads")),
        ("Tokens", lambda r: fmt_int(r, "tokens")),
        ("Tokens/s", lambda r: fmt_num(r, "tokens_per_sec", precision=2)),
        ("Response time", lambda r: fmt_num(r, "response_time_s", "s")),
        ("Prefill time", lambda r: fmt_num(r, "prefill_time_s", "s")),
        ("Decode time", lambda r: fmt_num(r, "decode_time_s", "s")),
        ("RSS delta", lambda r: fmt_num(r, "rss_delta_gb", " GB")),
        ("Setup RSS delta", lambda r: fmt_num(r, "setup_rss_delta_gb", " GB")),
        ("Prefill RSS delta", lambda r: fmt_num(r, "prefill_rss_delta_gb", " GB")),
        ("Decode RSS delta", lambda r: fmt_num(r, "decode_rss_delta_gb", " GB")),
        ("Setup RSS actual", lambda r: fmt_num(r, "rss_after_setup_gb", " GB")),
        ("Prefill RSS actual", lambda r: fmt_num(r, "rss_after_prefill_gb", " GB")),
        ("Decode RSS actual", lambda r: fmt_num(r, "rss_after_decode_gb", " GB")),
        ("MPS setup", lambda r: fmt_num(r, "mps_allocated_after_setup_gb", " GB")),
        ("MPS prefill", lambda r: fmt_num(r, "mps_allocated_after_prefill_gb", " GB")),
        ("MPS decode", lambda r: fmt_num(r, "mps_allocated_after_decode_gb", " GB")),
        ("MLX setup", lambda r: fmt_num(r, "mlx_active_after_setup_gb", " GB")),
        ("MLX prefill", lambda r: fmt_num(r, "mlx_active_after_prefill_gb", " GB")),
        ("MLX decode", lambda r: fmt_num(r, "mlx_active_after_decode_gb", " GB")),
        ("Resident hit", lambda r: fmt_pct(r, "resident_hit_rate", "cache_hit_rate")),
        ("Predict hit", lambda r: fmt_pct(r, "prediction_hit_rate")),
        ("Fallback weight", lambda r: fmt_pct(r, "fallback_weight_rate")),
        ("On-demand loads", lambda r: fmt_int(r, "on_demand_loads")),
        ("Prefetch drops", lambda r: fmt_int(r, "prefetch_queue_drops")),
        ("Prefetch wait", lambda r: fmt_num(r, "prefetch_wait_time_s", "s")),
        ("Warm hits", lambda r: fmt_int(r, "warm_hits")),
        ("Hot evictions", lambda r: fmt_int(r, "hot_evictions")),
        ("Warm evictions", lambda r: fmt_int(r, "warm_evictions")),
        ("Async misses", lambda r: fmt_int(r, "async_cold_miss_prefetches")),
        ("Sync SSD loads", lambda r: fmt_int(r, "sync_ssd_loads")),
        ("Expert hits/misses", fmt_hits),
        ("Load budget", lambda r: str(r.get("load_budget", "all"))),
        ("VRAM experts", lambda r: str(r.get("vram_experts", ""))),
        ("RAM experts", lambda r: str(r.get("ram_experts", ""))),
        ("Cluster aware", lambda r: str(r.get("cluster_aware", False))),
    ]

    lines = [
        "| Metric | " + " | ".join(mode_labels) + " |",
        "|---|" + "|".join("---:" for _ in mode_labels) + "|",
    ]
    for metric, getter in metric_rows:
        values = [getter(row) for row in rows]
        if all(value in {"", "0.000 GB", "0.000s", "0.00%", "0"} for value in values):
            continue
        lines.append("| " + metric + " | " + " | ".join(values) + " |")
    if diff:
        lines.append("")
        lines.append(
            f"Output diff: same=`{diff.get('same')}`, similarity=`{diff.get('similarity')}`, "
            f"common_prefix_chars=`{diff.get('common_prefix_chars')}`, "
            f"len_lazy=`{diff.get('len_lazy')}`, len_full_dram=`{diff.get('len_full_dram')}`"
        )
    return "\n".join(lines)


def _row_from_stats(mode: str, backend: str, tokens: list[int], elapsed: float,
                    rss_delta: float, stats: dict) -> dict:
    effective_budget = stats.get("effective_expert_budget")
    ideal_budget = stats.get("ideal_active_experts")
    max_budget = stats.get("max_allowed_expert_budget")
    load_budget = (
        "all" if mode == "full_dram" else
        f"{effective_budget}/{max_budget} (ideal {ideal_budget})"
    )
    return {
        "mode": mode,
        "backend": backend,
        "cpu_threads": stats.get("cpu_threads", 0),
        "cpu_thread_env": stats.get("cpu_thread_env", {}),
        "tokens": len(tokens),
        "tokens_per_sec": round(len(tokens) / max(elapsed, 1e-6), 2),
        "response_time_s": round(elapsed, 3),
        "rss_delta_gb": round(rss_delta, 3),
        "setup_time_s": stats.get("setup_time_s", 0.0),
        "setup_rss_delta_gb": stats.get("setup_rss_delta_gb", 0.0),
        "model_init_rss_delta_gb": stats.get("model_init_rss_delta_gb", 0.0),
        "base_load_rss_delta_gb": stats.get("base_load_rss_delta_gb", 0.0),
        "device_move_rss_delta_gb": stats.get("device_move_rss_delta_gb", 0.0),
        "expert_setup_rss_delta_gb": stats.get("expert_setup_rss_delta_gb", 0.0),
        "rss_before_setup_gb": stats.get("rss_before_setup_gb", 0.0),
        "rss_after_model_init_gb": stats.get("rss_after_model_init_gb", 0.0),
        "rss_after_base_load_gb": stats.get("rss_after_base_load_gb", 0.0),
        "rss_after_device_move_gb": stats.get("rss_after_device_move_gb", 0.0),
        "rss_after_setup_gb": stats.get("rss_after_setup_gb", 0.0),
        "prefill_time_s": stats.get("prefill_time_s", 0.0),
        "decode_time_s": stats.get("decode_time_s", 0.0),
        "prefill_rss_delta_gb": stats.get("prefill_rss_delta_gb", 0.0),
        "decode_rss_delta_gb": stats.get("decode_rss_delta_gb", 0.0),
        "rss_before_prefill_gb": stats.get("rss_before_prefill_gb", 0.0),
        "rss_after_prefill_gb": stats.get("rss_after_prefill_gb", 0.0),
        "rss_after_decode_gb": stats.get("rss_after_decode_gb", 0.0),
        "mps_allocated_before_setup_gb": stats.get("mps_allocated_before_setup_gb", 0.0),
        "mps_allocated_after_setup_gb": stats.get("mps_allocated_after_setup_gb", 0.0),
        "mps_allocated_before_prefill_gb": stats.get("mps_allocated_before_prefill_gb", 0.0),
        "mps_allocated_after_prefill_gb": stats.get("mps_allocated_after_prefill_gb", 0.0),
        "mps_allocated_after_decode_gb": stats.get("mps_allocated_after_decode_gb", 0.0),
        "mps_driver_allocated_after_setup_gb": stats.get("mps_driver_allocated_after_setup_gb", 0.0),
        "mps_driver_allocated_after_prefill_gb": stats.get("mps_driver_allocated_after_prefill_gb", 0.0),
        "mps_driver_allocated_after_decode_gb": stats.get("mps_driver_allocated_after_decode_gb", 0.0),
        "mlx_active_before_setup_gb": stats.get("mlx_active_before_setup_gb", 0.0),
        "mlx_active_after_setup_gb": stats.get("mlx_active_after_setup_gb", 0.0),
        "mlx_active_before_prefill_gb": stats.get("mlx_active_before_prefill_gb", 0.0),
        "mlx_active_after_prefill_gb": stats.get("mlx_active_after_prefill_gb", 0.0),
        "mlx_active_after_decode_gb": stats.get("mlx_active_after_decode_gb", 0.0),
        "mlx_cache_after_setup_gb": stats.get("mlx_cache_after_setup_gb", 0.0),
        "mlx_cache_after_prefill_gb": stats.get("mlx_cache_after_prefill_gb", 0.0),
        "mlx_cache_after_decode_gb": stats.get("mlx_cache_after_decode_gb", 0.0),
        "cache_hit_rate": stats.get("cache_hit_rate", 1.0 if mode == "full_dram" else 0),
        "resident_hit_rate": stats.get("resident_hit_rate", stats.get("cache_hit_rate", 1.0 if mode == "full_dram" else 0)),
        "prediction_hit_rate": stats.get("prediction_hit_rate", 0.0),
        "on_demand_loads": stats.get("on_demand_loads", 0),
        "on_demand_load_time_s": stats.get("on_demand_load_time_s", 0.0),
        "async_cold_miss_prefetches": stats.get("async_cold_miss_prefetches", 0),
        "prefetch_queue_drops": stats.get("prefetch_queue_drops", 0),
        "prefetch_wait_time_s": stats.get("prefetch_wait_time_s", 0.0),
        "sync_ssd_loads": stats.get("sync_ssd_loads", 0),
        "resident_vram_hits": stats.get("resident_vram_hits", 0),
        "resident_ram_hits": stats.get("resident_ram_hits", 0),
        "warm_hits": stats.get("warm_hits", 0),
        "warm_evictions": stats.get("warm_evictions", 0),
        "hot_evictions": stats.get("hot_evictions", 0),
        "quality_safe": stats.get("quality_safe", mode != "full_dram"),
        "miss_policy": stats.get("miss_policy", "full_dram" if mode == "full_dram" else "on_demand"),
        "cache_hits": stats.get("cache_hits", 0),
        "cache_misses": stats.get("cache_misses", 0),
        "fallback_weight_rate": stats.get("fallback_weight_rate", 0.0),
        "fallback_weight_mass": stats.get("fallback_weight_mass", 0.0),
        "hit_weight_mass": stats.get("hit_weight_mass", 0.0),
        "total_route_weight_mass": stats.get("total_route_weight_mass", 0.0),
        "route_layer_stats": stats.get("route_layer_stats", []),
        "load_budget": load_budget,
        "ideal_active_experts": stats.get("ideal_active_experts"),
        "requested_ram_load_ratio": stats.get("requested_ram_load_ratio"),
        "requested_expert_budget": stats.get("requested_expert_budget"),
        "max_allowed_expert_budget": stats.get("max_allowed_expert_budget"),
        "effective_expert_budget": stats.get("effective_expert_budget"),
        "effective_vram_expert_budget": stats.get("effective_vram_expert_budget"),
        "effective_ram_expert_budget": stats.get("effective_ram_expert_budget"),
        "min_vram_expert_budget": stats.get("min_vram_expert_budget"),
        "budget_clamped": stats.get("budget_clamped"),
        "prefill_experts": stats.get("prefill_expert_ids", []),
        "prefill_ram_experts": stats.get("prefill_ram_expert_ids", []),
        "activated_experts": stats.get("activated_expert_ids", []),
        "predicted_current_experts": stats.get("predicted_current_expert_ids", []),
        "predicted_experts": stats.get("prefetch_submitted_ids", []),
        "promoted_experts": stats.get("promoted_expert_ids", []),
        "on_demand_experts": stats.get("on_demand_expert_ids", []),
        "fallback_experts": stats.get("fallback_expert_ids", []),
        "vram_experts": (
            "all" if mode == "full_dram"
            else f"{stats.get('vram_experts', 0)}/{stats.get('vram_capacity', 0)}"
        ),
        "ram_experts": (
            "all" if mode == "full_dram"
            else f"{stats.get('ram_experts', 0)}/{stats.get('ram_capacity_dynamic', 0)}"
        ),
        "cluster_aware": False if mode == "full_dram" else stats.get("cluster_aware", False),
    }


def _build_model_cfg(cfg, model_path_val):
    from chronos.model.config import ChronosConfig
    from chronos.model.checkpoint import config_dict_for_checkpoint, sniff_checkpoint_config
    from chronos.export import config_dict_from_export, is_export_artifact

    cfg = cfg or {}
    cfg_sources = []
    try:
        if model_path_val and is_export_artifact(model_path_val):
            ckpt_cfg = config_dict_from_export(model_path_val)
            cfg_sources = [model_path_val]
        else:
            ckpt_cfg, cfg_sources = config_dict_for_checkpoint(
                model_path_val if model_path_val else None,
                require_unsniffable=False,
            )
    except Exception:
        ckpt_cfg, cfg_sources = {}, []
    sniffed = ckpt_cfg or (sniff_checkpoint_config(model_path_val) if model_path_val else {})
    sniff_note = ""
    if sniffed:
        summary = ", ".join(
            f"{k}={v}" for k, v in sniffed.items()
            if k in ("vocab_size", "hidden_size", "num_hidden_layers",
                     "num_experts", "moe_intermediate_size")
        )
        source_text = "; ".join(cfg_sources) if cfg_sources else model_path_val
        sniff_note = f"[checkpoint config: {summary} | source={source_text}]\n"

    model_cfg_kwargs = {
        "hidden_size":          sniffed.get("hidden_size",        cfg.get("hidden_size", 768)),
        "num_hidden_layers":    sniffed.get("num_hidden_layers",  cfg.get("num_hidden_layers", 8)),
        "num_experts":          sniffed.get("num_experts",        cfg.get("num_experts", 4)),
        "num_experts_per_tok":  sniffed.get("num_experts_per_tok", cfg.get("num_experts_per_tok", 1)),
        "num_shared_experts":   sniffed.get("num_shared_experts", cfg.get("num_shared_experts", 1)),
        "lookahead_steps":      sniffed.get("lookahead_steps", cfg.get("lookahead_steps", 2)),
        "kv_latent_dim":        sniffed.get("kv_latent_dim", cfg.get("kv_latent_dim", 64)),
        "sliding_window_size":  sniffed.get("sliding_window_size", cfg.get("sliding_window_size", 2048)),
        "use_hybrid_attention": sniffed.get("use_hybrid_attention", cfg.get("use_hybrid_attention", True)),
        "vram_budget_gb":       sniffed.get("vram_budget_gb", cfg.get("vram_budget_gb", 4.0)),
        "pinned_memory_max_fraction": sniffed.get("pinned_memory_max_fraction", cfg.get("pinned_memory_max_fraction", 0.25)),
        "storage_format":       sniffed.get("storage_format", cfg.get("storage_format", "safetensors")),
        "offload_miss_policy_default": sniffed.get("offload_miss_policy_default", cfg.get("offload_miss_policy_default", "on_demand")),
        "recommended_resident_experts": sniffed.get("recommended_resident_experts", cfg.get("recommended_resident_experts", None)),
        "use_moe": True,
    }
    if "cluster_manifest_path" in sniffed:
        model_cfg_kwargs["cluster_manifest_path"] = sniffed["cluster_manifest_path"]
    if "rope_dim" in sniffed:
        model_cfg_kwargs["rope_dim"] = sniffed["rope_dim"]
    if "num_attention_heads" in sniffed:
        model_cfg_kwargs["num_attention_heads"] = sniffed["num_attention_heads"]
    if "num_key_value_heads" in sniffed:
        model_cfg_kwargs["num_key_value_heads"] = sniffed["num_key_value_heads"]
    if "vocab_size" in sniffed:
        model_cfg_kwargs["vocab_size"] = sniffed["vocab_size"]
    if "moe_intermediate_size" in sniffed:
        model_cfg_kwargs["intermediate_size"] = sniffed["moe_intermediate_size"]
        model_cfg_kwargs["moe_intermediate_size"] = sniffed["moe_intermediate_size"]

    for opt_key, cfg_key in [
        ("num_attention_heads",   "num_attention_heads"),
        ("num_key_value_heads",   "num_key_value_heads"),
        ("rope_dim",              "rope_dim"),
        ("max_position_embeddings", "max_seq_len"),
        ("tie_word_embeddings",   "tie_word_embeddings"),
    ]:
        if opt_key in {"num_attention_heads", "num_key_value_heads", "rope_dim"} \
                and opt_key in model_cfg_kwargs:
            continue
        if cfg_key not in cfg:
            continue
        val = cfg[cfg_key]
        if val in (None, "", 0):
            continue
        model_cfg_kwargs[opt_key] = val

    if "num_attention_heads_total_dim" in sniffed:
        h = model_cfg_kwargs["hidden_size"]
        n_heads = model_cfg_kwargs.get("num_attention_heads", cfg.get("num_attention_heads", 8))
        head_dim = h // n_heads
        if head_dim > 0:
            correct_n_heads = sniffed["num_attention_heads_total_dim"] // head_dim
            correct_n_kv    = sniffed["num_kv_heads_total_dim"]       // head_dim
            if correct_n_heads > 0:
                model_cfg_kwargs["num_attention_heads"] = correct_n_heads
            if correct_n_kv > 0:
                model_cfg_kwargs["num_key_value_heads"] = correct_n_kv

    return ChronosConfig(**model_cfg_kwargs), sniffed, sniff_note


def _run_inference_modes(cfg, selected_backend, selected_mode, model_path_val,
                         prompt_val, max_tok, temp, raw_prompt_val,
                         ram_load_ratio=1.0, sweep_ram_load_ratios=False,
                         miss_policy="on_demand") -> dict:
    from transformers import AutoTokenizer
    from chronos.backend.mac_diagnostics import mac_backend_diagnostics

    if not str(prompt_val or "").strip():
        return {"error": "Please enter a prompt."}

    backend_diag = mac_backend_diagnostics(configure_threads=True)
    model_cfg, sniffed, sniff_note = _build_model_cfg(cfg, model_path_val)
    backend, backend_note = _resolve_inference_backend(selected_backend, model_path_val, sniffed)
    cpu_threads_cfg = (cfg or {}).get("cpu_threads", "auto")
    cpu_budget_cfg = (cfg or {}).get("cpu_budget_percent", 100)
    tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
    token_ids, attn_mask = _encode_prompt(tokenizer, prompt_val, bool(raw_prompt_val))

    mode = _normalize_inference_mode(selected_mode)
    miss_policy = _normalize_miss_policy(miss_policy)
    rows: list[dict] = []
    outputs = {"lazy": "", "full_dram": "", "single": ""}
    raw = {
        "mode": mode,
        "backend": backend,
        "backend_note": backend_note.strip(),
        "checkpoint_note": sniff_note.strip(),
        "backend_diagnostics": backend_diag,
        "expert_budget_policy": {
            "source": "checkpoint num_experts_per_tok",
            "cap": "floor(active_experts * 1.25)",
            "miss_policy": miss_policy,
            **_bounded_offload_expert_budget(model_cfg, ram_load_ratio),
        },
        "rows": rows,
    }

    ratio_values = [_normalize_ram_load_ratio(ram_load_ratio)]
    if mode in {"compare", "offload"} and bool(sweep_ram_load_ratios):
        ratio_values = list(RAM_LOAD_SWEEP_RATIOS)
    ratio_values = list(dict.fromkeys(float(r) for r in ratio_values))

    representative_lazy_text = ""
    if mode in {"compare", "offload"}:
        for ratio in ratio_values:
            run_model_cfg = _clone_model_cfg(model_cfg)
            mem0 = _memory_snapshot_gb()
            t0 = time.monotonic()
            if backend == "mlx":
                lazy_tokens, lazy_stats = _run_mlx_lazy_inference(
                    run_model_cfg,
                    model_path_val,
                    token_ids,
                    int(max_tok),
                    float(temp),
                    do_sample=(mode != "compare"),
                    ram_load_ratio=ratio,
                    quality_safe=(mode == "compare" and not bool(sweep_ram_load_ratios)),
                )
            else:
                lazy_tokens, lazy_stats = _run_torch_inference(
                    backend,
                    run_model_cfg,
                    model_path_val,
                    token_ids,
                    attn_mask,
                    int(max_tok),
                    float(temp),
                    tokenizer.eos_token_id,
                    do_sample=(mode != "compare"),
                    ram_load_ratio=ratio,
                    miss_policy=miss_policy,
                    cpu_threads=cpu_threads_cfg,
                    cpu_budget_percent=cpu_budget_cfg,
                )
            lazy_elapsed = time.monotonic() - t0
            lazy_mem = _memory_snapshot_gb()
            lazy_text = tokenizer.decode(lazy_tokens, skip_special_tokens=True)
            row = _row_from_stats(
                "lazy_offload", backend, lazy_tokens, lazy_elapsed,
                lazy_mem - mem0, lazy_stats,
            )
            if len(ratio_values) > 1:
                row["mode"] = f"lazy_offload_{ratio:g}x"
            row["ram_load_ratio"] = round(float(ratio), 3)
            rows.append(row)
            if abs(float(ratio) - _normalize_ram_load_ratio(ram_load_ratio)) < 1e-9:
                representative_lazy_text = lazy_text
        if not representative_lazy_text and rows:
            representative_lazy_text = lazy_text
        outputs["lazy"] = representative_lazy_text

    full_text = ""
    if mode in {"compare", "fullload"}:
        full_model_cfg = _clone_model_cfg(model_cfg)
        mem1 = _memory_snapshot_gb()
        t1 = time.monotonic()
        if backend == "mlx":
            full_tokens, full_stats = _run_mlx_full_dram(
                full_model_cfg,
                model_path_val,
                token_ids,
                int(max_tok),
                float(temp),
                do_sample=(mode != "compare"),
            )
        else:
            full_tokens, full_stats = _run_torch_full_dram(
                backend,
                full_model_cfg,
                model_path_val,
                token_ids,
                attn_mask,
                int(max_tok),
                float(temp),
                tokenizer.eos_token_id,
                do_sample=(mode != "compare"),
                cpu_threads=cpu_threads_cfg,
                cpu_budget_percent=cpu_budget_cfg,
            )
        full_elapsed = time.monotonic() - t1
        full_mem = _memory_snapshot_gb()
        full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
        rows.append(_row_from_stats(
            "full_dram", backend, full_tokens, full_elapsed,
            full_mem - mem1, full_stats,
        ))
        outputs["full_dram"] = full_text

    diff = _text_diff_summary(representative_lazy_text, full_text) if mode == "compare" else None
    if diff:
        raw["output_diff"] = diff
    raw["rows"] = rows
    raw["outputs"] = outputs
    raw["chart"] = _rows_to_chart_records(rows)
    if mode == "offload":
        outputs["single"] = representative_lazy_text
    elif mode == "fullload":
        outputs["single"] = full_text
    return raw


def generate_api(cfg, selected_backend="auto", selected_mode="compare", model_path_val="",
                 prompt_val="", max_tok=128, temp=0.85, raw_prompt_val=False,
                 ram_load_ratio=1.0, sweep_ram_load_ratios=False,
                 miss_policy="on_demand") -> dict:
    try:
        return _run_inference_modes(
            cfg, selected_backend, selected_mode, model_path_val, prompt_val,
            max_tok, temp, raw_prompt_val, ram_load_ratio, sweep_ram_load_ratios,
            miss_policy,
        )
    except Exception as e:
        import traceback

        return {"error": str(e), "traceback": traceback.format_exc()}


def build_inference_tab(config_state: gr.State):
    with gr.Tab(t("tab.inference")) as tab:
        register_translatable(tab, "tab.inference")

        model_path = gr.Textbox(
            label=t("infer.model_path"), placeholder="./out/chronos_512_moe.pth"
        )
        register_translatable(model_path, "infer.model_path")

        prompt = gr.Textbox(
            label=t("infer.prompt"), lines=4,
            placeholder="Explain how Chronos handles expert prefetch during prefill..."
        )
        register_translatable(prompt, "infer.prompt")

        with gr.Row():
            inference_backend = gr.Dropdown(
                choices=_available_inference_backend_choices(),
                value=_default_inference_backend_value(),
                label=t("infer.backend"),
            )
            inference_mode = gr.Dropdown(
                choices=list(INFERENCE_MODE_CHOICES),
                value="compare",
                label=t("infer.mode"),
            )
            max_tokens  = gr.Slider(16, 512, value=128, step=16, label=t("infer.max_tokens"))
            temperature = gr.Slider(0.1, 2.0, value=0.85, step=0.05, label=t("infer.temperature"))
            register_translatable(inference_backend, "infer.backend")
            register_translatable(inference_mode, "infer.mode")
            register_translatable(max_tokens,  "infer.max_tokens")
            register_translatable(temperature, "infer.temperature")
        raw_prompt = gr.Checkbox(value=False, label=t("infer.raw_prompt"))
        register_translatable(raw_prompt, "infer.raw_prompt")
        with gr.Row():
            miss_policy = gr.Dropdown(
                choices=list(MISS_POLICY_CHOICES),
                value="on_demand",
                label=t("infer.miss_policy"),
            )
            ram_load_ratio = gr.Dropdown(
                choices=list(RAM_LOAD_RATIO_CHOICES),
                value="1.00",
                label=t("infer.ram_load_ratio"),
                allow_custom_value=True,
                filterable=True,
            )
            sweep_ratios = gr.Checkbox(
                value=False,
                label=t("infer.sweep_ram_load_ratios"),
            )
            register_translatable(miss_policy, "infer.miss_policy")
            register_translatable(ram_load_ratio, "infer.ram_load_ratio")
            register_translatable(sweep_ratios, "infer.sweep_ram_load_ratios")

        gen_btn = gr.Button(t("infer.generate"), variant="primary")
        register_translatable(gen_btn, "infer.generate")

        output_box = gr.Textbox(
            label=t("infer.output"), lines=10, interactive=False, visible=False,
        )
        with gr.Row():
            lazy_output_box = gr.Textbox(
                label=t("infer.lazy_output"), lines=10, interactive=False, visible=True,
            )
            full_output_box = gr.Textbox(
                label=t("infer.full_output"), lines=10, interactive=False, visible=True,
            )
        stats_md = gr.Markdown(value=_format_inference_stats([]))
        stats_chart = gr.LinePlot(
            value=_empty_inference_df(),
            x="x", y="normalized_value", color="metric",
            title="Inference mode comparison",
            x_title="Mode / RAM load ratio",
            y_title="Normalized value",
            tooltip=["mode", "metric", "value", "unit"],
            sort=None,
            height=520,
        )
        raw_stats = gr.JSON(label=t("infer.compare_table"), value={}, visible=False)
        register_translatable(output_box, "infer.output")
        register_translatable(lazy_output_box, "infer.lazy_output")
        register_translatable(full_output_box, "infer.full_output")
        register_translatable(raw_stats, "infer.compare_table")

        def _sniff_checkpoint(path: str) -> dict:
            """Inspect a Chronos .pth and return the topology fields encoded
            in the saved tensor shapes. Returns {} if the file is unreadable
            or doesn't look like a Chronos checkpoint.

            Sniffed fields:
                vocab_size, hidden_size, num_hidden_layers, num_experts,
                moe_intermediate_size, num_attention_heads, num_key_value_heads.
            """
            import os as _os
            if not path or not _os.path.exists(path):
                return {}
            try:
                import torch as _t
                sd = _t.load(path, map_location="cpu")
                if not isinstance(sd, dict):
                    return {}
                out = {}

                embed = sd.get("model.embed_tokens.weight")
                if embed is not None:
                    out["vocab_size"], out["hidden_size"] = int(embed.shape[0]), int(embed.shape[1])

                # Count layers via highest layer index in keys.
                layer_idxs = set()
                for k in sd.keys():
                    if k.startswith("model.layers."):
                        try:
                            layer_idxs.add(int(k.split(".")[2]))
                        except (ValueError, IndexError):
                            pass
                if layer_idxs:
                    out["num_hidden_layers"] = max(layer_idxs) + 1

                # Count experts on layer 0.
                expert_idxs = set()
                for k in sd.keys():
                    if k.startswith("model.layers.0.mlp.experts."):
                        try:
                            expert_idxs.add(int(k.split(".")[5]))
                        except (ValueError, IndexError):
                            pass
                if expert_idxs:
                    out["num_experts"] = max(expert_idxs) + 1

                # Per-expert FFN width from gate_proj shape (intermediate, hidden).
                gate = sd.get("model.layers.0.mlp.experts.0.gate_proj.weight")
                if gate is not None:
                    out["moe_intermediate_size"] = int(gate.shape[0])

                # Attention head counts from q/k_proj shapes.
                qw = sd.get("model.layers.0.self_attn.q_proj.weight")
                kw = sd.get("model.layers.0.self_attn.k_proj.weight")
                if qw is not None and kw is not None and out.get("hidden_size"):
                    h = out["hidden_size"]
                    # qw: (n_heads * head_dim, hidden); kw: (n_kv * head_dim, hidden)
                    # head_dim is typically hidden_size // n_heads, but we can't
                    # split q without knowing head_dim. Conservative: assume the
                    # ratio matches MiniMind defaults (head_dim = h // num_heads).
                    # Solve via the kw / qw ratio assuming both share head_dim.
                    out["num_attention_heads_total_dim"] = int(qw.shape[0])
                    out["num_kv_heads_total_dim"] = int(kw.shape[0])

                # MLA attention uses split query and latent KV projections:
                #   q_nope_proj: (n_heads * nope_dim, hidden)
                #   q_rope_proj: (n_heads * rope_dim, hidden)
                #   kv_down_proj: (kv_latent_dim, hidden)
                #   v_proj: (n_kv_heads * head_dim, kv_latent_dim)
                qnope = sd.get("model.layers.0.self_attn.q_nope_proj.weight")
                qrope = sd.get("model.layers.0.self_attn.q_rope_proj.weight")
                kvdown = sd.get("model.layers.0.self_attn.kv_down_proj.weight")
                vproj = sd.get("model.layers.0.self_attn.v_proj.weight")
                if qnope is not None and qrope is not None and out.get("hidden_size"):
                    total_q = int(qnope.shape[0] + qrope.shape[0])
                    h = out["hidden_size"]
                    if total_q > 0 and h % total_q == 0:
                        # For MiniMind/Chronos checkpoints head_dim is
                        # hidden_size / num_attention_heads, so total_q equals
                        # hidden_size and n_heads is recovered by divisibility.
                        pass
                    candidates = [
                        n for n in range(1, out["hidden_size"] + 1)
                        if out["hidden_size"] % n == 0
                        and int(qrope.shape[0]) % n == 0
                        and int(qnope.shape[0]) % n == 0
                    ]
                    if candidates:
                        # Prefer the configured/default 8 when valid; else
                        # choose the largest plausible head count.
                        preferred = 8 if 8 in candidates else max(candidates)
                        head_dim = out["hidden_size"] // preferred
                        rope_dim = int(qrope.shape[0]) // preferred
                        nope_dim = int(qnope.shape[0]) // preferred
                        if rope_dim + nope_dim == head_dim:
                            out["num_attention_heads"] = preferred
                            out["rope_dim"] = rope_dim
                            out["num_attention_heads_total_dim"] = preferred * head_dim
                if kvdown is not None:
                    out["kv_latent_dim"] = int(kvdown.shape[0])
                if vproj is not None and out.get("kv_latent_dim"):
                    head_dim = out.get("hidden_size", 0) // max(out.get("num_attention_heads", 8), 1)
                    if head_dim > 0:
                        out["num_key_value_heads"] = max(1, int(vproj.shape[0]) // head_dim)
                        out["num_kv_heads_total_dim"] = int(vproj.shape[0])

                # Odd layers are SlidingWindowAttention in the hybrid stack.
                # If layer 1 exists, its q/k projection shapes are a direct
                # source of head counts and should override weaker defaults.
                sw_q = sd.get("model.layers.1.self_attn.q_proj.weight")
                sw_k = sd.get("model.layers.1.self_attn.k_proj.weight")
                if sw_q is not None and sw_k is not None and out.get("hidden_size"):
                    out["num_attention_heads_total_dim"] = int(sw_q.shape[0])
                    out["num_kv_heads_total_dim"] = int(sw_k.shape[0])
                return out
            except Exception:
                return {}

        def _build_model_cfg(cfg, model_path_val):
            from chronos.model.config import ChronosConfig
            from chronos.model.checkpoint import config_dict_for_checkpoint, sniff_checkpoint_config
            from chronos.export import config_dict_from_export, is_export_artifact

            cfg = cfg or {}
            cfg_sources = []
            try:
                if model_path_val and is_export_artifact(model_path_val):
                    ckpt_cfg = config_dict_from_export(model_path_val)
                    cfg_sources = [model_path_val]
                else:
                    ckpt_cfg, cfg_sources = config_dict_for_checkpoint(
                        model_path_val if model_path_val else None,
                        require_unsniffable=False,
                    )
            except Exception:
                ckpt_cfg, cfg_sources = {}, []
            sniffed = ckpt_cfg or (sniff_checkpoint_config(model_path_val) if model_path_val else {})
            sniff_note = ""
            if sniffed:
                summary = ", ".join(
                    f"{k}={v}" for k, v in sniffed.items()
                    if k in ("vocab_size", "hidden_size", "num_hidden_layers",
                             "num_experts", "moe_intermediate_size")
                )
                source_text = "; ".join(cfg_sources) if cfg_sources else model_path_val
                sniff_note = f"[checkpoint config: {summary} | source={source_text}]\n"

            model_cfg_kwargs = {
                "hidden_size":          sniffed.get("hidden_size",        cfg.get("hidden_size", 768)),
                "num_hidden_layers":    sniffed.get("num_hidden_layers",  cfg.get("num_hidden_layers", 8)),
                "num_experts":          sniffed.get("num_experts",        cfg.get("num_experts", 4)),
                "num_experts_per_tok":  sniffed.get("num_experts_per_tok", cfg.get("num_experts_per_tok", 1)),
                "num_shared_experts":   sniffed.get("num_shared_experts", cfg.get("num_shared_experts", 1)),
                "lookahead_steps":      sniffed.get("lookahead_steps", cfg.get("lookahead_steps", 2)),
                "kv_latent_dim":        sniffed.get("kv_latent_dim", cfg.get("kv_latent_dim", 64)),
                "sliding_window_size":  sniffed.get("sliding_window_size", cfg.get("sliding_window_size", 2048)),
                "use_hybrid_attention": sniffed.get("use_hybrid_attention", cfg.get("use_hybrid_attention", True)),
                "vram_budget_gb":       sniffed.get("vram_budget_gb", cfg.get("vram_budget_gb", 4.0)),
                "pinned_memory_max_fraction": sniffed.get("pinned_memory_max_fraction", cfg.get("pinned_memory_max_fraction", 0.25)),
                "storage_format":       sniffed.get("storage_format", cfg.get("storage_format", "safetensors")),
                "use_moe": True,
            }
            if "cluster_manifest_path" in sniffed:
                model_cfg_kwargs["cluster_manifest_path"] = sniffed["cluster_manifest_path"]
            if "rope_dim" in sniffed:
                model_cfg_kwargs["rope_dim"] = sniffed["rope_dim"]
            if "num_attention_heads" in sniffed:
                model_cfg_kwargs["num_attention_heads"] = sniffed["num_attention_heads"]
            if "num_key_value_heads" in sniffed:
                model_cfg_kwargs["num_key_value_heads"] = sniffed["num_key_value_heads"]
            if "vocab_size" in sniffed:
                model_cfg_kwargs["vocab_size"] = sniffed["vocab_size"]
            if "moe_intermediate_size" in sniffed:
                model_cfg_kwargs["intermediate_size"] = sniffed["moe_intermediate_size"]
                model_cfg_kwargs["moe_intermediate_size"] = sniffed["moe_intermediate_size"]

            for opt_key, cfg_key in [
                ("num_attention_heads",   "num_attention_heads"),
                ("num_key_value_heads",   "num_key_value_heads"),
                ("rope_dim",              "rope_dim"),
                ("max_position_embeddings", "max_seq_len"),
                ("tie_word_embeddings",   "tie_word_embeddings"),
            ]:
                if opt_key in {"num_attention_heads", "num_key_value_heads", "rope_dim"} \
                        and opt_key in model_cfg_kwargs:
                    continue
                if cfg_key not in cfg:
                    continue
                val = cfg[cfg_key]
                if val in (None, "", 0):
                    continue
                model_cfg_kwargs[opt_key] = val

            if "num_attention_heads_total_dim" in sniffed:
                h = model_cfg_kwargs["hidden_size"]
                n_heads = model_cfg_kwargs.get("num_attention_heads", cfg.get("num_attention_heads", 8))
                head_dim = h // n_heads
                if head_dim > 0:
                    correct_n_heads = sniffed["num_attention_heads_total_dim"] // head_dim
                    correct_n_kv    = sniffed["num_kv_heads_total_dim"]       // head_dim
                    if correct_n_heads > 0:
                        model_cfg_kwargs["num_attention_heads"] = correct_n_heads
                    if correct_n_kv > 0:
                        model_cfg_kwargs["num_key_value_heads"] = correct_n_kv

            return ChronosConfig(**model_cfg_kwargs), sniffed, sniff_note

        def generate_unified(cfg, selected_backend, selected_mode, model_path_val,
                             prompt_val, max_tok, temp, raw_prompt_val,
                             miss_policy_val, ram_load_ratio_val, sweep_ratios_val):
            raw = generate_api(
                cfg,
                selected_backend=selected_backend,
                selected_mode=selected_mode,
                model_path_val=model_path_val,
                prompt_val=prompt_val,
                max_tok=max_tok,
                temp=temp,
                raw_prompt_val=raw_prompt_val,
                ram_load_ratio=ram_load_ratio_val,
                sweep_ram_load_ratios=sweep_ratios_val,
                miss_policy=miss_policy_val,
            )
            if raw.get("error"):
                text = f"Error: {raw.get('error')}"
                if raw.get("traceback"):
                    text = f"{text}\n{raw['traceback']}"
                return (
                    gr.update(value=text, visible=True),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    f"_error: {raw.get('error')}_",
                    _empty_inference_df(),
                    gr.update(value=raw, visible=True),
                )

            rows = raw.get("rows", [])
            diff = raw.get("output_diff")
            mode = _normalize_inference_mode(raw.get("mode"))
            outputs = raw.get("outputs", {})
            table_md = _format_inference_stats(rows, diff)
            chart_df = _rows_to_chart_df(rows)
            if mode == "compare":
                return (
                    gr.update(value="", visible=False),
                    gr.update(value=outputs.get("lazy", ""), visible=True),
                    gr.update(value=outputs.get("full_dram", ""), visible=True),
                    table_md,
                    chart_df,
                    gr.update(value=raw, visible=True),
                )
            return (
                gr.update(value=outputs.get("single", ""), visible=True),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                table_md,
                chart_df,
                gr.update(value=raw, visible=True),
            )

        gen_btn.click(
            fn=generate_unified,
            inputs=[
                config_state, inference_backend, inference_mode, model_path, prompt,
                max_tokens, temperature, raw_prompt, miss_policy, ram_load_ratio, sweep_ratios,
            ],
            outputs=[output_box, lazy_output_box, full_output_box, stats_md, stats_chart, raw_stats],
            api_name=None,
            api_visibility="private",
        )

        api_btn = gr.Button("api_generate", visible=False)
        api_out = gr.JSON(value={}, visible=False)
        api_btn.click(
            fn=generate_api,
            inputs=[
                config_state, inference_backend, inference_mode, model_path, prompt,
                max_tokens, temperature, raw_prompt, ram_load_ratio, sweep_ratios, miss_policy,
            ],
            outputs=[api_out],
            api_name="generate",
        )

    return tab
