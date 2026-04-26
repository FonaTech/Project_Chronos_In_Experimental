"""
chronos/io/cluster_layout.py

Offline expert co-occurrence analysis and clustered storage layout generator.

Pipeline:
1. Collect expert activation logs during a calibration forward pass
2. Build co-occurrence frequency matrix from logs
3. Run Louvain community detection (or greedy clustering) to group
   frequently co-activated experts into contiguous physical blocks
4. Repack model weights on SSD so co-occurring experts are stored
   sequentially → maximizes NVMe sequential read bandwidth

Usage:
    python -m chronos.io.cluster_layout \
        --model_path ./out/chronos_512_moe.pth \
        --data_path  ./dataset/calib.jsonl \
        --output_dir ./expert_cache_clustered
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import os
import json
import argparse
import collections
from typing import Any, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# ── Co-occurrence matrix ──────────────────────────────────────────

def _batch_input_ids_and_mask(
    batch: Any,
    device: str | torch.device,
    pad_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        input_ids = batch.get("input_ids")
        if input_ids is None:
            input_ids = batch.get("x")
        attention_mask = batch.get("attention_mask")
    elif isinstance(batch, (tuple, list)):
        input_ids = batch[0]
        attention_mask = None
    else:
        input_ids = batch
        attention_mask = None

    if input_ids is None:
        raise ValueError("calibration batch does not contain input_ids")
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    elif pad_token_id is not None:
        attention_mask = (input_ids != int(pad_token_id)).to(torch.long)
    return input_ids, attention_mask


def collect_activation_log(
    model,
    dataloader,
    device,
    max_batches=50,
    *,
    pad_token_id: int | None = None,
    top_k: int | None = None,
) -> List[List[int]]:
    """
    Run calibration forward passes and record expert routing events.

    Older versions only used layer-0 top-1 routing. That under-represents the
    SSD/RAM access pattern because the cache manager must serve all MoE layers
    and all active top-k experts. This collector records the union of all
    selected experts across MoE layers for every token position, preserving
    token order so the co-occurrence window captures temporal locality.

    Returns: list of sequences, each sequence is a list of expert IDs.
    """
    from chronos.model.moe_chronos import ChronosMOEFeedForward

    moe_layers = [
        layer.mlp for layer in model.model.layers
        if isinstance(layer.mlp, ChronosMOEFeedForward)
    ]
    if not moe_layers:
        return []

    model.eval()
    activation_log = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids, attention_mask = _batch_input_ids_and_mask(
                batch, device, pad_token_id=pad_token_id
            )
            model(input_ids, attention_mask=attention_mask, use_cache=False)

            topk_by_layer = []
            for moe in moe_layers:
                probs = moe.last_router_probs  # [B, S, E]
                if probs is None:
                    continue
                k = int(top_k or getattr(moe, "num_experts_per_tok", 1) or 1)
                k = max(1, min(k, probs.shape[-1]))
                topk_by_layer.append(torch.topk(probs, k=k, dim=-1).indices.cpu())
            if not topk_by_layer:
                continue

            B, S = input_ids.shape
            mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None
            for b in range(B):
                seq: list[int] = []
                for t in range(S):
                    if mask_cpu is not None and int(mask_cpu[b, t].item()) == 0:
                        continue
                    token_experts = set()
                    for layer_topk in topk_by_layer:
                        token_experts.update(int(e) for e in layer_topk[b, t].reshape(-1).tolist())
                    seq.extend(sorted(token_experts))
                if seq:
                    activation_log.append(seq)

    return activation_log


def build_cooccurrence_matrix(activation_log: List[List[int]], num_experts: int) -> np.ndarray:
    """
    Build symmetric co-occurrence matrix C where C[i,j] = number of times
    expert i and expert j are both activated within a sliding window of 4 tokens.
    """
    C = np.zeros((num_experts, num_experts), dtype=np.float32)
    window = 4

    for seq in activation_log:
        for t in range(len(seq)):
            anchor = seq[t]
            for k in range(1, window + 1):
                if t + k < len(seq):
                    neighbor = seq[t + k]
                    if anchor != neighbor:
                        C[anchor, neighbor] += 1
                        C[neighbor, anchor] += 1

    # Normalize
    row_sum = C.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return C / row_sum


def cluster_experts_greedy(
    cooccurrence: np.ndarray,
    n_clusters: int = None,
) -> List[List[int]]:
    """
    Greedy co-occurrence clustering: iteratively merge the pair of experts
    with highest co-occurrence score into the same cluster.

    Falls back to equal-size partitioning if cooccurrence is uniform.
    Returns list of clusters, each cluster is a list of expert IDs.
    """
    num_experts = cooccurrence.shape[0]
    if n_clusters is None:
        n_clusters = max(2, num_experts // 2)

    # Start: each expert in its own cluster
    clusters = {i: [i] for i in range(num_experts)}
    expert_to_cluster = {i: i for i in range(num_experts)}
    C = cooccurrence.copy()
    np.fill_diagonal(C, 0)

    merges_needed = num_experts - n_clusters
    for _ in range(merges_needed):
        # Find highest co-occurrence pair in different clusters
        best_score = -1
        best_i, best_j = 0, 1
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                ci, cj = expert_to_cluster[i], expert_to_cluster[j]
                if ci != cj and C[i, j] > best_score:
                    best_score = C[i, j]
                    best_i, best_j = i, j

        ci = expert_to_cluster[best_i]
        cj = expert_to_cluster[best_j]
        if ci == cj:
            continue
        # Merge cj into ci
        for eid in clusters[cj]:
            expert_to_cluster[eid] = ci
        clusters[ci].extend(clusters.pop(cj))

    return list(clusters.values())


def _normalize_clusters(clusters: List[List[int]], num_experts: int) -> List[List[int]]:
    seen = set()
    out: list[list[int]] = []
    for cluster in clusters:
        clean = []
        for eid in cluster:
            eid = int(eid)
            if 0 <= eid < num_experts and eid not in seen:
                clean.append(eid)
                seen.add(eid)
        if clean:
            out.append(sorted(clean))
    for eid in range(num_experts):
        if eid not in seen:
            out.append([eid])
    return out


def cluster_experts_louvain_or_greedy(cooccurrence: np.ndarray) -> tuple[List[List[int]], str]:
    """
    Attempt Louvain community detection via python-louvain.
    Falls back to greedy clustering if not available.
    """
    n = int(cooccurrence.shape[0])
    try:
        import community as community_louvain
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                w = float(cooccurrence[i, j])
                if w > 0:
                    G.add_edge(i, j, weight=w)

        partition = community_louvain.best_partition(G, weight='weight')
        clusters_dict = collections.defaultdict(list)
        for node, comm in partition.items():
            clusters_dict[comm].append(node)
        return _normalize_clusters(list(clusters_dict.values()), n), "louvain"

    except Exception:
        return _normalize_clusters(cluster_experts_greedy(cooccurrence), n), "greedy"


def try_louvain_clustering(cooccurrence: np.ndarray) -> List[List[int]]:
    return cluster_experts_louvain_or_greedy(cooccurrence)[0]


# ── Storage layout ────────────────────────────────────────────────

def build_cluster_layout(clusters: List[List[int]]) -> Dict:
    """
    Build a layout descriptor mapping expert_id → (cluster_id, position_in_cluster).
    """
    layout = {}
    for cid, cluster in enumerate(clusters):
        for pos, eid in enumerate(cluster):
            layout[eid] = {"cluster_id": cid, "position": pos, "cluster": cluster}
    return layout


def repack_expert_weights(
    ssd_dir: str,
    clusters: List[List[int]],
    output_dir: str,
    num_layers: int,
):
    """
    Rewrite expert .pt files in cluster order so that co-occurring experts
    are stored in contiguous blocks on disk (maximizes sequential read).

    Output naming: expert_l{layer}_c{cluster_id}_p{position}.pt
    Also writes cluster_layout.json for runtime use.
    """
    os.makedirs(output_dir, exist_ok=True)
    layout_meta = []

    for cid, cluster in enumerate(clusters):
        for pos, eid in enumerate(cluster):
            for li in range(num_layers):
                src = os.path.join(ssd_dir, f"expert_l{li}_e{eid}.pt")
                dst = os.path.join(output_dir, f"expert_l{li}_c{cid}_p{pos}.pt")
                if os.path.exists(src) and not os.path.exists(dst):
                    import shutil
                    shutil.copy2(src, dst)
            layout_meta.append({
                "expert_id": eid,
                "cluster_id": cid,
                "position": pos,
                "cluster_members": cluster,
            })

    layout_path = os.path.join(output_dir, "cluster_layout.json")
    with open(layout_path, "w") as f:
        json.dump(layout_meta, f, indent=2)

    print(f"Cluster layout saved: {layout_path}")
    print(f"Clusters ({len(clusters)}):")
    for cid, cluster in enumerate(clusters):
        print(f"  Cluster {cid}: experts {cluster}")
    return layout_path


def load_cluster_layout(layout_path: str) -> Dict[int, Dict]:
    """Load cluster_layout.json → {expert_id: metadata}."""
    with open(layout_path) as f:
        meta = json.load(f)
    return {entry["expert_id"]: entry for entry in meta}


# ── Safetensors cluster repack ────────────────────────────────────

def repack_expert_weights_safetensors(
    model,
    clusters: List[List[int]],
    output_dir: str,
    num_layers: int = None,
    dtype: "torch.dtype" = None,
) -> str:
    """
    Pack every expert of a live `ChronosForCausalLM` into cluster .ctsr files
    plus a cluster_manifest.json. This is the runtime path consumed by
    ExpertStore when `storage_format == "safetensors"`.

    Unlike the legacy `repack_expert_weights` (which copies .pt files around),
    this reads weights directly from the model's MoE layers so it works even
    before any offload has happened.

    Returns the path to cluster_manifest.json.
    """
    import torch as _torch
    from chronos.io.storage import ClusterStorage
    from chronos.model.moe_chronos import ChronosMOEFeedForward

    moe_layers = [
        layer.mlp for layer in model.model.layers
        if isinstance(layer.mlp, ChronosMOEFeedForward)
    ]
    if num_layers is None:
        num_layers = len(moe_layers)
    num_experts = moe_layers[0].num_experts if moe_layers else 0

    manifest_path = ClusterStorage.write_clusters(
        moe_layers=moe_layers,
        clusters=_normalize_clusters(clusters, num_experts),
        output_dir=output_dir,
        num_layers=num_layers,
        num_experts=num_experts,
        dtype=dtype if dtype is not None else _torch.float16,
    )
    return manifest_path


def _parse_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.float16
    normalized = str(dtype).lower().replace("torch.", "")
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype for clustered cache: {dtype}")


def build_clustered_expert_cache(
    model,
    config,
    data_path: str,
    output_dir: str,
    *,
    tokenizer_path: str | None = None,
    device: str = "cpu",
    max_batches: int = 50,
    batch_size: int = 4,
    max_seq_len: int = 256,
    dtype: str | torch.dtype | None = torch.float16,
    source_checkpoint: str | None = None,
) -> dict:
    """
    Build a cluster-aware safetensors expert cache from a live model.

    This is the post-training calibration path: run representative data once,
    infer expert co-occurrence, cluster experts, and write .ctsr shards plus
    cluster_manifest.json for Chronos lazy loading.
    """
    import chronos.deps
    from chronos.data.flexible_dataset import FlexibleDataset
    from chronos.model.moe_chronos import ChronosMOEFeedForward

    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"calibration data not found: {data_path}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or chronos.deps.get_tokenizer_path())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = FlexibleDataset(data_path, tokenizer, max_length=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()
    activation_log = collect_activation_log(
        model,
        loader,
        device,
        max_batches=max_batches,
        pad_token_id=tokenizer.pad_token_id,
        top_k=getattr(config, "num_experts_per_tok", None),
    )
    num_experts = int(getattr(config, "num_experts", 0))
    if num_experts <= 0:
        raise ValueError("config.num_experts must be positive to build clustered expert cache")

    C = build_cooccurrence_matrix(activation_log, num_experts)
    clusters, method = cluster_experts_louvain_or_greedy(C)
    manifest_path = repack_expert_weights_safetensors(
        model,
        clusters,
        output_dir,
        num_layers=int(getattr(config, "num_hidden_layers", 0) or 0),
        dtype=_parse_torch_dtype(dtype),
    )

    moe_layers = [
        layer.mlp for layer in model.model.layers
        if isinstance(layer.mlp, ChronosMOEFeedForward)
    ]
    summary = {
        "source_checkpoint": os.path.abspath(source_checkpoint) if source_checkpoint else None,
        "calibration_data_path": os.path.abspath(data_path),
        "tokenizer_path": tokenizer_path or chronos.deps.get_tokenizer_path(),
        "output_dir": os.path.abspath(output_dir),
        "manifest_path": os.path.abspath(manifest_path),
        "method": method,
        "num_sequences": len(activation_log),
        "num_experts": num_experts,
        "num_moe_layers": len(moe_layers),
        "max_batches": int(max_batches),
        "batch_size": int(batch_size),
        "max_seq_len": int(max_seq_len),
        "dtype": str(_parse_torch_dtype(dtype)).replace("torch.", ""),
        "clusters": clusters,
        "cooccurrence_matrix": np.round(C, 6).tolist(),
    }
    summary_path = os.path.join(output_dir, "cluster_build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    summary["summary_path"] = os.path.abspath(summary_path)
    return summary


def build_clustered_expert_cache_from_checkpoint(
    model_path: str,
    data_path: str,
    output_dir: str,
    *,
    config_path: str | None = None,
    tokenizer_path: str | None = None,
    device: str = "cpu",
    max_batches: int = 50,
    batch_size: int = 4,
    max_seq_len: int = 256,
    dtype: str | torch.dtype | None = torch.float16,
) -> dict:
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import (
        chronos_config_from_checkpoint,
        load_checkpoint_state_dict,
        load_state_dict_controlled,
    )

    config, _sources = chronos_config_from_checkpoint(
        model_path,
        project_config_path=config_path,
        require_unsniffable=True,
    )
    model = ChronosForCausalLM(config)
    weights = load_checkpoint_state_dict(model_path, map_location="cpu")
    load_state_dict_controlled(model, weights)
    return build_clustered_expert_cache(
        model,
        config,
        data_path,
        output_dir,
        tokenizer_path=tokenizer_path,
        device=device,
        max_batches=max_batches,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        source_checkpoint=model_path,
    )


# ── CLI entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chronos Expert Cluster Layout Generator")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ssd_dir", type=str, default="./expert_cache")
    parser.add_argument("--output_dir", type=str, default="./expert_cache_clustered")
    parser.add_argument("--format", choices=["safetensors", "pt"], default="safetensors")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    if args.format == "safetensors":
        summary = build_clustered_expert_cache_from_checkpoint(
            args.model_path,
            args.data_path,
            args.output_dir,
            config_path=args.config_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            max_batches=args.max_batches,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            dtype=torch.float16,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    from chronos.model.checkpoint import chronos_config_from_checkpoint
    from chronos.data.flexible_dataset import FlexibleDataset
    config, sources = chronos_config_from_checkpoint(
        args.model_path,
        project_config_path=args.config_path,
        require_unsniffable=True,
    )
    print(f"Config sources: {', '.join(sources)}")
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import load_checkpoint_state_dict, load_state_dict_controlled

    model = ChronosForCausalLM(config)
    weights = load_checkpoint_state_dict(args.model_path, map_location="cpu")
    load_state_dict_controlled(model, weights)
    model = model.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or chronos.deps.get_tokenizer_path()
    )
    dataset = FlexibleDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Collecting activation log...")
    log = collect_activation_log(
        model,
        loader,
        args.device,
        args.max_batches,
        pad_token_id=tokenizer.pad_token_id,
        top_k=config.num_experts_per_tok,
    )
    print(f"  {len(log)} sequences collected")

    C = build_cooccurrence_matrix(log, config.num_experts)
    print(f"Co-occurrence matrix:\n{np.round(C, 3)}")

    clusters, method = cluster_experts_louvain_or_greedy(C)
    print(f"Clustering method: {method}")
    layout_path = repack_expert_weights(
        args.ssd_dir, clusters, args.output_dir,
        num_layers=config.num_hidden_layers,
    )
    print(f"Done. Layout: {layout_path}")


if __name__ == "__main__":
    main()
