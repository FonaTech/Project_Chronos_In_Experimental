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
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


# ── Co-occurrence matrix ──────────────────────────────────────────

def collect_activation_log(model, dataloader, device, max_batches=50) -> List[List[int]]:
    """
    Run calibration forward passes and record which expert is selected
    at each token position (layer 0 routing, top-1).

    Returns: list of sequences, each sequence is a list of expert IDs.
    """
    from chronos.model.moe_chronos import ChronosMOEFeedForward

    moe_layers = [l for l in model.model.layers if isinstance(l.mlp, ChronosMOEFeedForward)]
    if not moe_layers:
        return []

    model.eval()
    activation_log = []

    with torch.no_grad():
        for i, (input_ids, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            input_ids = input_ids.to(device)
            model(input_ids, use_cache=False)
            # Use layer 0 routing as representative
            probs = moe_layers[0].last_router_probs  # [B, S, E]
            if probs is None:
                continue
            top1 = probs.argmax(dim=-1)  # [B, S]
            for b in range(top1.shape[0]):
                activation_log.append(top1[b].cpu().tolist())

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


def try_louvain_clustering(cooccurrence: np.ndarray) -> List[List[int]]:
    """
    Attempt Louvain community detection via python-louvain.
    Falls back to greedy clustering if not available.
    """
    try:
        import community as community_louvain
        import networkx as nx

        G = nx.Graph()
        n = cooccurrence.shape[0]
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
        return list(clusters_dict.values())

    except ImportError:
        return cluster_experts_greedy(cooccurrence)


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
        clusters=clusters,
        output_dir=output_dir,
        num_layers=num_layers,
        num_experts=num_experts,
        dtype=dtype if dtype is not None else _torch.float16,
    )
    return manifest_path


# ── CLI entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chronos Expert Cluster Layout Generator")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ssd_dir", type=str, default="./expert_cache")
    parser.add_argument("--output_dir", type=str, default="./expert_cache_clustered")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    args = parser.parse_args()

    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from dataset.lm_dataset import PretrainDataset

    config = ChronosConfig(use_moe=True)
    model = ChronosForCausalLM(config).to(args.device)
    weights = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(weights, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(
        chronos.deps.get_tokenizer_path()
    )
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Collecting activation log...")
    log = collect_activation_log(model, loader, args.device, args.max_batches)
    print(f"  {len(log)} sequences collected")

    C = build_cooccurrence_matrix(log, config.num_experts)
    print(f"Co-occurrence matrix:\n{np.round(C, 3)}")

    clusters = try_louvain_clustering(C)
    layout_path = repack_expert_weights(
        args.ssd_dir, clusters, args.output_dir,
        num_layers=config.num_hidden_layers,
    )
    print(f"Done. Layout: {layout_path}")


if __name__ == "__main__":
    main()
