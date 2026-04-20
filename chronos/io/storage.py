"""
chronos/io/storage.py

Cluster-aware safetensors backend for Chronos expert weights.

A .ctsr file is a plain safetensors container packing every expert in one
Louvain co-occurrence cluster. Tensor keys follow the pattern
``l{layer_id}_e{expert_id}__{param_name}``.

A sidecar ``cluster_manifest.json`` maps (cluster_id) → (file, expert_ids)
and (expert_id) → cluster_id so the runtime can:

1. Look up which .ctsr file owns an expert in O(1).
2. Read an entire cluster in a single mmap + zero-copy tensor view, turning
   N random .pt pickle decodes into one sequential pread.

The format intentionally stays minimal — no compression, no encryption, no
tensor sharding — so that it maps cleanly onto future io_uring / DirectIO
backends without schema churn.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import save_file as _st_save_file
    from safetensors import safe_open as _st_safe_open
    _HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover
    _HAS_SAFETENSORS = False


MANIFEST_FILENAME = "cluster_manifest.json"
MANIFEST_VERSION = 1
KEY_SEP = "__"  # layer/expert prefix vs param-name separator


def _key(layer_id: int, expert_id: int, param_name: str) -> str:
    return f"l{layer_id}_e{expert_id}{KEY_SEP}{param_name}"


def _parse_key(key: str) -> Optional[Tuple[int, int, str]]:
    """Inverse of _key. Returns (layer_id, expert_id, param_name) or None."""
    if KEY_SEP not in key:
        return None
    prefix, param_name = key.split(KEY_SEP, 1)
    if not (prefix.startswith("l") and "_e" in prefix):
        return None
    try:
        layer_str, expert_str = prefix[1:].split("_e", 1)
        return int(layer_str), int(expert_str), param_name
    except ValueError:
        return None


@dataclass
class ClusterManifest:
    """Parsed cluster_manifest.json."""
    version: int
    num_experts: int
    num_layers: int
    storage_format: str
    # cluster_id -> (filename, expert_ids)
    clusters: Dict[int, Tuple[str, List[int]]]
    # expert_id -> cluster_id
    expert_to_cluster: Dict[int, int]

    @classmethod
    def from_dict(cls, data: Dict) -> "ClusterManifest":
        clusters: Dict[int, Tuple[str, List[int]]] = {}
        for entry in data["clusters"]:
            clusters[int(entry["cluster_id"])] = (
                entry["file"],
                [int(e) for e in entry["experts"]],
            )
        e2c = {int(k): int(v) for k, v in data["expert_to_cluster"].items()}
        return cls(
            version=int(data.get("version", MANIFEST_VERSION)),
            num_experts=int(data["num_experts"]),
            num_layers=int(data["num_layers"]),
            storage_format=data.get("storage_format", "safetensors"),
            clusters=clusters,
            expert_to_cluster=e2c,
        )

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "num_experts": self.num_experts,
            "num_layers": self.num_layers,
            "storage_format": self.storage_format,
            "clusters": [
                {"cluster_id": cid, "file": fname, "experts": list(eids)}
                for cid, (fname, eids) in sorted(self.clusters.items())
            ],
            "expert_to_cluster": {str(k): v for k, v in self.expert_to_cluster.items()},
        }

    def cluster_members(self, expert_id: int) -> List[int]:
        cid = self.expert_to_cluster[expert_id]
        return list(self.clusters[cid][1])

    def cluster_file(self, cluster_id: int) -> str:
        return self.clusters[cluster_id][0]


class ClusterStorage:
    """
    Reader/writer for cluster-packed expert weights.

    Write path (one-shot, offline):
        ClusterStorage.write_clusters(
            moe_layers, clusters, output_dir, num_layers, num_experts
        )

    Read path (runtime, hot loop):
        storage = ClusterStorage(output_dir)
        per_layer = storage.load_cluster(cluster_id, dtype=torch.float16)
        # per_layer: {expert_id: {layer_id: {param_name: Tensor}}}
    """

    def __init__(self, root_dir: str):
        if not _HAS_SAFETENSORS:
            raise ImportError(
                "ClusterStorage requires the `safetensors` package. "
                "Install with `pip install safetensors>=0.4.0`."
            )
        self.root_dir = root_dir
        manifest_path = os.path.join(root_dir, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"cluster_manifest.json not found at {manifest_path}. "
                "Run offload_all_to_ssd() or repack_expert_weights_safetensors() first."
            )
        with open(manifest_path) as f:
            self.manifest = ClusterManifest.from_dict(json.load(f))

    # ── read path ────────────────────────────────────────────────

    def load_cluster(
        self,
        cluster_id: int,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
        """
        mmap a whole cluster file and return nested dict:
            {expert_id: {layer_id: {param_name: Tensor}}}

        Tensors are materialized by safetensors (CPU, target dtype). We keep
        the logical layout compatible with the existing RAM buffer structure
        used by ExpertStore so the refactor is a drop-in replacement.
        """
        file_name = self.manifest.cluster_file(cluster_id)
        path = os.path.join(self.root_dir, file_name)
        nested: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        with _st_safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                parsed = _parse_key(key)
                if parsed is None:
                    continue
                layer_id, expert_id, param_name = parsed
                t = f.get_tensor(key)
                if dtype is not None and t.dtype != dtype:
                    t = t.to(dtype)
                nested.setdefault(expert_id, {}).setdefault(layer_id, {})[param_name] = t
        return nested

    def cluster_for_expert(self, expert_id: int) -> int:
        return self.manifest.expert_to_cluster[expert_id]

    def experts_in_cluster(self, cluster_id: int) -> List[int]:
        return list(self.manifest.clusters[cluster_id][1])

    # ── write path ───────────────────────────────────────────────

    @staticmethod
    def write_clusters(
        moe_layers,
        clusters: List[List[int]],
        output_dir: str,
        num_layers: int,
        num_experts: int,
        dtype: torch.dtype = torch.float16,
    ) -> str:
        """
        Serialize `moe_layers` (list of ChronosMOEFeedForward) into one
        safetensors file per cluster and write the manifest.

        Returns the path to cluster_manifest.json.
        """
        if not _HAS_SAFETENSORS:
            raise ImportError(
                "write_clusters requires the `safetensors` package. "
                "Install with `pip install safetensors>=0.4.0`."
            )
        os.makedirs(output_dir, exist_ok=True)

        manifest_clusters: Dict[int, Tuple[str, List[int]]] = {}
        expert_to_cluster: Dict[int, int] = {}

        for cid, cluster in enumerate(clusters):
            tensors: Dict[str, torch.Tensor] = {}
            for eid in cluster:
                expert_to_cluster[int(eid)] = cid
                for li in range(num_layers):
                    if li >= len(moe_layers):
                        continue
                    expert = moe_layers[li].experts[eid]
                    for pname, pval in expert.state_dict().items():
                        tensors[_key(li, int(eid), pname)] = pval.detach().to(dtype=dtype).cpu().contiguous()

            file_name = f"cluster_{cid}.ctsr"
            file_path = os.path.join(output_dir, file_name)
            _st_save_file(tensors, file_path)
            manifest_clusters[cid] = (file_name, [int(e) for e in cluster])

        manifest = ClusterManifest(
            version=MANIFEST_VERSION,
            num_experts=num_experts,
            num_layers=num_layers,
            storage_format="safetensors",
            clusters=manifest_clusters,
            expert_to_cluster=expert_to_cluster,
        )
        manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        return manifest_path

    @staticmethod
    def has_manifest(root_dir: str) -> bool:
        return os.path.exists(os.path.join(root_dir, MANIFEST_FILENAME))
