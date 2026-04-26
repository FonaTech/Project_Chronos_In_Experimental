"""
chronos/mlx/expert_store.py — Expert weight management for MLX (unified memory).

On Apple Silicon there is no VRAM / RAM split — all memory is unified.
"Prefetch" means loading from SSD into Python-side mx.array objects and
calling mx.eval() to materialise them in Metal shared memory.

Three logical tiers:
  Tier 0 (hot):  mx.array already eval'd (resident in Metal)
  Tier 1 (warm): mx.array allocated but not yet eval'd
  Tier 2 (cold): weights stored as .npy / .pt files on SSD
"""
import os
import json
import threading
import collections
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

from chronos.io.storage import MANIFEST_FILENAME, KEY_SEP, ClusterManifest

try:
    from safetensors import safe_open as _safe_open
    from safetensors.numpy import save_file as _np_save_file
    _HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover
    _HAS_SAFETENSORS = False


class MLXExpertStore:
    """
    Expert weight store for MLX (unified memory — no H2D stream needed).

    Public API is intentionally similar to ExpertStore (PyTorch) so
    CacheManager can swap backends.
    """

    _PARAM_NAMES = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")

    def __init__(self, model, config, ssd_dir: str = "./expert_cache_mlx"):
        self.config = config
        self.ssd_dir = ssd_dir
        self.num_experts = config.num_experts
        os.makedirs(ssd_dir, exist_ok=True)

        from chronos.mlx.moe import ChronosMLXMOE
        self.moe_layers: List[ChronosMLXMOE] = [
            layer.mlp for layer in model.layers
            if isinstance(layer.mlp, ChronosMLXMOE)
        ]
        self.num_layers = len(self.moe_layers)

        # LRU tracking which experts are "hot" (eval'd in Metal). On Apple
        # unified memory this is a residency budget, not real VRAM.
        requested = getattr(config, "recommended_resident_experts", None)
        if requested in (None, "", 0):
            requested = getattr(config, "num_experts_per_tok", 1)
        capacity = max(1, min(self.num_experts, int(requested)))
        self._hot_lru: collections.OrderedDict = collections.OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

        # Warm buffers: {expert_id: {layer_idx: {param_name: mx.array}}}
        self._warm: Dict[int, Dict[int, Dict[str, mx.array]]] = {}
        warm_requested = getattr(config, "recommended_ram_experts", None)
        if warm_requested in (None, "", 0):
            warm_requested = min(self.num_experts, max(capacity, capacity * 2))
        self._warm_capacity = max(capacity, min(self.num_experts, int(warm_requested)))
        self._warm_lru: collections.OrderedDict = collections.OrderedDict()
        self._cluster_manifest: ClusterManifest | None = None
        self._loaded_clusters: set[int] = set()
        self.storage_format = "npy"
        self.attach_cluster_manifest(ssd_dir)
        self._saved_live: Dict[int, Dict[int, Dict[str, mx.array]]] = {}
        self._stats = collections.Counter()

    # ── SSD serialisation ─────────────────────────────────────────

    def offload_all_to_ssd(self, clusters: Optional[List[List[int]]] = None):
        """Save all expert weights to SSD as .npy files."""
        if clusters is not None:
            self._write_clustered(clusters)
            return
        for li, moe in enumerate(self.moe_layers):
            for ei, expert in enumerate(moe.experts):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    arr = getattr(expert, proj).weight
                    path = self._ssd_path(ei, li, f"{proj}.weight")
                    if not os.path.exists(path):
                        np.save(path, np.array(arr))

    def _write_clustered(self, clusters: List[List[int]]):
        if not _HAS_SAFETENSORS:
            raise ImportError("MLX clustered expert cache requires safetensors.")
        os.makedirs(self.ssd_dir, exist_ok=True)
        expert_to_cluster: dict[int, int] = {}
        manifest_clusters: dict[int, tuple[str, list[int]]] = {}
        for cid, cluster in enumerate(clusters):
            tensors = {}
            for eid in cluster:
                expert_to_cluster[int(eid)] = cid
                for li, moe in enumerate(self.moe_layers):
                    expert = moe.experts[int(eid)]
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        key = f"l{li}_e{int(eid)}{KEY_SEP}{proj}.weight"
                        tensors[key] = np.array(getattr(expert, proj).weight)
            file_name = f"cluster_{cid}.ctsr"
            _np_save_file(tensors, os.path.join(self.ssd_dir, file_name))
            manifest_clusters[cid] = (file_name, [int(e) for e in cluster])
        manifest = ClusterManifest(
            version=1,
            num_experts=self.num_experts,
            num_layers=self.num_layers,
            storage_format="safetensors",
            clusters=manifest_clusters,
            expert_to_cluster=expert_to_cluster,
        )
        with open(os.path.join(self.ssd_dir, MANIFEST_FILENAME), "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        self.attach_cluster_manifest(self.ssd_dir)

    def _ssd_path(self, expert_id: int, layer_idx: int, param_name: str) -> str:
        return os.path.join(
            self.ssd_dir, f"expert_l{layer_idx}_e{expert_id}_{param_name}.npy"
        )

    def _load_from_ssd(self, expert_id: int) -> Dict[int, Dict[str, mx.array]]:
        if self._cluster_manifest is not None:
            self._load_cluster_for_expert(expert_id)
            with self._lock:
                return self._warm.get(expert_id, {})

        layer_states = {}
        for li in range(self.num_layers):
            params = {}
            for name in ("gate_proj", "up_proj", "down_proj"):
                for subname in ("weight",):
                    fname = f"{name}.{subname}"
                    path = self._ssd_path(expert_id, li, fname)
                    if os.path.exists(path):
                        params[fname] = mx.array(np.load(path))
            if params:
                layer_states[li] = params
        return layer_states

    def attach_cluster_manifest(self, manifest_dir: Optional[str] = None) -> bool:
        target = manifest_dir or self.ssd_dir
        manifest_path = os.path.join(target, MANIFEST_FILENAME)
        if not os.path.exists(manifest_path):
            return False
        if not _HAS_SAFETENSORS:
            return False
        with open(manifest_path, encoding="utf-8") as f:
            manifest = ClusterManifest.from_dict(json.load(f))
        if (
            int(getattr(manifest, "num_experts", -1)) != int(self.num_experts)
            or int(getattr(manifest, "num_layers", -1)) != int(self.num_layers)
        ):
            return False
        if set(manifest.expert_to_cluster) != set(range(int(self.num_experts))):
            return False
        for _cid, (file_name, experts) in manifest.clusters.items():
            if not os.path.exists(os.path.join(target, file_name)):
                return False
            if any(int(e) < 0 or int(e) >= int(self.num_experts) for e in experts):
                return False
        self._cluster_manifest = manifest
        self.ssd_dir = target
        self.storage_format = "safetensors"
        self._loaded_clusters.clear()
        return True

    def _parse_cluster_key(self, key: str):
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

    def _load_cluster_for_expert(self, expert_id: int) -> None:
        manifest = self._cluster_manifest
        if manifest is None or expert_id not in manifest.expert_to_cluster:
            return
        cid = manifest.expert_to_cluster[int(expert_id)]
        with self._lock:
            if (
                cid in self._loaded_clusters
                and expert_id in self._warm
                and self._layer_states_complete(self._warm.get(expert_id))
            ):
                return
        file_name = manifest.cluster_file(cid)
        path = os.path.join(self.ssd_dir, file_name)
        nested: Dict[int, Dict[int, Dict[str, mx.array]]] = {}
        with _safe_open(path, framework="mlx", device="cpu") as f:
            for key in f.keys():
                parsed = self._parse_cluster_key(key)
                if parsed is None:
                    continue
                li, eid, pname = parsed
                nested.setdefault(eid, {}).setdefault(li, {})[pname] = f.get_tensor(key)
        ordered = sorted(nested.items(), key=lambda item: int(item[0]) == int(expert_id))
        with self._lock:
            for eid, layer_states in ordered:
                self._put_warm_locked(eid, layer_states)
            self._loaded_clusters.add(cid)

    # ── Prefetch (SSD → unified memory) ──────────────────────────

    def prefetch_to_ram(self, expert_ids: List[int]):
        """
        Load expert weights from SSD into mx.array (not yet eval'd).
        Call from a background thread for async prefetch.
        """
        for eid in expert_ids:
            with self._lock:
                if eid in self._warm and self._layer_states_complete(self._warm.get(int(eid))):
                    self._touch_warm_locked(int(eid))
                    continue
            layer_states = self._complete_or_saved_state(int(eid), self._load_from_ssd(eid))
            if not layer_states:
                self._stats["prefetch_misses"] += 1
                continue
            with self._lock:
                self._put_warm_locked(int(eid), layer_states)
                self._stats["prefetch_loads"] += 1

    def _touch_warm_locked(self, expert_id: int):
        if expert_id in self._warm_lru:
            self._warm_lru.move_to_end(expert_id)

    def _put_warm_locked(self, expert_id: int, layer_states: Dict[int, Dict[str, mx.array]]):
        self._warm[expert_id] = layer_states
        self._warm_lru[expert_id] = True
        self._warm_lru.move_to_end(expert_id)
        self._evict_warm_locked()

    def _evict_warm_locked(self):
        while len(self._warm_lru) > self._warm_capacity:
            evict_id = None
            for candidate in self._warm_lru.keys():
                if candidate not in self._hot_lru:
                    evict_id = candidate
                    break
            if evict_id is None:
                return
            self._warm_lru.pop(evict_id, None)
            self._warm.pop(evict_id, None)
            self._replace_expert_with_placeholder(evict_id)

    def _replace_expert_with_placeholder(self, expert_id: int):
        from chronos.mlx.moe import LazyFeedForwardMLX

        for moe in self.moe_layers:
            if 0 <= expert_id < len(moe.experts):
                moe.experts[expert_id] = LazyFeedForwardMLX()

    def _is_placeholder(self, expert) -> bool:
        from chronos.mlx.moe import LazyFeedForwardMLX

        return isinstance(expert, LazyFeedForwardMLX)

    def _layer_states_complete(self, layer_states: Dict[int, Dict[str, mx.array]] | None) -> bool:
        if not isinstance(layer_states, dict):
            return False
        for li in range(int(self.num_layers)):
            params = layer_states.get(li)
            if not isinstance(params, dict):
                return False
            if any(name not in params for name in self._PARAM_NAMES):
                return False
        return True

    def _complete_or_saved_state(
        self,
        expert_id: int,
        layer_states: Dict[int, Dict[str, mx.array]] | None,
    ) -> Dict[int, Dict[str, mx.array]]:
        if self._layer_states_complete(layer_states):
            return layer_states or {}
        saved = self._saved_live.get(int(expert_id), {})
        if self._layer_states_complete(saved):
            self._stats["saved_live_repairs"] += 1
            return saved
        return {}

    def _expert_live_all_layers(self, expert_id: int) -> bool:
        if int(expert_id) < 0 or int(expert_id) >= int(self.num_experts):
            return False
        for moe in self.moe_layers:
            if int(expert_id) >= len(moe.experts):
                return False
            expert = moe.experts[int(expert_id)]
            if self._is_placeholder(expert):
                return False
            try:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    getattr(expert, proj).weight
            except Exception:
                return False
        return True

    def hot_expert_ids(self) -> set[int]:
        """Return only experts that are hot and materialized in every MoE layer."""
        with self._lock:
            stale = [
                int(eid) for eid in self._hot_lru.keys()
                if not self._expert_live_all_layers(int(eid))
            ]
            for eid in stale:
                self._hot_lru.pop(eid, None)
            if stale:
                self._stats["stale_hot_drops"] += len(stale)
            return {
                int(eid) for eid in self._hot_lru.keys()
                if int(eid) < int(self.num_experts)
            }

    # ── Promote to "hot" (materialise in Metal) ───────────────────

    def promote_to_vram(self, expert_id: int) -> bool:
        """
        Materialise expert weights in Metal shared memory via mx.eval().
        On MLX there is no actual copy — just lazy evaluation.
        """
        expert_id = int(expert_id)
        if expert_id < 0 or expert_id >= int(self.num_experts):
            return False

        with self._lock:
            if expert_id in self._hot_lru:
                if self._expert_live_all_layers(expert_id):
                    self._hot_lru.move_to_end(expert_id)
                    self._stats["hot_hits"] += 1
                    return True
                self._hot_lru.pop(expert_id, None)
                self._stats["stale_hot_repairs"] += 1
            layer_states = self._warm.get(expert_id)
            if layer_states is not None:
                self._touch_warm_locked(expert_id)

        if not self._layer_states_complete(layer_states):
            self.prefetch_to_ram([expert_id])
            with self._lock:
                layer_states = self._warm.get(expert_id)
                if layer_states is not None:
                    self._touch_warm_locked(expert_id)
        layer_states = self._complete_or_saved_state(expert_id, layer_states)
        if not layer_states:
            self._stats["promote_misses"] += 1
            return False

        # mx.eval() forces computation and pins arrays in Metal memory
        from chronos.mlx.moe import FeedForwardMLX, LazyFeedForwardMLX

        hidden = int(getattr(self.config, "hidden_size"))
        intermediate = int(getattr(self.config, "moe_intermediate_size"))
        for li in range(int(self.num_layers)):
            params = layer_states[li]
            mx.eval(*(params[name] for name in self._PARAM_NAMES))
            # Write back into the live model
            expert = self.moe_layers[li].experts[expert_id]
            if isinstance(expert, LazyFeedForwardMLX):
                expert = FeedForwardMLX(hidden, intermediate)
                self.moe_layers[li].experts[expert_id] = expert
            for fname in self._PARAM_NAMES:
                proj_name = fname.rsplit(".", 1)[0]
                getattr(expert, proj_name).weight = params[fname]

        if not self._expert_live_all_layers(expert_id):
            with self._lock:
                self._hot_lru.pop(expert_id, None)
                self._stats["promote_incomplete"] += 1
            return False

        with self._lock:
            if expert_id in self._hot_lru:
                self._hot_lru.move_to_end(expert_id)
            else:
                while len(self._hot_lru) >= self._capacity:
                    evicted, _ = self._hot_lru.popitem(last=False)
                    if evicted == expert_id:
                        continue
                    # Keep warm weights in unified memory, but drop the
                    # executable live module so masked MoE cannot call an
                    # expert outside the hot execution budget.
                    self._replace_expert_with_placeholder(evicted)
                    self._stats["hot_evictions"] += 1
                self._hot_lru[expert_id] = True
            self._stats["promotions"] += 1
        return True

    def replace_live_experts_with_placeholders(self):
        from chronos.mlx.moe import LazyFeedForwardMLX

        for li, moe in enumerate(self.moe_layers):
            for ei, expert in enumerate(moe.experts):
                state = {}
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    state[f"{proj}.weight"] = getattr(expert, proj).weight
                self._saved_live.setdefault(ei, {})[li] = state
                moe.experts[ei] = LazyFeedForwardMLX()
        with self._lock:
            self._hot_lru.clear()

    def restore_live_experts_from_saved(self):
        from chronos.mlx.moe import FeedForwardMLX, LazyFeedForwardMLX

        hidden = int(getattr(self.config, "hidden_size", 0) or 0)
        intermediate = int(getattr(self.config, "moe_intermediate_size", 0) or 0)
        for eid, layers in self._saved_live.items():
            for li, state in layers.items():
                expert = self.moe_layers[li].experts[eid]
                if isinstance(expert, LazyFeedForwardMLX):
                    expert = FeedForwardMLX(hidden, intermediate)
                    self.moe_layers[li].experts[eid] = expert
                for fname, arr in state.items():
                    proj_name = fname.rsplit(".", 1)[0]
                    getattr(expert, proj_name).weight = arr

    def sync_h2d(self):
        """No-op on MLX — Metal handles synchronisation internally."""
        pass

    # ── Availability mask ─────────────────────────────────────────

    def vram_availability_mask(self) -> mx.array:
        """[num_experts] float32 mask — 1.0 if expert is hot, 0.0 otherwise."""
        mask = np.zeros(self.num_experts, dtype=np.float32)
        for eid in self.hot_expert_ids():
            mask[int(eid)] = 1.0
        return mx.array(mask)

    def warm_up(self, expert_ids: Optional[List[int]] = None):
        if expert_ids is None:
            expert_ids = list(range(min(self.num_experts, self._capacity)))
        self.prefetch_to_ram(expert_ids)
        for eid in expert_ids:
            self.promote_to_vram(int(eid))

    def stats(self) -> dict:
        hot_ids = self.hot_expert_ids()
        return {
            "backend": "mlx",
            "hot_experts": len(hot_ids),
            "hot_capacity": self._capacity,
            "warm_experts": len(self._warm),
            "warm_capacity": self._warm_capacity,
            "storage_format": self.storage_format,
            "cluster_aware": self._cluster_manifest is not None,
            "num_clusters": len(self._cluster_manifest.clusters) if self._cluster_manifest else 0,
            "h2d_stream": "metal_unified",
            **{k: int(v) for k, v in self._stats.items()},
        }
