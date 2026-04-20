"""
chronos/io/expert_store.py

Three-tier expert weight storage: VRAM (hot) → RAM pinned (warm) → SSD (cold).

Responsibilities:
- Load expert weights from SSD into RAM pinned memory (async-friendly)
- Promote RAM-resident weights to VRAM on demand
- Evict VRAM weights via LRU when budget is exceeded
- Track per-expert residency state for soft-gating mask generation
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import contextlib
import os
import threading
import collections
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from chronos.io.storage import ClusterStorage, MANIFEST_FILENAME
from chronos.io.io_simulator import maybe_sleep as _maybe_sleep_ssd

try:
    from chronos.runtime.metrics import safe_record as _metric
except Exception:
    def _metric(name, value):  # fallback
        pass


class LRUCache:
    """Thread-safe LRU cache tracking which expert IDs are in a tier."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def contains(self, key: int) -> bool:
        with self._lock:
            return key in self._cache

    def touch(self, key: int):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)

    def put(self, key: int) -> Optional[int]:
        """Insert key; returns evicted key if capacity exceeded, else None."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return None
            evicted = None
            if len(self._cache) >= self.capacity:
                evicted, _ = self._cache.popitem(last=False)
            self._cache[key] = True
            return evicted

    def remove(self, key: int):
        with self._lock:
            self._cache.pop(key, None)

    def keys(self) -> List[int]:
        with self._lock:
            return list(self._cache.keys())

    def __len__(self):
        with self._lock:
            return len(self._cache)


class ExpertStore:
    """
    Manages three-tier storage for MoE expert weights.

    Tier 0 — VRAM: active experts as nn.Module parameters on GPU
    Tier 1 — RAM:  pinned-memory tensors ready for fast H2D transfer
    Tier 2 — SSD:  full expert weights saved as .pt files

    Usage:
        store = ExpertStore(model, config, ssd_dir="./expert_cache")
        store.offload_all_to_ssd()          # serialize experts to SSD
        store.prefetch_to_ram([0, 2])       # load experts 0,2 into RAM
        store.promote_to_vram(0)            # move expert 0 RAM→VRAM
        mask = store.vram_availability_mask()  # [num_experts] bool tensor
    """

    def __init__(self, model, config, ssd_dir: str = "./expert_cache"):
        self.config = config
        self.ssd_dir = ssd_dir
        self.device = next(model.parameters()).device
        os.makedirs(ssd_dir, exist_ok=True)

        # Dedicated CUDA stream for H2D weight transfers.
        # Runs concurrently with the default compute stream so weight copies
        # never stall matrix multiplications on the current token.
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self._h2d_stream: Optional[torch.cuda.Stream] = torch.cuda.Stream(device=self.device)
        else:
            self._h2d_stream = None

        # Collect all MoE layers and their expert modules
        from chronos.model.moe_chronos import ChronosMOEFeedForward
        self.moe_layers: List[ChronosMOEFeedForward] = [
            layer.mlp for layer in model.model.layers
            if isinstance(layer.mlp, ChronosMOEFeedForward)
        ]
        self.num_experts = config.num_experts
        self.num_layers = len(self.moe_layers)

        # Estimate VRAM capacity in number of experts
        expert_bytes = self._expert_size_bytes()
        vram_bytes = config.vram_budget_gb * (1024 ** 3)
        # Reserve 50% headroom for shared experts + attention KV cache
        self.vram_capacity = max(1, int(vram_bytes * 0.5 / max(expert_bytes, 1)))

        # Pinned RAM capacity: bounded by physical RAM safety limit
        # Never exceed pinned_memory_max_fraction of total physical RAM
        max_fraction = getattr(config, 'pinned_memory_max_fraction', 0.25)
        physical_ram_bytes = self._physical_ram_bytes()
        safe_pinned_bytes = physical_ram_bytes * max_fraction
        # Each expert occupies expert_bytes * num_layers in RAM
        per_expert_ram = max(expert_bytes * self.num_layers, 1)
        ram_capacity_by_safety = max(1, int(safe_pinned_bytes / per_expert_ram))
        # Also cap at 4x VRAM capacity (original heuristic)
        self.ram_capacity = min(self.vram_capacity * 4, ram_capacity_by_safety)

        self.vram_lru = LRUCache(capacity=self.vram_capacity)
        self.ram_lru = LRUCache(capacity=self.ram_capacity)

        # RAM pinned buffers: {expert_id: {layer_idx: tensor_dict}}
        self._ram_buffers: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._ram_lock = threading.Lock()

        # M3: per-expert H2D events. Every promote_to_vram records an event
        # on _h2d_stream and stores it here; wait_for_experts synchronizes
        # only on the needed ones so the compute stream never blocks on
        # unrelated transfers.
        self._pending_events: Dict[int, "torch.cuda.Event"] = {}
        self._events_lock = threading.Lock()

        # Storage backend: "safetensors" (clustered) or "pt" (legacy, one file per expert)
        self.storage_format = getattr(config, "storage_format", "safetensors")
        self._cluster_storage: Optional[ClusterStorage] = None
        if self.storage_format == "safetensors" and ClusterStorage.has_manifest(ssd_dir):
            # Manifest already exists on disk — load it up front so prefetch
            # can run cluster-aware from step 0.
            self._cluster_storage = ClusterStorage(ssd_dir)

    @staticmethod
    def _physical_ram_bytes() -> int:
        """Query available (not total) RAM at call time for dynamic safety."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            # Use available memory so we respect current system load
            return vm.available
        except ImportError:
            return 8 * (1024 ** 3)  # conservative 8 GB fallback

    def _safe_ram_capacity(self) -> int:
        """Recompute RAM capacity based on current available memory."""
        max_fraction = getattr(self.config, 'pinned_memory_max_fraction', 0.25)
        available_bytes = self._physical_ram_bytes()
        safe_bytes = available_bytes * max_fraction
        per_expert_ram = max(self._expert_size_bytes() * self.num_layers, 1)
        return max(1, min(self.vram_capacity * 4, int(safe_bytes / per_expert_ram)))

    def _expert_size_bytes(self) -> int:
        if not self.moe_layers:
            return 0
        expert = self.moe_layers[0].experts[0]
        return sum(p.numel() * 2 for p in expert.parameters())  # fp16

    def _ssd_path(self, expert_id: int, layer_idx: int) -> str:
        return os.path.join(self.ssd_dir, f"expert_l{layer_idx}_e{expert_id}.pt")

    # ── SSD operations ────────────────────────────────────────────

    def offload_all_to_ssd(self, clusters: Optional[List[List[int]]] = None):
        """
        Serialize all expert weights to SSD.

        - storage_format == "safetensors" (default): writes one .ctsr file per
          cluster + cluster_manifest.json. If `clusters` is None, falls back
          to a single all-experts cluster (still mmap-friendly, just no
          co-occurrence reordering benefit).
        - storage_format == "pt" (legacy): one .pt pickle per (expert, layer).
        """
        if self.storage_format == "safetensors":
            self._offload_safetensors(clusters)
        else:
            self._offload_legacy_pt()

    def _offload_legacy_pt(self):
        for li, moe in enumerate(self.moe_layers):
            for ei, expert in enumerate(moe.experts):
                path = self._ssd_path(ei, li)
                if not os.path.exists(path):
                    state = {k: v.half().cpu() for k, v in expert.state_dict().items()}
                    torch.save(state, path)

    def _offload_safetensors(self, clusters: Optional[List[List[int]]]):
        if clusters is None:
            # Trivial single-cluster layout — better than per-expert pickles
            # because we still get one mmap-friendly contiguous file.
            clusters = [list(range(self.num_experts))]
        ClusterStorage.write_clusters(
            moe_layers=self.moe_layers,
            clusters=clusters,
            output_dir=self.ssd_dir,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            dtype=torch.float16,
        )
        self._cluster_storage = ClusterStorage(self.ssd_dir)

    def attach_cluster_manifest(self, manifest_dir: Optional[str] = None):
        """Load (or reload) a cluster_manifest.json. Called by CacheManager
        after an offline repack, or to point ExpertStore at a different layout."""
        target = manifest_dir or self.ssd_dir
        if ClusterStorage.has_manifest(target):
            self._cluster_storage = ClusterStorage(target)
            self.storage_format = "safetensors"

    def _load_from_ssd(self, expert_id: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """Load expert weights for all layers from SSD into CPU tensors.

        Legacy .pt path. The clustered safetensors path uses
        `_load_cluster_into_ram` instead.
        """
        layer_states: Dict[int, Dict[str, torch.Tensor]] = {}
        for li in range(self.num_layers):
            path = self._ssd_path(expert_id, li)
            if os.path.exists(path):
                layer_states[li] = torch.load(path, map_location="cpu")
        return layer_states

    # ── RAM operations ────────────────────────────────────────────

    def prefetch_to_ram(self, expert_ids: List[int]):
        """
        Synchronously load expert weights from SSD into pinned RAM.
        Call this from a background thread for async prefetch.

        Cluster-aware path: when a safetensors manifest is attached, each
        request expands to its full Louvain cluster so co-occurring experts
        arrive in one sequential mmap — this is where we actually convert
        random reads into sequential reads. Requests that land in the same
        cluster are deduplicated.
        """
        if not expert_ids:
            return

        if self._cluster_storage is not None:
            self._prefetch_clustered(expert_ids)
        else:
            self._prefetch_legacy(expert_ids)

    def _prefetch_clustered(self, expert_ids: List[int]):
        """Group requests by cluster, load each cluster at most once."""
        cs = self._cluster_storage
        seen_clusters: set = set()
        clusters_to_load: List[int] = []
        for eid in expert_ids:
            if eid not in cs.manifest.expert_to_cluster:
                continue
            if self.ram_lru.contains(eid):
                self.ram_lru.touch(eid)
                continue
            cid = cs.cluster_for_expert(eid)
            if cid in seen_clusters:
                continue
            seen_clusters.add(cid)
            clusters_to_load.append(cid)

        for cid in clusters_to_load:
            # Simulate SSD read latency when CHRONOS_SIM_SSD_MS is set
            _maybe_sleep_ssd()
            # One mmap read pulls the whole cluster — all experts, all layers.
            nested = cs.load_cluster(cid, dtype=torch.float16)
            # nested: {expert_id: {layer_id: {param_name: Tensor}}}
            dynamic_cap = self._safe_ram_capacity()
            self.ram_lru.capacity = dynamic_cap
            for eid, layer_states in nested.items():
                if self.ram_lru.contains(eid):
                    self.ram_lru.touch(eid)
                    continue
                pinned = self._pin_layer_states(layer_states)
                with self._ram_lock:
                    evicted = self.ram_lru.put(eid)
                    if evicted is not None:
                        self._ram_buffers.pop(evicted, None)
                    self._ram_buffers[eid] = pinned

    def _prefetch_legacy(self, expert_ids: List[int]):
        for eid in expert_ids:
            if self.ram_lru.contains(eid):
                self.ram_lru.touch(eid)
                continue
            # Simulate SSD read latency
            _maybe_sleep_ssd()
            dynamic_cap = self._safe_ram_capacity()
            self.ram_lru.capacity = dynamic_cap

            layer_states = self._load_from_ssd(eid)
            if not layer_states:
                continue
            pinned = self._pin_layer_states(layer_states)
            with self._ram_lock:
                evicted = self.ram_lru.put(eid)
                if evicted is not None:
                    self._ram_buffers.pop(evicted, None)
                self._ram_buffers[eid] = pinned

    @staticmethod
    def _pin_layer_states(
        layer_states: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        pinned: Dict[int, Dict[str, torch.Tensor]] = {}
        for li, state in layer_states.items():
            pinned[li] = {}
            for k, v in state.items():
                try:
                    pinned[li][k] = v.pin_memory() if not v.is_pinned() else v
                except RuntimeError:
                    pinned[li][k] = v.cpu()
        return pinned

    # ── VRAM operations ───────────────────────────────────────────

    def promote_to_vram(self, expert_id: int, blocking: bool = False) -> bool:
        """
        Move expert from RAM → VRAM. Returns True if the H2D copy was issued.

        blocking=False (default, M3 path): issues H2D on _h2d_stream, records
        an event, and returns immediately. Caller must wait on the event (via
        ``wait_for_experts``) before the expert is used by compute.

        blocking=True (legacy path): synchronizes _h2d_stream before return,
        so the expert is guaranteed visible to the compute stream.
        """
        if not self.ram_lru.contains(expert_id):
            self.prefetch_to_ram([expert_id])

        with self._ram_lock:
            if expert_id not in self._ram_buffers:
                return False
            layer_states = self._ram_buffers[expert_id]

        evicted = self.vram_lru.put(expert_id)
        if evicted is not None:
            self._evict_from_vram(evicted)

        ctx = (
            torch.cuda.stream(self._h2d_stream)
            if self._h2d_stream is not None
            else _null_context()
        )
        with ctx:
            for li, state in layer_states.items():
                expert = self.moe_layers[li].experts[expert_id]
                expert.load_state_dict(
                    {k: v.to(self.device, non_blocking=True) for k, v in state.items()},
                    strict=False,
                )
            # Record an event on the H2D stream so consumers can wait on
            # *just this expert* instead of the whole stream.
            if self._h2d_stream is not None:
                evt = torch.cuda.Event(blocking=False)
                evt.record(self._h2d_stream)
                with self._events_lock:
                    self._pending_events[expert_id] = evt

        if blocking:
            self.sync_h2d()
        _metric("vram_experts", len(self.vram_lru))
        _metric("ram_experts", len(self.ram_lru))
        return True

    def wait_for_experts(self, expert_ids: List[int]) -> None:
        """Block the current (compute) stream until the named experts' H2D
        copies are done. No-op on CPU. Only syncs the relevant events; other
        in-flight transfers keep running."""
        if self._h2d_stream is None:
            return
        current = torch.cuda.current_stream(self.device)
        with self._events_lock:
            events = [self._pending_events.pop(eid, None) for eid in expert_ids]
        for evt in events:
            if evt is not None:
                # Make the compute stream wait for this event rather than
                # calling evt.synchronize() on the host, which would block
                # the CPU too.
                current.wait_event(evt)

    def sync_h2d(self):
        """Wait for *all* pending H2D weight transfers to complete (legacy)."""
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()
        with self._events_lock:
            self._pending_events.clear()

    def _evict_from_vram(self, expert_id: int):
        """Write VRAM expert back to RAM buffer before eviction."""
        for li, moe in enumerate(self.moe_layers):
            if expert_id < len(moe.experts):
                state = {k: v.half().cpu().pin_memory()
                         for k, v in moe.experts[expert_id].state_dict().items()}
                with self._ram_lock:
                    if expert_id not in self._ram_buffers:
                        self._ram_buffers[expert_id] = {}
                    self._ram_buffers[expert_id][li] = state
        self.vram_lru.remove(expert_id)

    # ── Mask generation ───────────────────────────────────────────

    def vram_availability_mask(self) -> torch.Tensor:
        """Returns [num_experts] bool tensor: True if expert is in VRAM."""
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        for eid in self.vram_lru.keys():
            if eid < self.num_experts:
                mask[eid] = True
        return mask

    def stats(self) -> dict:
        available_ram_gb = self._physical_ram_bytes() / (1024 ** 3)
        pinned_used_gb = (
            len(self.ram_lru) * self._expert_size_bytes() * self.num_layers / (1024 ** 3)
        )
        dynamic_cap = self._safe_ram_capacity()
        return {
            "vram_experts": len(self.vram_lru),
            "vram_capacity": self.vram_capacity,
            "ram_experts": len(self.ram_lru),
            "ram_capacity_dynamic": dynamic_cap,
            "expert_size_kb": self._expert_size_bytes() // 1024,
            "pinned_ram_used_gb": round(pinned_used_gb, 3),
            "available_ram_gb": round(available_ram_gb, 1),
            "pinned_ram_fraction": round(pinned_used_gb / max(available_ram_gb, 1), 4),
            "h2d_stream": "dedicated" if self._h2d_stream is not None else "default",
            "storage_format": self.storage_format,
            "cluster_aware": self._cluster_storage is not None,
            "num_clusters": (
                len(self._cluster_storage.manifest.clusters)
                if self._cluster_storage is not None else 0
            ),
        }


@contextlib.contextmanager
def _null_context():
    yield
