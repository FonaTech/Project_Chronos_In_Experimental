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
import threading
import collections
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np


class MLXExpertStore:
    """
    Expert weight store for MLX (unified memory — no H2D stream needed).

    Public API is intentionally similar to ExpertStore (PyTorch) so
    CacheManager can swap backends.
    """

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

        # LRU tracking which experts are "hot" (eval'd in Metal)
        capacity = max(1, int(config.vram_budget_gb * 1.5))  # rough heuristic
        self._hot_lru: collections.OrderedDict = collections.OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

        # Warm buffers: {expert_id: {layer_idx: {param_name: mx.array}}}
        self._warm: Dict[int, Dict[int, Dict[str, mx.array]]] = {}

    # ── SSD serialisation ─────────────────────────────────────────

    def offload_all_to_ssd(self):
        """Save all expert weights to SSD as .npy files."""
        for li, moe in enumerate(self.moe_layers):
            for ei, expert in enumerate(moe.experts):
                for name, param in vars(expert).items():
                    if isinstance(param, mx.array):
                        path = self._ssd_path(ei, li, name)
                        if not os.path.exists(path):
                            np.save(path, np.array(param))

    def _ssd_path(self, expert_id: int, layer_idx: int, param_name: str) -> str:
        return os.path.join(
            self.ssd_dir, f"expert_l{layer_idx}_e{expert_id}_{param_name}.npy"
        )

    def _load_from_ssd(self, expert_id: int) -> Dict[int, Dict[str, mx.array]]:
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

    # ── Prefetch (SSD → unified memory) ──────────────────────────

    def prefetch_to_ram(self, expert_ids: List[int]):
        """
        Load expert weights from SSD into mx.array (not yet eval'd).
        Call from a background thread for async prefetch.
        """
        for eid in expert_ids:
            with self._lock:
                if eid in self._warm:
                    continue
            layer_states = self._load_from_ssd(eid)
            if not layer_states:
                continue
            with self._lock:
                self._warm[eid] = layer_states

    # ── Promote to "hot" (materialise in Metal) ───────────────────

    def promote_to_vram(self, expert_id: int) -> bool:
        """
        Materialise expert weights in Metal shared memory via mx.eval().
        On MLX there is no actual copy — just lazy evaluation.
        """
        with self._lock:
            if expert_id not in self._warm:
                return False
            layer_states = self._warm[expert_id]

        # mx.eval() forces computation and pins arrays in Metal memory
        for li, params in layer_states.items():
            mx.eval(*params.values())
            # Write back into the live model
            expert = self.moe_layers[li].experts[expert_id]
            for fname, arr in params.items():
                parts = fname.split(".")
                obj = expert
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], arr)

        with self._lock:
            if expert_id in self._hot_lru:
                self._hot_lru.move_to_end(expert_id)
            else:
                if len(self._hot_lru) >= self._capacity:
                    self._hot_lru.popitem(last=False)
                self._hot_lru[expert_id] = True
        return True

    def sync_h2d(self):
        """No-op on MLX — Metal handles synchronisation internally."""
        pass

    # ── Availability mask ─────────────────────────────────────────

    def vram_availability_mask(self) -> mx.array:
        """[num_experts] float32 mask — 1.0 if expert is hot, 0.0 otherwise."""
        mask = np.zeros(self.num_experts, dtype=np.float32)
        with self._lock:
            for eid in self._hot_lru:
                if eid < self.num_experts:
                    mask[eid] = 1.0
        return mx.array(mask)

    def stats(self) -> dict:
        return {
            "backend": "mlx",
            "hot_experts": len(self._hot_lru),
            "hot_capacity": self._capacity,
            "warm_experts": len(self._warm),
            "h2d_stream": "metal_unified",
        }
