"""
chronos/runtime/inference_engine.py

End-to-end Chronos inference engine with:
- Async prefetch driven by LookaheadRouter
- Soft gating fallback via CacheManager availability mask
- Token/s throughput measurement
- Optional streamer support

Usage:
    engine = ChronosInferenceEngine(model, config, ssd_dir="./expert_cache")
    engine.setup()  # offload experts to SSD, warm up cache
    output_ids = engine.generate(input_ids, max_new_tokens=200)
    print(engine.last_stats)
"""
import sys
import chronos.deps  # ensure minimind on sys.path

import time
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.moe_chronos import ChronosMOEFeedForward
from chronos.runtime.cache_manager import CacheManager
from chronos.router.prefill_scheduler import PrefillScheduler


def _rss_gb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def _backend_memory_snapshot(device) -> dict:
    dtype = str(device).split(":", 1)[0]
    try:
        from chronos.backend.mac_diagnostics import mps_memory_snapshot, mlx_memory_snapshot
    except Exception:
        return {}
    if dtype == "mps":
        return mps_memory_snapshot()
    if dtype == "mlx":
        return mlx_memory_snapshot()
    return {}


def _backend_memory_fields(snapshot: dict, suffix: str) -> dict:
    out = {}
    for key in (
        "mps_allocated_gb",
        "mps_driver_allocated_gb",
        "mlx_active_gb",
        "mlx_cache_gb",
        "mlx_peak_gb",
    ):
        if key in snapshot:
            prefix = key[:-3] if key.endswith("_gb") else key
            out[f"{prefix}_{suffix}_gb"] = round(float(snapshot[key]), 6)
    return out


class ChronosInferenceEngine:
    """
    Wraps ChronosForCausalLM with the full async prefetch + soft gating pipeline.

    The decode loop:
    1. Get availability mask from CacheManager
    2. Forward pass with mask → soft gating handles cache misses
    3. Extract LookaheadRouter probs from output
    4. Submit future expert predictions to prefetcher (non-blocking)
    5. Repeat
    """

    def __init__(
        self,
        model: ChronosForCausalLM,
        config: ChronosConfig,
        ssd_dir: str = "./expert_cache",
    ):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.cache_manager = CacheManager(model, config, ssd_dir=ssd_dir)
        self.last_stats: dict = {}
        self._runtime_stats: dict = {}
        self._route_cache_scores: Dict[int, float] = {}
        self._last_prefill_ram_expert_ids: List[int] = []

    def _moe_layers(self) -> List[ChronosMOEFeedForward]:
        return [
            l.mlp for l in self.model.model.layers
            if isinstance(l.mlp, ChronosMOEFeedForward)
        ]

    @staticmethod
    def _normalize_miss_policy(miss_policy: str | None) -> str:
        policy = (miss_policy or "on_demand").strip().lower().replace("-", "_")
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
        return policy if policy in {"on_demand", "sync_on_demand", "fallback_diagnostic"} else "on_demand"

    def _install_moe_runtime_hooks(self, miss_policy: str):
        store = self.cache_manager.expert_store

        def load_one(expert_id: int) -> dict:
            expert_id = int(expert_id)
            already_vram = store.vram_lru.contains(expert_id)
            already_ram = store.ram_lru.contains(expert_id)
            t0 = time.monotonic()
            ok = False
            try:
                if already_vram or already_ram or miss_policy == "sync_on_demand":
                    promoted = self.cache_manager.ensure_resident([expert_id])
                    ok = store.vram_lru.contains(expert_id)
                else:
                    # README architecture: cold SSD misses must not enter the
                    # decode critical path. Queue the exact missing expert for
                    # RAM prefetch and let shared fallback cover this token.
                    self.cache_manager.prefetch_experts_to_ram([expert_id])
                    promoted = []
                    ok = False
            finally:
                elapsed = time.monotonic() - t0
                self._runtime_stats["on_demand_loads"] += 0 if already_vram else 1
                self._runtime_stats["on_demand_load_time_s"] += elapsed
                if already_vram:
                    self._runtime_stats["resident_vram_hits"] += 1
                elif already_ram:
                    self._runtime_stats["resident_ram_hits"] += 1
                else:
                    if miss_policy == "on_demand":
                        self._runtime_stats["async_cold_miss_prefetches"] = (
                            int(self._runtime_stats.get("async_cold_miss_prefetches", 0)) + 1
                        )
                    else:
                        self._runtime_stats["sync_ssd_loads"] = (
                            int(self._runtime_stats.get("sync_ssd_loads", 0)) + 1
                        )
                if ok:
                    self._runtime_stats["on_demand_loaded_experts"].append(expert_id)
                    if promoted:
                        self._runtime_stats["on_demand_promoted_experts"].extend(promoted)
            return {"ok": ok, "already_vram": already_vram, "already_ram": already_ram}

        def touch_one(expert_id: int) -> None:
            store.touch_expert(int(expert_id))

        for moe in self._moe_layers():
            moe.runtime_miss_policy = miss_policy
            moe.runtime_on_demand_loader = load_one if miss_policy in {"on_demand", "sync_on_demand"} else None
            moe.runtime_touch_expert = touch_one

    def _clear_moe_runtime_hooks(self):
        for moe in self._moe_layers():
            moe.runtime_miss_policy = "fallback_diagnostic"
            moe.runtime_on_demand_loader = None
            moe.runtime_touch_expert = None

    def _consume_moe_runtime_stats(self) -> dict:
        out = {
            "on_demand_experts": [],
            "fallback_experts": [],
            "on_demand_weight_mass": 0.0,
            "fallback_executed_weight_mass": 0.0,
        }
        for moe in self._moe_layers():
            out["on_demand_experts"].extend(getattr(moe, "last_on_demand_experts", []) or [])
            out["fallback_experts"].extend(getattr(moe, "last_fallback_experts", []) or [])
            out["on_demand_weight_mass"] += float(getattr(moe, "last_on_demand_weight_mass", 0.0) or 0.0)
            out["fallback_executed_weight_mass"] += float(getattr(moe, "last_fallback_weight_mass", 0.0) or 0.0)
        return out

    def _recent_route_scores(
        self,
        current_ids: List[int],
        predicted_ids: List[int],
        ram_predicted_ids: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """Decayed route scores used as a sticky eviction priority.

        Higher scores stay hotter in both RAM and VRAM. This intentionally
        spans tokens; replacing it with only the current step turns tiny
        budgets into a promote/evict loop.
        """
        decayed = {
            int(eid): float(score) * 0.85
            for eid, score in self._route_cache_scores.items()
            if float(score) * 0.85 >= 0.05
        }
        scores = decayed
        for eid in current_ids:
            scores[int(eid)] = scores.get(int(eid), 0.0) + 3.0
        for eid in predicted_ids:
            scores[int(eid)] = scores.get(int(eid), 0.0) + 2.0
        for eid in ram_predicted_ids or []:
            scores[int(eid)] = scores.get(int(eid), 0.0) + 1.0
        self._route_cache_scores = scores
        return scores

    def _router_topk(self, probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the MoE top-k ids/weights using the model's routing policy."""
        top_k = int(getattr(self.config, "num_experts_per_tok", 1) or 1)
        top_k = max(1, min(top_k, probs.shape[-1]))
        topk_weight, topk_idx = torch.topk(probs, k=top_k, dim=-1, sorted=False)
        if bool(getattr(self.config, "norm_topk_prob", False)):
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weight, topk_idx

    @staticmethod
    def _ordered_ids_from_scores(scores: torch.Tensor, capacity: int) -> List[int]:
        if scores.numel() == 0 or capacity <= 0 or not torch.any(scores > 0):
            return []
        ordered = torch.argsort(scores, descending=True).tolist()
        return [int(eid) for eid in ordered if float(scores[eid].item()) > 0.0][:capacity]

    def _ordered_sticky_ids(self, capacity: int) -> List[int]:
        if capacity <= 0 or not self._route_cache_scores:
            return []
        return [
            int(eid)
            for eid, _score in sorted(
                self._route_cache_scores.items(),
                key=lambda item: (-float(item[1]), int(item[0])),
            )
        ][:capacity]

    def _promotion_ready_ids(self, expert_ids: List[int]) -> List[int]:
        """Only return IDs already staged in RAM/VRAM.

        Calling ensure_resident on a cold SSD expert during decode would
        synchronously read from disk. The predictive path keeps SSD IO in
        the background and promotes only warm experts.
        """
        store = self.cache_manager.expert_store
        ready = []
        for eid in dict.fromkeys(int(e) for e in expert_ids or []):
            if store.vram_lru.contains(eid) or store.ram_lru.contains(eid):
                ready.append(eid)
        return ready

    def setup(self, warm_expert_ids: Optional[List[int]] = None):
        """
        One-time setup: serialize experts to SSD, start prefetcher, warm cache.
        Call once before the first generate().
        """
        print("[Chronos] Offloading expert weights to SSD...")
        self.cache_manager.expert_store.offload_all_to_ssd()
        self.cache_manager.start()
        print("[Chronos] Warming up VRAM cache...")
        self.cache_manager.warm_up(warm_expert_ids)
        print(f"[Chronos] Ready. Cache stats: {self.cache_manager.stats()}")

    def setup_from_state_dict(
        self,
        state_dict: dict,
        warm_expert_ids: Optional[List[int]] = None,
    ):
        """Setup path for real lazy expert loading on CPU/MPS.

        The caller loads base/non-expert weights into the live model, then
        hands the full checkpoint here so ExpertStore can shard only expert
        tensors to SSD and replace live experts with placeholders.
        """
        print("[Chronos] Offloading expert weights to SSD from checkpoint...")
        self.cache_manager.expert_store.offload_from_state_dict(state_dict)
        self.cache_manager.start()
        print("[Chronos] Warming up VRAM cache...")
        self.cache_manager.warm_up(warm_expert_ids)
        print(f"[Chronos] Ready. Cache stats: {self.cache_manager.stats()}")

    def setup_from_state_reader(
        self,
        reader,
        warm_expert_ids: Optional[List[int]] = None,
    ):
        """Setup lazy expert loading from a safetensors/GGUF export reader."""
        print("[Chronos] Offloading expert weights to SSD from export reader...")
        self.cache_manager.expert_store.offload_from_state_reader(reader)
        self.cache_manager.start()
        print("[Chronos] Warming up VRAM cache...")
        self.cache_manager.warm_up(warm_expert_ids)
        print(f"[Chronos] Ready. Cache stats: {self.cache_manager.stats()}")

    def teardown(self):
        self.cache_manager.stop()

    def _get_current_expert_ids(self, capacity: Optional[int] = None) -> List[int]:
        """Extract active top-k expert IDs from every MoE layer.

        Older code only looked at layer 0, which made the lazy path report
        false cache hits whenever later layers routed to cold experts.
        """
        moe_layers = self._moe_layers()
        num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        if not moe_layers or num_experts <= 0:
            return []
        scores = torch.zeros(num_experts, dtype=torch.float32)
        for moe in moe_layers:
            if moe.last_router_probs is None:
                continue
            probs = moe.last_router_probs[:, -1, :].detach().cpu()  # [B, E]
            weights, ids = self._router_topk(probs)
            scores.index_add_(0, ids.reshape(-1), weights.reshape(-1).to(scores.dtype))
        if capacity is None:
            capacity = num_experts
        return self._ordered_ids_from_scores(scores, min(int(capacity), num_experts))

    def _predict_expert_ids(
        self,
        lookahead_probs: Optional[torch.Tensor],
        future_offset: int = 1,
        capacity: Optional[int] = None,
        all_future: bool = False,
    ) -> List[int]:
        """Select globally hot expert IDs from LookaheadRouter predictions."""
        if lookahead_probs is None:
            return []
        probs = lookahead_probs.detach().cpu()
        if probs.dim() != 4 or probs.shape[-1] == 0:
            return []
        num_experts = int(probs.shape[-1])
        scores = torch.zeros(num_experts, dtype=torch.float32)
        if all_future:
            if probs.shape[2] <= 1:
                return []
            future = probs[:, -1, 1:, :]  # [B, K, E]
            K = future.shape[1]
            for offset in range(K):
                step_probs = future[:, offset, :]
                weights, ids = self._router_topk(step_probs)
                weight_scale = 1.0 / float(offset + 1)
                scores.index_add_(
                    0,
                    ids.reshape(-1),
                    weights.reshape(-1).to(scores.dtype) * weight_scale,
                )
        else:
            if probs.shape[2] <= future_offset:
                return []
            step_probs = probs[:, -1, future_offset, :]  # [B, E]
            weights, ids = self._router_topk(step_probs)
            scores.index_add_(0, ids.reshape(-1), weights.reshape(-1).to(scores.dtype))
        if capacity is None:
            capacity = num_experts
        return self._ordered_ids_from_scores(scores, min(int(capacity), num_experts))

    def _route_cache_stats(
        self,
        masks: Optional[List[torch.Tensor]],
        *,
        include_all_tokens: bool,
    ) -> Dict[str, object]:
        """Measure real per-layer top-k cache coverage for the last forward."""
        moe_layers = self._moe_layers()
        totals = {
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_weight_mass": 0.0,
            "fallback_weight_mass": 0.0,
            "total_route_weight_mass": 0.0,
            "activated_expert_ids": [],
            "per_layer": [],
        }
        if not moe_layers:
            return totals

        activated: set[int] = set()
        for layer_idx, moe in enumerate(moe_layers):
            if moe.last_router_probs is None:
                continue
            probs = moe.last_router_probs.detach().cpu()
            if not include_all_tokens:
                probs = probs[:, -1:, :]
            flat_probs = probs.reshape(-1, probs.shape[-1])
            weights, ids = self._router_topk(flat_probs)
            flat_ids = ids.reshape(-1).to(torch.long)
            flat_weights = weights.reshape(-1).to(torch.float32)
            if flat_ids.numel() == 0:
                continue

            if masks is None:
                available = torch.ones(probs.shape[-1], dtype=torch.bool)
            elif len(masks) == 0:
                available = torch.zeros(probs.shape[-1], dtype=torch.bool)
            else:
                raw_mask = masks[min(layer_idx, len(masks) - 1)].detach().cpu()
                available = raw_mask.to(dtype=torch.bool)
            if available.numel() < probs.shape[-1]:
                padded = torch.zeros(probs.shape[-1], dtype=torch.bool)
                padded[:available.numel()] = available
                available = padded

            hit_mask = available[flat_ids]
            hits = int(hit_mask.sum().item())
            total = int(flat_ids.numel())
            misses = total - hits
            hit_mass = float(flat_weights[hit_mask].sum().item())
            miss_mass = float(flat_weights[~hit_mask].sum().item())
            total_mass = hit_mass + miss_mass

            activated.update(int(eid) for eid in flat_ids.unique().tolist())
            totals["cache_hits"] += hits
            totals["cache_misses"] += misses
            totals["hit_weight_mass"] += hit_mass
            totals["fallback_weight_mass"] += miss_mass
            totals["total_route_weight_mass"] += total_mass
            totals["per_layer"].append({
                "layer": layer_idx,
                "hit_rate": round(hits / max(total, 1), 4),
                "fallback_weight_rate": round(miss_mass / max(total_mass, 1e-12), 4),
                "hits": hits,
                "misses": misses,
                "activated_experts": sorted(int(eid) for eid in flat_ids.unique().tolist()),
                "loaded_experts": sorted(int(eid) for eid in available.nonzero().flatten().tolist()),
            })

        totals["activated_expert_ids"] = sorted(activated)
        return totals

    def _plan_prefill_experts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[List[int], List[int]]:
        """Probe prompt routing with shared fallback, then load hot experts.

        The live model contains placeholder experts after lazy setup. A cold
        first prefill would therefore run entirely through shared fallback and
        can diverge from the real model before decode starts. This probe pass
        does not materialize expert modules; it only records gate choices, then
        returns the most frequent expert IDs bounded by the active cache budget.
        The caller reruns the real prefill after those experts are promoted.
        """
        if not getattr(self.config, "use_moe", True):
            return [], []
        num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        if num_experts <= 0:
            return [], []

        cold_mask = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
        masks = [cold_mask] * len(self.cache_manager.expert_store.moe_layers)
        saved_hooks = [
            (
                moe.runtime_miss_policy,
                moe.runtime_on_demand_loader,
                moe.runtime_touch_expert,
            )
            for moe in self._moe_layers()
        ]
        try:
            for moe in self._moe_layers():
                moe.runtime_miss_policy = "fallback_diagnostic"
                moe.runtime_on_demand_loader = None
                moe.runtime_touch_expert = None
            probe_start = {
                "on_demand_loads": int(self._runtime_stats.get("on_demand_loads", 0)),
                "on_demand_load_time_s": float(self._runtime_stats.get("on_demand_load_time_s", 0.0)),
                "sync_ssd_loads": int(self._runtime_stats.get("sync_ssd_loads", 0)),
                "resident_vram_hits": int(self._runtime_stats.get("resident_vram_hits", 0)),
                "resident_ram_hits": int(self._runtime_stats.get("resident_ram_hits", 0)),
                "loaded_len": len(self._runtime_stats.get("on_demand_loaded_experts", [])),
                "promoted_len": len(self._runtime_stats.get("on_demand_promoted_experts", [])),
            }
            self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                available_expert_masks=masks,
            )
            self._runtime_stats["on_demand_loads"] = probe_start["on_demand_loads"]
            self._runtime_stats["on_demand_load_time_s"] = probe_start["on_demand_load_time_s"]
            self._runtime_stats["sync_ssd_loads"] = probe_start["sync_ssd_loads"]
            self._runtime_stats["resident_vram_hits"] = probe_start["resident_vram_hits"]
            self._runtime_stats["resident_ram_hits"] = probe_start["resident_ram_hits"]
            self._runtime_stats["on_demand_loaded_experts"] = self._runtime_stats["on_demand_loaded_experts"][:probe_start["loaded_len"]]
            self._runtime_stats["on_demand_promoted_experts"] = self._runtime_stats["on_demand_promoted_experts"][:probe_start["promoted_len"]]
        finally:
            for moe, saved in zip(self._moe_layers(), saved_hooks):
                moe.runtime_miss_policy, moe.runtime_on_demand_loader, moe.runtime_touch_expert = saved

        scores = torch.zeros(num_experts, dtype=torch.float32)
        for layer in self.model.model.layers:
            moe = getattr(layer, "mlp", None)
            if not isinstance(moe, ChronosMOEFeedForward) or moe.last_router_probs is None:
                continue
            probs = moe.last_router_probs.reshape(-1, num_experts)
            weights, expert_ids = self._router_topk(probs.detach().cpu())
            scores.index_add_(
                0,
                expert_ids.reshape(-1).to(torch.long),
                weights.reshape(-1).to(scores.dtype),
            )

        if not torch.any(scores > 0):
            return [], []
        vram_capacity = max(1, min(
            int(getattr(self.cache_manager.expert_store, "vram_capacity", 1) or 1),
            num_experts,
        ))
        ram_capacity = max(vram_capacity, min(
            int(getattr(self.cache_manager.expert_store, "ram_capacity", vram_capacity) or vram_capacity),
            num_experts,
        ))
        ram_ids = self._ordered_ids_from_scores(scores, ram_capacity)
        return ram_ids[:vram_capacity], ram_ids

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.85,
        top_p: float = 0.85,
        top_k: int = 50,
        eos_token_id: int = 2,
        streamer=None,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        scheduler: Optional[PrefillScheduler] = None,
        miss_policy: str = "on_demand",
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        miss_policy = self._normalize_miss_policy(miss_policy)
        self._route_cache_scores = {}
        self._last_prefill_ram_expert_ids = []
        self._runtime_stats = {
            "on_demand_loads": 0,
            "on_demand_load_time_s": 0.0,
            "sync_ssd_loads": 0,
            "async_cold_miss_prefetches": 0,
            "resident_vram_hits": 0,
            "resident_ram_hits": 0,
            "on_demand_loaded_experts": [],
            "on_demand_promoted_experts": [],
        }
        self._install_moe_runtime_hooks(miss_policy)

        try:
            # ── Prefill phase: front-load expert IO before generation ──────
            prefill_t0 = time.monotonic()
            prefill_mem0 = _rss_gb()
            prefill_backend_mem0 = _backend_memory_snapshot(self.device)
            if scheduler is not None:
                scheduler.prepare(input_ids, device=str(self.device))
                scheduler.wait()  # block until SSD→VRAM load complete
                prefill_expert_ids: List[int] = []
                prefill_ram_expert_ids: List[int] = []
            else:
                prefill_expert_ids, prefill_ram_expert_ids = self._plan_prefill_experts(input_ids, attention_mask)
                self._last_prefill_ram_expert_ids = prefill_ram_expert_ids
                if prefill_ram_expert_ids:
                    self.cache_manager.expert_store.prefetch_to_ram(prefill_ram_expert_ids)
                if prefill_expert_ids:
                    self.cache_manager.ensure_resident(prefill_expert_ids)

            past_key_values = None
            finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
            if streamer:
                streamer.put(input_ids.cpu())

            tokens_generated = 0
            cache_hits, cache_misses = 0, 0
            activated_expert_ids: List[int] = []
            submitted_prefetch_ids: List[int] = []
            promoted_expert_ids: List[int] = []
            predicted_current_expert_ids: List[int] = []
            on_demand_expert_ids: List[int] = []
            fallback_expert_ids: List[int] = []
            route_layer_stats: List[dict] = []
            hit_weight_mass = 0.0
            fallback_weight_mass = 0.0
            total_route_weight_mass = 0.0
            on_demand_weight_mass = 0.0
            fallback_executed_weight_mass = 0.0
            prediction_hits = 0
            prediction_total = 0
            # M3: holds last step's lookahead predictions; we submit them as
            # prefetch BEFORE this step's forward so the H2D stream overlaps
            # with compute. None on the first iteration (no predictions yet).
            prev_lookahead_probs = None
            prev_current_ids: List[int] = []
            prefill_time_s = 0.0
            prefill_mem1 = prefill_mem0
            prefill_backend_mem1 = prefill_backend_mem0
            prefill_done_t = prefill_t0

            for step in range(max_new_tokens):
            # ── M3 pipeline: fire async prefetch BEFORE forward ──
                predicted_ids: List[int] = []
                ram_predicted_ids: List[int] = []
                if scheduler is None:
                    if prev_lookahead_probs is not None:
                        store = self.cache_manager.expert_store
                        vram_capacity = int(getattr(store, "vram_capacity", 1) or 1)
                        ram_capacity = int(getattr(store, "ram_capacity", vram_capacity) or vram_capacity)
                        ram_predicted_ids = self._predict_expert_ids(
                            prev_lookahead_probs,
                            capacity=ram_capacity,
                            all_future=True,
                        )
                        predicted_ids = self._predict_expert_ids(
                            prev_lookahead_probs,
                            future_offset=1,
                            capacity=vram_capacity,
                        )
                        predicted_current_expert_ids.extend(predicted_ids)
                        submitted_prefetch_ids.extend(
                            self.cache_manager.prefetch_experts_to_ram(ram_predicted_ids)
                        )
                        submitted_prefetch_ids.extend(
                            self.cache_manager.prefetch_for_next_step(prev_lookahead_probs)
                        )
                        sticky_ids = self._ordered_sticky_ids(vram_capacity)
                        promote_candidates = list(dict.fromkeys(
                            predicted_ids + sticky_ids + prev_current_ids
                        ))
                        ready_promote = self._promotion_ready_ids(promote_candidates)
                        if ready_promote:
                            promoted_expert_ids.extend(
                                self.cache_manager.ensure_resident(ready_promote[:vram_capacity])
                            )
                    elif prev_current_ids:
                        ready_promote = self._promotion_ready_ids(prev_current_ids)
                        if ready_promote:
                            promoted_expert_ids.extend(
                                self.cache_manager.ensure_resident(ready_promote)
                            )

            # Get availability masks AFTER ensure_resident so the mask
            # reflects experts that just became available.
                if scheduler is not None:
                    masks = scheduler.avail_masks(device=str(self.device))
                else:
                    masks = self.cache_manager.availability_masks_all_layers()

                past_len = past_key_values[0][0].shape[1] if past_key_values else 0
                outputs, lookahead_probs = self.model(
                    input_ids[:, past_len:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    available_expert_masks=masks,
                )

                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1
                    )

                route_stats = self._route_cache_stats(
                    masks,
                    include_all_tokens=(past_key_values is None),
                )
                cache_hits += int(route_stats["cache_hits"])
                cache_misses += int(route_stats["cache_misses"])
                hit_weight_mass += float(route_stats["hit_weight_mass"])
                fallback_weight_mass += float(route_stats["fallback_weight_mass"])
                total_route_weight_mass += float(route_stats["total_route_weight_mass"])
                route_layer_stats = route_stats["per_layer"]
                current_ids = route_stats["activated_expert_ids"]
                activated_expert_ids.extend(current_ids)
                rt = self._consume_moe_runtime_stats()
                on_demand_expert_ids.extend(rt["on_demand_experts"])
                fallback_expert_ids.extend(rt["fallback_experts"])
                on_demand_weight_mass += float(rt["on_demand_weight_mass"])
                fallback_executed_weight_mass += float(rt["fallback_executed_weight_mass"])
                if predicted_ids:
                    current_set = set(int(eid) for eid in current_ids)
                    prediction_total += len(current_set)
                    prediction_hits += len(current_set.intersection(int(eid) for eid in predicted_ids))
                if scheduler is None:
                    self.cache_manager.expert_store.reprioritize_resident_experts(
                        self._recent_route_scores(current_ids, predicted_ids, ram_predicted_ids)
                    )

                # Stash predictions for the NEXT iteration's prefetch.
                prev_lookahead_probs = lookahead_probs
                prev_current_ids = current_ids

                # Sampling
                logits = outputs.logits[:, -1, :] / temperature
                if repetition_penalty != 1.0:
                    for i in range(input_ids.shape[0]):
                        logits[i, torch.unique(input_ids[i])] /= repetition_penalty
                if top_k > 0:
                    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    mask_p = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                    mask_p[..., 1:] = mask_p[..., :-1].clone()
                    mask_p[..., 0] = 0
                    logits[mask_p.scatter(1, sorted_indices, mask_p)] = -float('inf')

                next_token = (
                    torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                    if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
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

                if streamer:
                    streamer.put(next_token.cpu())

                if eos_token_id is not None:
                    finished |= next_token.squeeze(-1).eq(eos_token_id)
                    if finished.all():
                        if step == 0:
                            prefill_done_t = time.monotonic()
                            prefill_mem1 = _rss_gb()
                            prefill_backend_mem1 = _backend_memory_snapshot(self.device)
                            prefill_time_s = prefill_done_t - prefill_t0
                        break

                if step == 0:
                    prefill_done_t = time.monotonic()
                    prefill_mem1 = _rss_gb()
                    prefill_backend_mem1 = _backend_memory_snapshot(self.device)
                    prefill_time_s = prefill_done_t - prefill_t0

            # ── Post-generation: update frequency heuristic ───────────────
            if scheduler is not None:
                scheduler.record_activation(list(set(activated_expert_ids)))

            end_t = time.monotonic()
            end_mem = _rss_gb()
            end_backend_mem = _backend_memory_snapshot(self.device)
            if tokens_generated == 0:
                prefill_done_t = end_t
                prefill_mem1 = end_mem
                prefill_backend_mem1 = end_backend_mem
                prefill_time_s = end_t - prefill_t0
            elapsed = end_t - prefill_t0
            decode_time_s = max(0.0, end_t - prefill_done_t)
            total_cache = cache_hits + cache_misses
            resident_hit_rate = cache_hits / max(total_cache, 1)
            fallback_weight_rate = fallback_weight_mass / max(total_route_weight_mass, 1e-12)
            executed_fallback_rate = fallback_executed_weight_mass / max(total_route_weight_mass, 1e-12)
            prediction_hit_rate = prediction_hits / max(prediction_total, 1)
            runtime_stats = dict(self._runtime_stats)
            promoted_expert_ids.extend(runtime_stats.get("on_demand_promoted_experts", []))
            backend_fields = {}
            backend_fields.update(_backend_memory_fields(prefill_backend_mem0, "before_prefill"))
            backend_fields.update(_backend_memory_fields(prefill_backend_mem1, "after_prefill"))
            backend_fields.update(_backend_memory_fields(end_backend_mem, "after_decode"))
            self.last_stats = {
                "tokens_generated": tokens_generated,
                "elapsed_s": round(elapsed, 3),
                "tokens_per_sec": round(tokens_generated / max(elapsed, 1e-6), 2),
                "prefill_time_s": round(prefill_time_s, 3),
                "decode_time_s": round(decode_time_s, 3),
                "prefill_rss_delta_gb": round(prefill_mem1 - prefill_mem0, 3),
                "decode_rss_delta_gb": round(end_mem - prefill_mem1, 3),
                "rss_before_prefill_gb": round(prefill_mem0, 3),
                "rss_after_prefill_gb": round(prefill_mem1, 3),
                "rss_after_decode_gb": round(end_mem, 3),
                **backend_fields,
                "cache_hit_rate": round(1.0 if miss_policy == "on_demand" and executed_fallback_rate == 0.0 else resident_hit_rate, 4),
                "resident_hit_rate": round(resident_hit_rate, 4),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "fallback_weight_rate": round(executed_fallback_rate, 4),
                "resident_miss_weight_rate": round(fallback_weight_rate, 4),
                "fallback_weight_mass": round(fallback_executed_weight_mass, 4),
                "resident_miss_weight_mass": round(fallback_weight_mass, 4),
                "hit_weight_mass": round(hit_weight_mass, 4),
                "total_route_weight_mass": round(total_route_weight_mass, 4),
                "on_demand_weight_mass": round(on_demand_weight_mass, 4),
                "on_demand_weight_rate": round(on_demand_weight_mass / max(total_route_weight_mass, 1e-12), 4),
                "on_demand_loads": int(runtime_stats.get("on_demand_loads", 0)),
                "on_demand_load_time_s": round(float(runtime_stats.get("on_demand_load_time_s", 0.0)), 4),
                "sync_ssd_loads": int(runtime_stats.get("sync_ssd_loads", 0)),
                "async_cold_miss_prefetches": int(runtime_stats.get("async_cold_miss_prefetches", 0)),
                "resident_vram_hits": int(runtime_stats.get("resident_vram_hits", 0)),
                "resident_ram_hits": int(runtime_stats.get("resident_ram_hits", 0)),
                "quality_safe": miss_policy == "sync_on_demand",
                "miss_policy": miss_policy,
                "prediction_hit_rate": round(prediction_hit_rate, 4),
                "prediction_hits": prediction_hits,
                "prediction_total": prediction_total,
                "route_layer_stats": route_layer_stats,
                "prefill_expert_ids": prefill_expert_ids,
                "prefill_ram_expert_ids": prefill_ram_expert_ids,
                "activated_expert_ids": sorted(set(activated_expert_ids)),
                "predicted_current_expert_ids": sorted(set(predicted_current_expert_ids)),
                "prefetch_submitted_ids": sorted(set(submitted_prefetch_ids)),
                "promoted_expert_ids": sorted(set(promoted_expert_ids)),
                "on_demand_expert_ids": sorted(set(on_demand_expert_ids)),
                "fallback_expert_ids": sorted(set(fallback_expert_ids)),
                "prefetch_submitted_total": len(submitted_prefetch_ids),
                "promoted_total": len(promoted_expert_ids),
                **self.cache_manager.stats(),
            }

            if streamer:
                streamer.end()
            return input_ids
        finally:
            self._clear_moe_runtime_hooks()
