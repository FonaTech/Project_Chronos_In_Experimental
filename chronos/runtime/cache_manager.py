"""
chronos/runtime/cache_manager.py

Unified VRAM/RAM cache manager for Project Chronos inference.

Wraps ExpertStore + AsyncPrefetcher into a single interface used by
the inference engine. Provides:
- availability_mask() for soft gating
- step() to advance prefetch schedule after each decode step
- stats() for monitoring
"""
from typing import List, Optional
import os
import torch

from chronos.io.expert_store import ExpertStore
from chronos.io.async_prefetcher import AsyncPrefetcher, PrefetchScheduler
from chronos.io.storage import ClusterStorage, MANIFEST_FILENAME


class CacheManager:
    """
    Single entry point for all expert caching operations during inference.

    Lifecycle:
        mgr = CacheManager(model, config, ssd_dir="./expert_cache")
        mgr.start()
        for each decode step:
            mask = mgr.availability_mask()          # pass to model.forward
            outputs, lp = model(x, available_expert_masks=[mask]*L)
            mgr.step(lp, current_expert_ids)        # schedule next prefetch
        mgr.stop()
    """

    def __init__(self, model, config, ssd_dir: str = "./expert_cache"):
        self.config = config
        self.expert_store = ExpertStore(model, config, ssd_dir=ssd_dir)
        self.prefetcher = AsyncPrefetcher(self.expert_store)
        self.scheduler = PrefetchScheduler(self.prefetcher, self.expert_store)
        self._num_layers = len(self.expert_store.moe_layers)

        # Auto-attach any cluster manifest present on disk. Also supports an
        # explicit override via config.cluster_manifest_path pointing to a
        # different layout (e.g. a post-calibration Louvain repack).
        manifest_override = getattr(config, "cluster_manifest_path", None)
        if manifest_override and ClusterStorage.has_manifest(manifest_override):
            self.expert_store.attach_cluster_manifest(manifest_override)
        elif ClusterStorage.has_manifest(ssd_dir):
            self.expert_store.attach_cluster_manifest(ssd_dir)

    def start(self):
        """Start background prefetch thread."""
        self.prefetcher.start()

    def stop(self):
        """Stop background prefetch thread."""
        self.prefetcher.stop()

    def warm_up(self, initial_expert_ids: Optional[List[int]] = None):
        """
        Pre-load a set of experts into VRAM before generation starts.
        Defaults to experts 0..vram_capacity-1.
        """
        if initial_expert_ids is None:
            initial_expert_ids = list(range(
                min(self.config.num_experts, self.expert_store.vram_capacity)
            ))
        self.expert_store.prefetch_to_ram(initial_expert_ids)
        for eid in initial_expert_ids:
            self.expert_store.promote_to_vram(eid)
        # Ensure all H2D copies are complete before model sees the weights
        self.expert_store.sync_h2d()

    def availability_mask(self) -> torch.Tensor:
        """[num_experts] bool — True if expert is currently in VRAM."""
        return self.expert_store.vram_availability_mask()

    def availability_masks_all_layers(self) -> List[torch.Tensor]:
        """Returns the same mask replicated for each MoE layer."""
        mask = self.availability_mask()
        return [mask] * self._num_layers

    def step(self, lookahead_probs, current_expert_ids: List[int]):
        """
        Legacy single-call API (M1+M2 compatible): blocking promote of
        current experts, then async submit of future predictions, then
        full sync. Use the M3 split below to get pipeline overlap.
        """
        self.scheduler.step(lookahead_probs, current_expert_ids)
        self.expert_store.sync_h2d()

    # ── M3: split prefetch from blocking sync ────────────────────

    def prefetch_for_next_step(self, lookahead_probs) -> List[int]:
        """Fire off async prefetch for the LookaheadRouter's predictions.
        Non-blocking. Should be called BEFORE the next forward so H2D and
        compute overlap. Returns the submitted expert IDs."""
        return self.scheduler.prefetch_only(lookahead_probs)

    def prefetch_experts_to_ram(self, expert_ids: List[int]) -> List[int]:
        """Queue exact expert IDs for SSD->RAM prefetch without touching VRAM."""
        expert_ids = list(dict.fromkeys(int(eid) for eid in expert_ids or []))
        if expert_ids:
            self.prefetcher.submit(expert_ids)
        return expert_ids

    def ensure_resident(self, expert_ids: List[int]) -> List[int]:
        """Promote `expert_ids` to VRAM (non-blocking H2D) and then have the
        compute stream wait on *only those* events. Returns the list newly
        promoted. After this call returns, the named experts are guaranteed
        visible to the next forward without globally syncing other in-flight
        H2D copies."""
        newly = self.scheduler.promote_current(expert_ids, blocking=False)
        # The compute stream waits on each newly-promoted expert's event;
        # already-resident experts contribute no event, so this only blocks
        # on the truly necessary transfers.
        self.expert_store.wait_for_experts(newly)
        return newly

    def stats(self) -> dict:
        return {
            **self.expert_store.stats(),
            **self.prefetcher.stats,
        }
