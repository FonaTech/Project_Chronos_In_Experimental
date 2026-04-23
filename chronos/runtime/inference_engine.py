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
from typing import List, Optional

import torch
import torch.nn.functional as F

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.moe_chronos import ChronosMOEFeedForward
from chronos.runtime.cache_manager import CacheManager
from chronos.router.prefill_scheduler import PrefillScheduler


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

    def teardown(self):
        self.cache_manager.stop()

    def _get_current_expert_ids(self) -> List[int]:
        """Extract top-1 expert IDs from the last forward pass (layer 0)."""
        moe_layers = [
            l.mlp for l in self.model.model.layers
            if isinstance(l.mlp, ChronosMOEFeedForward)
        ]
        if not moe_layers or moe_layers[0].last_router_probs is None:
            return []
        probs = moe_layers[0].last_router_probs  # [B, S, E]
        return probs[:, -1, :].argmax(dim=-1).unique().cpu().tolist()

    def _plan_prefill_experts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Probe prompt routing with shared fallback, then load hot experts.

        The live model contains placeholder experts after lazy setup. A cold
        first prefill would therefore run entirely through shared fallback and
        can diverge from the real model before decode starts. This probe pass
        does not materialize expert modules; it only records gate choices, then
        returns the most frequent expert IDs bounded by the active cache budget.
        The caller reruns the real prefill after those experts are promoted.
        """
        if not getattr(self.config, "use_moe", True):
            return []
        num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        if num_experts <= 0:
            return []

        cold_mask = torch.zeros(num_experts, dtype=torch.bool, device=self.device)
        masks = [cold_mask] * len(self.cache_manager.expert_store.moe_layers)
        self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            available_expert_masks=masks,
        )

        counts = torch.zeros(num_experts, dtype=torch.long)
        top_k = int(getattr(self.config, "num_experts_per_tok", 1) or 1)
        top_k = max(1, min(top_k, num_experts))
        for layer in self.model.model.layers:
            moe = getattr(layer, "mlp", None)
            if not isinstance(moe, ChronosMOEFeedForward) or moe.last_router_probs is None:
                continue
            probs = moe.last_router_probs.reshape(-1, num_experts)
            expert_ids = torch.topk(probs, k=top_k, dim=-1).indices.reshape(-1).cpu()
            counts += torch.bincount(expert_ids, minlength=num_experts)

        if not torch.any(counts):
            return []
        capacity = max(1, min(
            int(getattr(self.cache_manager.expert_store, "vram_capacity", 1) or 1),
            num_experts,
        ))
        ordered = torch.argsort(counts, descending=True).tolist()
        return [int(eid) for eid in ordered if int(counts[eid]) > 0][:capacity]

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
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # ── Prefill phase: front-load expert IO before generation ──────────
        if scheduler is not None:
            scheduler.prepare(input_ids, device=str(self.device))
            scheduler.wait()  # block until SSD→VRAM load complete
            prefill_expert_ids: List[int] = []
        else:
            prefill_expert_ids = self._plan_prefill_experts(input_ids, attention_mask)
            if prefill_expert_ids:
                self.cache_manager.ensure_resident(prefill_expert_ids)

        past_key_values = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
        if streamer:
            streamer.put(input_ids.cpu())

        t0 = time.monotonic()
        tokens_generated = 0
        cache_hits, cache_misses = 0, 0
        activated_expert_ids: List[int] = []
        # M3: holds last step's lookahead predictions; we submit them as
        # prefetch BEFORE this step's forward so the H2D stream overlaps
        # with compute. None on the first iteration (no predictions yet).
        prev_lookahead_probs = None
        prev_current_ids: List[int] = []

        for step in range(max_new_tokens):
            # ── M3 pipeline: fire async prefetch BEFORE forward ──
            if scheduler is None:
                if prev_lookahead_probs is not None:
                    # Submit next-token predictions to the async SSD→RAM thread.
                    self.cache_manager.prefetch_for_next_step(prev_lookahead_probs)
                if prev_current_ids:
                    # Ensure the experts we actually need NOW are resident in
                    # VRAM, waiting only on their own H2D events.
                    self.cache_manager.ensure_resident(prev_current_ids)

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

            # Track cache hit/miss from mask
            mask_tensor = masks[0] if masks else None
            current_ids = self._get_current_expert_ids()
            if mask_tensor is not None:
                for eid in current_ids:
                    if eid < len(mask_tensor) and mask_tensor[eid]:
                        cache_hits += 1
                    else:
                        cache_misses += 1
            activated_expert_ids.extend(current_ids)

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
                    break

        # ── Post-generation: update frequency heuristic ───────────────────
        if scheduler is not None:
            scheduler.record_activation(list(set(activated_expert_ids)))

        elapsed = time.monotonic() - t0
        total_cache = cache_hits + cache_misses
        self.last_stats = {
            "tokens_generated": tokens_generated,
            "elapsed_s": round(elapsed, 3),
            "tokens_per_sec": round(tokens_generated / max(elapsed, 1e-6), 2),
            "cache_hit_rate": round(cache_hits / max(total_cache, 1), 4),
            "prefill_expert_ids": prefill_expert_ids,
            **self.cache_manager.stats(),
        }

        if streamer:
            streamer.end()
        return input_ids
