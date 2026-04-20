"""
chronos/router/prefill_scheduler.py

Orchestrates the dual-layer pipeline:

  Prefill phase:
    1. Run IntentClassifier on full prompt → IntentVector
    2. ExpertPredictor converts IntentVector → ExpertSet
    3. Batch-load all predicted experts to VRAM (or Metal) BEFORE
       generation starts — so generation sees zero IO latency

  Generation phase:
    4. Pass per-layer avail masks to model.forward()
    5. After generation, update ExpertPredictor frequency table

This is the key architectural difference from per-token LookaheadRouter:
IO is fully front-loaded into prefill, not spread across generation steps.

Usage (with IntentClassifier trained):
    scheduler = PrefillScheduler(
        classifier=intent_clf,
        predictor=expert_pred,
        expert_store=store,           # ExpertStore or MLXExpertStore
    )

    # Called once before generation
    expert_set = scheduler.prepare(prompt_ids, device)

    # Get masks to pass to model.forward()
    masks = scheduler.avail_masks(device)

    # After generation
    scheduler.record_activation(activated_expert_ids)

Usage (without IntentClassifier — fallback mode):
    scheduler = PrefillScheduler(
        classifier=None,              # No classifier
        predictor=expert_pred,
        expert_store=store,
    )
    # Falls back to frequency-based heuristic selection
"""
import time
import threading
from typing import List, Optional, Set

import torch


class PrefillScheduler:
    """
    Dual-layer prefill: intent classify → expert predict → batch preload.

    classifier: IntentClassifier or None (disables Layer 1, uses heuristic)
    predictor:  ExpertPredictor (always required)
    expert_store: ExpertStore (PyTorch) or MLXExpertStore
    async_load: if True, preloads in a background thread while the main
                model is still running the LM head on the prompt tokens
    """

    def __init__(
        self,
        classifier,           # IntentClassifier | None
        predictor,            # ExpertPredictor
        expert_store,
        async_load: bool = True,
    ):
        self.classifier = classifier
        self.predictor = predictor
        self.store = expert_store
        self.async_load = async_load

        self._current_expert_set = None
        self._load_thread: Optional[threading.Thread] = None
        self._load_done = threading.Event()

    # ── Public API ────────────────────────────────────────────────

    def prepare(
        self,
        prompt_ids: torch.Tensor,    # [1, S] token ids
        device: str = "cpu",
        classifier_device: str = "cpu",
    ) -> "ExpertSet":
        """
        Run prefill scheduling. Call this BEFORE model.generate().

        1. Run IntentClassifier (or use heuristic if classifier=None)
        2. Determine ExpertSet
        3. Kick off background loading (or synchronous if async=False)

        Returns ExpertSet immediately (loading may still be in progress
        if async_load=True — call wait() before first forward pass).
        """
        t0 = time.monotonic()

        if self.classifier is not None:
            self.classifier.eval()
            with torch.no_grad():
                ids = prompt_ids.to(classifier_device)
                intent = self.classifier(ids)
            expert_set = self.predictor.predict(intent)
        else:
            # No classifier: build a synthetic IntentVector from frequency table
            from chronos.router.intent_classifier import IntentVector
            freq = self.predictor._frequency
            per_layer = [freq.clone()
                         for _ in range(self.predictor.num_moe_layers)]
            intent = IntentVector(per_layer_probs=per_layer, global_probs=freq)
            expert_set = self.predictor.predict(intent)

        self._current_expert_set = expert_set
        elapsed_classify = (time.monotonic() - t0) * 1000

        # Trigger expert loading
        self._load_done.clear()
        if self.async_load:
            self._load_thread = threading.Thread(
                target=self._load_experts,
                args=(expert_set.expert_ids,),
                daemon=True,
                name="prefill-loader",
            )
            self._load_thread.start()
        else:
            self._load_experts(expert_set.expert_ids)

        mode = "async" if self.async_load else "sync"
        print(
            f"[PrefillScheduler] classify={elapsed_classify:.1f}ms  "
            f"experts={sorted(expert_set.expert_ids)}  "
            f"confidence={expert_set.confidence:.2f}  "
            f"fallback={expert_set.fallback}  load={mode}"
        )
        return expert_set

    def wait(self):
        """
        Block until background expert loading is complete.
        Call this just before the first generation forward pass.
        """
        self._load_done.wait()
        # Sync H2D stream on CUDA backends
        if hasattr(self.store, "sync_h2d"):
            self.store.sync_h2d()

    def avail_masks(self, device=None) -> List[torch.Tensor]:
        """
        Return per-layer float availability masks to pass to model.forward().
        Returns all-ones masks if no expert set has been computed yet
        (safe default: treat all experts as available).
        """
        if self._current_expert_set is None:
            num_layers = self.predictor.num_moe_layers
            num_experts = self.predictor.num_experts
            return [torch.ones(num_experts).to(device or "cpu")
                    for _ in range(num_layers)]
        return self.predictor.avail_masks_float(
            self._current_expert_set, device=device
        )

    def record_activation(self, activated_expert_ids: List[int]):
        """
        Call after generation completes with the list of expert IDs
        that were actually used. Updates the frequency heuristic.
        """
        self.predictor.update_frequency(activated_expert_ids)

    # ── Internal ─────────────────────────────────────────────────

    def _load_experts(self, expert_ids: Set[int]):
        """
        SSD → RAM → VRAM for the predicted expert set.
        Runs in background thread when async_load=True.
        """
        try:
            id_list = list(expert_ids)
            # Step 1: SSD → pinned RAM (bulk)
            self.store.prefetch_to_ram(id_list)
            # Step 2: RAM → VRAM (promote each)
            for eid in id_list:
                self.store.promote_to_vram(eid)
        finally:
            self._load_done.set()

    # ── Integration helpers ───────────────────────────────────────

    @classmethod
    def build(
        cls,
        main_model,
        config,
        expert_store,
        classifier_path: Optional[str] = None,
        device: str = "cpu",
        async_load: bool = True,
    ) -> "PrefillScheduler":
        """
        Factory: build a PrefillScheduler from a main model + config.
        Loads IntentClassifier from disk if path provided; otherwise
        creates an untrained one (will use heuristic fallback until trained).

        classifier_path: path to .pt checkpoint, or None to skip classifier
        """
        from chronos.router.intent_classifier import IntentClassifier
        from chronos.router.expert_predictor import ExpertPredictor

        from chronos.model.moe_chronos import ChronosMOEFeedForward
        num_moe_layers = sum(
            1 for l in main_model.model.layers
            if isinstance(l.mlp, ChronosMOEFeedForward)
        )

        predictor = ExpertPredictor(
            num_experts=config.num_experts,
            num_moe_layers=num_moe_layers,
            vram_capacity=expert_store.vram_capacity
            if hasattr(expert_store, "vram_capacity") else config.num_experts,
        )

        classifier = None
        if classifier_path is not None and __import__("os").path.exists(classifier_path):
            classifier = IntentClassifier.load(
                classifier_path,
                vocab_size=config.vocab_size,
                num_experts=config.num_experts,
                num_moe_layers=num_moe_layers,
            ).to(device)
            print(f"[PrefillScheduler] Loaded IntentClassifier from {classifier_path}")
        elif classifier_path is not None:
            # Path provided but file doesn't exist yet → create untrained
            classifier = IntentClassifier(
                vocab_size=config.vocab_size,
                num_experts=config.num_experts,
                num_moe_layers=num_moe_layers,
            ).to(device)
            print("[PrefillScheduler] IntentClassifier not found — "
                  "using heuristic fallback (train with chronos train --mode intent)")

        return cls(
            classifier=classifier,
            predictor=predictor,
            expert_store=expert_store,
            async_load=async_load,
        )
