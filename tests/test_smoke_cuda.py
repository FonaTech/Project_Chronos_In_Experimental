"""tests/test_smoke_cuda.py — runs only when CUDA is available.

Exercises the M3 double-stream pipeline on a real GPU. The same logic is
covered by tests/test_smoke.py::test_pipeline_overlap_simulated using a
sleep-based SSD simulator on CPU; this file proves the same flow works
with actual ``torch.cuda.Event`` and ``current_stream.wait_event(evt)``.

Skipped on hosts without CUDA (the smoke runner does this gracefully).
"""
import os
import sys
import tempfile
import time

import torch


def main():
    if not torch.cuda.is_available():
        print("SKIP test_smoke_cuda — no CUDA device on this host")
        sys.exit(0)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import chronos.deps  # noqa: F401
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.runtime.cache_manager import CacheManager

    cfg = ChronosConfig(
        hidden_size=256, num_hidden_layers=4, num_experts=8,
        use_moe=True, use_hybrid_attention=True,
        kv_latent_dim=32, rope_dim=8, sliding_window_size=64,
        vram_budget_gb=0.05, num_shared_experts=1,
    )
    model = ChronosForCausalLM(cfg).to("cuda")
    model.eval()

    ssd = tempfile.mkdtemp(prefix="chronos_cuda_smoke_")
    mgr = CacheManager(model, cfg, ssd_dir=ssd)
    mgr.expert_store.offload_all_to_ssd(
        clusters=[[i, i + 4] for i in range(4)]  # 4 clusters of 2 experts each
    )
    mgr.expert_store.attach_cluster_manifest()
    mgr.warm_up(initial_expert_ids=[0, 1])
    mgr.start()

    # Generate a few tokens with the new pipeline path
    x = torch.randint(0, cfg.vocab_size, (1, 16), device="cuda")
    past = None
    prev_lp = None
    prev_ids = []
    n = 8
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(n):
            if prev_lp is not None:
                mgr.prefetch_for_next_step(prev_lp)
                mgr.ensure_resident(prev_ids)
            mask = mgr.availability_mask()
            outputs, lp = model(
                x, past_key_values=past, use_cache=True,
                available_expert_masks=[mask] * len(mgr.expert_store.moe_layers),
            )
            past = outputs.past_key_values
            from chronos.model.moe_chronos import ChronosMOEFeedForward
            moes = [l.mlp for l in model.model.layers
                    if isinstance(l.mlp, ChronosMOEFeedForward)]
            cur = moes[0].last_router_probs[:, -1, :].argmax(dim=-1).unique().cpu().tolist()
            prev_lp, prev_ids = lp, cur
            x = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    elapsed = time.monotonic() - t0
    mgr.stop()

    # Sanity: events were used (h2d_stream is set on cuda)
    assert mgr.expert_store._h2d_stream is not None, "no _h2d_stream on cuda?!"
    print(f"PASS test_smoke_cuda — {n} tokens in {elapsed:.3f}s on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
