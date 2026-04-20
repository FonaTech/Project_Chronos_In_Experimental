"""
tests/test_smoke.py — Project Chronos smoke tests for CI
"""
import sys
import os

# When running pytest from repo root, ensure chronos package is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import chronos.deps  # auto-bootstrap minimind on sys.path

import torch
from torch.utils.data import DataLoader, TensorDataset



def make_model(window=16):
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    cfg = ChronosConfig(
        hidden_size=128, num_hidden_layers=4, num_experts=4,
        use_moe=True, use_hybrid_attention=True,
        kv_latent_dim=16, rope_dim=8, sliding_window_size=window,
        vram_budget_gb=0.5,
    )
    return ChronosForCausalLM(cfg), cfg


def test_forward():
    model, cfg = make_model()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    out, lp = model(x, labels=x)
    assert out.loss.item() > 0
    assert lp.shape == (2, 16, cfg.lookahead_steps + 1, cfg.num_experts)


def test_kv_cache_bounded():
    model, cfg = make_model(window=8)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out, _ = model(x, use_cache=True)
        past_kv = out.past_key_values
        for _ in range(30):
            xn = torch.randint(0, cfg.vocab_size, (1, 1))
            out2, _ = model(xn, past_key_values=past_kv, use_cache=True)
            past_kv = out2.past_key_values
    from chronos.model.hybrid_attention import SlidingWindowAttention
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.self_attn, SlidingWindowAttention):
            assert past_kv[i][0].shape[1] <= 8, f"Layer {i} KV not bounded"


def test_temporal_loss():
    from chronos.model.temporal_loss import temporal_locality_loss, total_loss
    probs = torch.rand(2, 16, 4)
    tl = temporal_locality_loss(probs)
    assert tl.item() >= 0
    ce = torch.tensor(2.0)
    bal = torch.tensor(0.01)
    loss = total_loss(ce, bal, probs, 5e-4, 1e-3)
    assert loss.item() > ce.item()


def test_lru_cache():
    from chronos.io.expert_store import LRUCache
    lru = LRUCache(capacity=3)
    lru.put(0); lru.put(1); lru.put(2)
    evicted = lru.put(3)
    assert evicted == 0
    assert lru.contains(3)
    assert not lru.contains(0)


def test_expert_store_init():
    from chronos.io.expert_store import ExpertStore
    model, cfg = make_model()
    store = ExpertStore(model, cfg, ssd_dir='/tmp/chronos_ci_test')
    s = store.stats()
    assert s['vram_capacity'] >= 1
    assert s['ram_capacity_dynamic'] >= 1
    assert 'available_ram_gb' in s


def test_async_prefetcher():
    import time
    from chronos.io.expert_store import ExpertStore
    from chronos.io.async_prefetcher import AsyncPrefetcher
    model, cfg = make_model()
    store = ExpertStore(model, cfg, ssd_dir='/tmp/chronos_ci_test')
    pf = AsyncPrefetcher(store)
    pf.start()
    pf.submit([0, 1])
    time.sleep(0.05)
    pf.stop()
    assert pf.stats['total_requests'] == 2


def test_hybrid_attention_layers():
    from chronos.model.hybrid_attention import MLAAttention, SlidingWindowAttention
    model, cfg = make_model()
    for i, layer in enumerate(model.model.layers):
        if i % 2 == 0:
            assert isinstance(layer.self_attn, MLAAttention), f"Layer {i} should be MLA"
        else:
            assert isinstance(layer.self_attn, SlidingWindowAttention), f"Layer {i} should be SW"


def test_benchmark_functions():
    from chronos.eval.benchmark import compute_perplexity, measure_throughput
    model, cfg = make_model()
    ids = torch.randint(0, cfg.vocab_size, (4, 16))
    ds = TensorDataset(ids, ids.clone())
    loader = DataLoader(ds, batch_size=2)
    ppl = compute_perplexity(model, loader, 'cpu', max_batches=2)
    assert ppl > 1.0
    tps = measure_throughput(model, ids[:1, :8], 'cpu', max_new_tokens=5)
    assert tps > 0


# ── M1: cluster-aware safetensors storage ─────────────────────────

def test_cluster_storage_roundtrip(tmp_path=None):
    """Write every expert to .ctsr, read them back, confirm bit-identity."""
    import tempfile, os as _os
    from chronos.io.storage import ClusterStorage
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    model, cfg = make_model()
    moe_layers = [l.mlp for l in model.model.layers
                  if isinstance(l.mlp, ChronosMOEFeedForward)]
    # Two clusters — exercise the manifest mapping.
    clusters = [[0, 2], [1, 3]]
    out_dir = tempfile.mkdtemp(prefix="chronos_ctsr_")
    ClusterStorage.write_clusters(
        moe_layers=moe_layers, clusters=clusters,
        output_dir=out_dir, num_layers=len(moe_layers),
        num_experts=cfg.num_experts, dtype=torch.float16,
    )
    cs = ClusterStorage(out_dir)
    for cid, expected_experts in [(0, [0, 2]), (1, [1, 3])]:
        nested = cs.load_cluster(cid, dtype=torch.float16)
        # All expected experts present
        assert sorted(nested.keys()) == sorted(expected_experts), \
            f"cluster {cid} keys mismatch: {nested.keys()}"
        # All layers present per expert
        for eid in expected_experts:
            assert sorted(nested[eid].keys()) == list(range(len(moe_layers))), \
                f"layer set mismatch for expert {eid}"
            # bit-identity vs source
            for li in range(len(moe_layers)):
                src_state = moe_layers[li].experts[eid].state_dict()
                for pname, src_t in src_state.items():
                    loaded = nested[eid][li][pname]
                    src_fp16 = src_t.detach().to(torch.float16).cpu()
                    assert torch.equal(loaded, src_fp16), \
                        f"mismatch at expert={eid} layer={li} param={pname}"


def test_expert_store_clustered_prefetch(tmp_path=None):
    """Requesting one expert pulls in its whole Louvain cluster."""
    import tempfile, os as _os
    from chronos.io.expert_store import ExpertStore
    model, cfg = make_model()
    ssd = tempfile.mkdtemp(prefix="chronos_es_")
    store = ExpertStore(model, cfg, ssd_dir=ssd)
    # Force a 2-expert cluster containing experts 0 and 2.
    clusters = [[0, 2], [1, 3]]
    store.offload_all_to_ssd(clusters=clusters)
    assert store.stats()["cluster_aware"] is True
    assert store.stats()["num_clusters"] == 2

    # Prefetch only expert 0 — expert 2 must come along for the ride.
    store.prefetch_to_ram([0])
    assert store.ram_lru.contains(0), "requested expert not in RAM"
    assert store.ram_lru.contains(2), "co-cluster expert not pulled"
    # Expert from a different cluster must NOT be loaded.
    assert not store.ram_lru.contains(1), "wrong-cluster expert leaked"


# ── M2: lookahead supervision loss ────────────────────────────────

def test_lookahead_supervision_grad():
    """One backward step must produce non-zero grad on the LookaheadRouter."""
    from chronos.model.temporal_loss import lookahead_supervision_loss
    model, cfg = make_model()
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    out, lookahead_probs = model(x, labels=x)
    # Build a teacher: layer-averaged real router probs (stop-grad).
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    probs = [l.mlp.last_router_probs for l in model.model.layers
             if isinstance(l.mlp, ChronosMOEFeedForward) and l.mlp.last_router_probs is not None]
    teacher = torch.stack(probs, dim=2).mean(dim=2).detach()  # [B, S, E]
    la_loss = lookahead_supervision_loss(lookahead_probs, teacher, cfg.lookahead_steps)
    assert la_loss.item() > 0, "lookahead loss should be strictly positive"
    la_loss.backward()
    last_layer = model.model.lookahead_router.proj[-1]
    assert last_layer.weight.grad is not None, "no grad on lookahead head"
    assert last_layer.weight.grad.abs().sum().item() > 0, "zero grad on lookahead head"


# ── M3: double-stream pipeline ────────────────────────────────────

def test_pipeline_overlap_simulated():
    """Measure the structural overlap benefit directly:

    The M3 pipeline promises `prefetch_for_next_step` is *non-blocking*
    (work happens in the async daemon) while the legacy `step(...)` path
    blocks the caller on promote_to_vram (which in turn calls the
    synchronous prefetch_to_ram → SSD sim sleep).

    Rather than chase wall-clock speedup through 8 forwards (tiny
    toy-model forward pass is ~0.5ms, drowned by thread jitter), we
    measure the two primitives under CHRONOS_SIM_SSD_MS=30:

      (a) prefetch_for_next_step(lp)      should return in <<30ms.
      (b) step(lp, [fresh_expert_id])     must block for ≥25ms.

    That is the structural invariant that makes the decode loop overlap
    possible on real GPU/SSD hardware. If it holds on CPU it holds
    everywhere.
    """
    import os as _os, tempfile, time as _time
    from chronos.runtime.cache_manager import CacheManager
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM

    cfg = ChronosConfig(
        hidden_size=64, num_hidden_layers=2, num_experts=9,
        use_moe=True, use_hybrid_attention=True,
        kv_latent_dim=8, rope_dim=4, sliding_window_size=16,
        vram_budget_gb=0.01,
        num_shared_experts=1,
    )
    model = ChronosForCausalLM(cfg)
    model.eval()

    ssd = tempfile.mkdtemp(prefix="chronos_overlap_")
    mgr = CacheManager(model, cfg, ssd_dir=ssd)
    # Three distinct clusters so we can use one for async prefetch, another
    # for the legacy-blocking test, with no overlap between them.
    mgr.expert_store.offload_all_to_ssd(
        clusters=[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    )
    mgr.expert_store.attach_cluster_manifest()
    mgr.warm_up(initial_expert_ids=[0])
    mgr.start()

    # Synthesize a lookahead tensor predicting expert 5 at t+1 (cluster 1).
    E = cfg.num_experts
    lp = torch.zeros(1, 1, cfg.lookahead_steps + 1, E)
    lp[..., 0] = 1.0 / E
    lp[0, 0, 1, 5] = 1.0           # t+1 → expert 5 → loads cluster [3,4,5]

    _os.environ["CHRONOS_SIM_SSD_MS"] = "30"
    try:
        # Case A: async prefetch must return fast
        t0 = _time.monotonic()
        mgr.prefetch_for_next_step(lp)
        async_ret_ms = (_time.monotonic() - t0) * 1000

        # Let the daemon finish
        _time.sleep(0.1)

        # Case B: legacy step() promoting an expert from a DIFFERENT
        # cluster (7 → cluster [6,7,8]) must block on SSD sim.
        lp2 = torch.zeros(1, 1, cfg.lookahead_steps + 1, E)
        lp2[..., 0] = 1.0 / E
        t0 = _time.monotonic()
        mgr.step(lp2, [7])
        legacy_blocked_ms = (_time.monotonic() - t0) * 1000
    finally:
        _os.environ.pop("CHRONOS_SIM_SSD_MS", None)
        mgr.stop()

    print(f"\n  [overlap] prefetch_for_next_step: {async_ret_ms:.1f}ms (want <<30)")
    print(f"  [overlap] legacy step() w/ miss:  {legacy_blocked_ms:.1f}ms (want ≥25)")
    assert async_ret_ms < 15, f"prefetch_for_next_step blocked {async_ret_ms:.1f}ms (expected <<30ms)"
    assert legacy_blocked_ms >= 25, f"legacy step() did not block on SSD (got {legacy_blocked_ms:.1f}ms)"


# ── M5/M6 tests ───────────────────────────────────────────────────

def test_backend_dispatcher():
    from chronos.backend import BackendDispatcher
    d = BackendDispatcher()
    avail = d.available()
    assert "cpu" in avail, f"cpu must be available, got {avail}"
    assert d.device_str("cpu") == "cpu"
    assert d.supports_training("cpu") is True
    assert d.supports_training("vulkan") is False  # no upstream autograd
    assert d.info("opencl").available is False     # stub
    sel = d.select()
    assert sel in avail


def test_param_estimator():
    from ui.estimator import ArchConfig, total_params, active_params
    cfg = ArchConfig(
        hidden_size=64, num_hidden_layers=2, num_experts=4,
        num_experts_per_tok=1, num_shared_experts=1,
        moe_intermediate_size=128, vocab_size=1000,
        num_attention_heads=4, num_key_value_heads=4,
        tie_word_embeddings=True,
    )
    tot = total_params(cfg)
    act = active_params(cfg)
    assert tot > 0 and act > 0, f"got tot={tot} act={act}"
    assert act < tot, "active must be smaller than total for MoE"
    # Sanity: embedding dominates; should be > vocab * hidden
    assert tot >= cfg.vocab_size * cfg.hidden_size


def test_metrics_bus():
    from chronos.runtime.metrics import MetricsBus
    bus = MetricsBus()
    for i in range(5):
        bus.record("foo", float(i))
    assert bus.latest("foo") == 4.0
    ser = bus.series("foo")
    assert len(ser) == 5
    snap = bus.snapshot()
    assert "foo" in snap
    bus.reset()
    assert bus.snapshot() == {}


def test_vllm_adapter_no_vllm():
    from chronos.serving import register_chronos_with_vllm, HAS_VLLM, is_available
    # On a host without vLLM, this is a graceful no-op (returns False).
    result = register_chronos_with_vllm(verbose=False)
    # Accept either True (if a user DID install vllm) or False (no-op).
    assert isinstance(result, bool)
    assert isinstance(HAS_VLLM, bool)
    assert is_available() == HAS_VLLM


def test_hf_save_load_roundtrip():
    import tempfile
    from chronos.model.hf_io import save_chronos_pretrained, load_chronos_pretrained
    torch.manual_seed(0)
    model, cfg = make_model()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    out1, _ = model(x)
    before = out1.logits.detach().clone()
    with tempfile.TemporaryDirectory() as d:
        save_chronos_pretrained(model, d)
        m2, cache = load_chronos_pretrained(d)
        m2.eval()
        out2, _ = m2(x)
        diff = (before - out2.logits.detach()).abs().max().item()
    assert diff < 1e-4, f"HF roundtrip mismatch: max diff={diff:.2e}"


def test_ui_build():
    # Verify the WebUI Blocks tree constructs without errors.
    # (We don't launch it — just check `build_app()` returns something.)
    from chronos.app import build_app
    app = build_app()
    assert app is not None


if __name__ == '__main__':
    tests = [
        test_forward,
        test_kv_cache_bounded,
        test_temporal_loss,
        test_lru_cache,
        test_expert_store_init,
        test_async_prefetcher,
        test_hybrid_attention_layers,
        test_benchmark_functions,
        test_cluster_storage_roundtrip,
        test_expert_store_clustered_prefetch,
        test_lookahead_supervision_grad,
        test_pipeline_overlap_simulated,
        test_backend_dispatcher,
        test_param_estimator,
        test_metrics_bus,
        test_vllm_adapter_no_vllm,
        test_hf_save_load_roundtrip,
        test_ui_build,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {t.__name__}: {e}')
    print(f'\n{passed}/{len(tests)} tests passed.')
    if passed < len(tests):
        sys.exit(1)
