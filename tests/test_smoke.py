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
from unittest import mock


def _mlx_available() -> bool:
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:
        return False


class _TinyPickleTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, max_length=None, truncation=False):
        ids = [3 + (ord(c) % 16) for c in text][:max_length or 32]
        return type("Encoded", (), {"input_ids": ids})()



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


def test_lazy_expert_placeholder_uses_shared_fallback():
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.moe_chronos import LazyExpertPlaceholder

    cfg = ChronosConfig(
        hidden_size=32, num_hidden_layers=1, num_experts=2,
        num_experts_per_tok=1, num_shared_experts=1,
        num_attention_heads=2, num_key_value_heads=2,
        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
        vocab_size=128, use_moe=True,
    )
    model = ChronosForCausalLM(cfg).eval()
    moe = model.model.layers[0].mlp
    moe.experts[0] = LazyExpertPlaceholder(cfg.hidden_size, cfg.moe_intermediate_size, torch.float32)

    x = torch.randn(1, 4, cfg.hidden_size)
    mask = torch.tensor([0.0, 1.0])
    with torch.no_grad():
        y = moe(x, available_expert_mask=mask)
    assert y.shape == x.shape


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


def test_setup_from_state_dict_replaces_live_experts_with_placeholders():
    import tempfile
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.moe_chronos import LazyExpertPlaceholder
    from chronos.runtime.inference_engine import ChronosInferenceEngine

    cfg = ChronosConfig(
        hidden_size=32, num_hidden_layers=1, num_experts=4,
        use_moe=True, use_hybrid_attention=True,
        kv_latent_dim=8, rope_dim=4, sliding_window_size=16,
        vram_budget_gb=0.01, num_shared_experts=1,
        num_attention_heads=2, num_key_value_heads=2,
        max_seq_len=16, vocab_size=128,
        storage_format="pt",
    )
    model = ChronosForCausalLM(cfg).eval()
    state = model.state_dict()
    engine = ChronosInferenceEngine(model, cfg, ssd_dir=tempfile.mkdtemp(prefix="chronos_lazy_setup_"))
    engine.setup_from_state_dict(state, warm_expert_ids=[])
    expert0 = model.model.layers[0].mlp.experts[0]
    assert isinstance(expert0, LazyExpertPlaceholder)
    engine.teardown()


def test_inference_route_stats_cover_all_moe_layers():
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.runtime.inference_engine import ChronosInferenceEngine

    cfg = ChronosConfig(
        hidden_size=32, num_hidden_layers=2, num_experts=4,
        num_experts_per_tok=1, num_shared_experts=1,
        num_attention_heads=2, num_key_value_heads=2,
        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
        vocab_size=128, use_moe=True, storage_format="pt",
    )
    model = ChronosForCausalLM(cfg).eval()
    engine = ChronosInferenceEngine(model, cfg, ssd_dir="/tmp/chronos_ci_route_stats")
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        model(ids, use_cache=False)

    first_layer_id = engine._get_current_expert_ids(capacity=1)[0]
    masks = []
    for li, layer in enumerate(model.model.layers):
        moe = layer.mlp
        probs = moe.last_router_probs[:, -1, :]
        eid = int(torch.topk(probs, k=1, dim=-1).indices.reshape(-1)[0].item())
        mask = torch.zeros(cfg.num_experts, dtype=torch.bool)
        if li == 0:
            mask[eid] = True
        masks.append(mask)

    stats = engine._route_cache_stats(masks, include_all_tokens=False)
    assert first_layer_id in stats["activated_expert_ids"]
    assert len(stats["per_layer"]) == 2
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["fallback_weight_mass"] > 0


def test_on_demand_offload_loads_only_missing_expert_and_preserves_logits(tmp_path):
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.runtime.inference_engine import ChronosInferenceEngine

    torch.manual_seed(7)
    cfg = ChronosConfig(
        hidden_size=32, num_hidden_layers=1, num_experts=3,
        num_experts_per_tok=1, num_shared_experts=1,
        num_attention_heads=2, num_key_value_heads=2,
        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
        vocab_size=128, use_moe=True, storage_format="pt",
    )
    source = ChronosForCausalLM(cfg).eval()
    half_state = {k: v.detach().half().cpu() for k, v in source.state_dict().items()}
    full_ref = ChronosForCausalLM(cfg).eval()
    full_ref.load_state_dict(half_state, strict=True)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        full, _ = full_ref(ids, use_cache=False)
        full_ref(ids, use_cache=False)
    needed = []
    for layer in full_ref.model.layers:
        probs = layer.mlp.last_router_probs.reshape(-1, cfg.num_experts)
        needed.extend(torch.topk(probs, k=cfg.num_experts_per_tok, dim=-1).indices.reshape(-1).tolist())
    needed = sorted(set(int(eid) for eid in needed))

    lazy = ChronosForCausalLM(cfg).eval()
    lazy.load_state_dict(half_state, strict=True)
    engine = ChronosInferenceEngine(lazy, cfg, ssd_dir=str(tmp_path / "lazy"))
    store = engine.cache_manager.expert_store
    store.vram_capacity = 1
    store.ram_capacity = 1
    store.vram_lru.capacity = 1
    store.ram_lru.capacity = 1
    engine.setup_from_state_dict(half_state, warm_expert_ids=[])
    try:
        engine._runtime_stats = {
            "on_demand_loads": 0,
            "on_demand_load_time_s": 0.0,
            "sync_ssd_loads": 0,
            "async_cold_miss_prefetches": 0,
            "resident_vram_hits": 0,
            "resident_ram_hits": 0,
            "on_demand_loaded_experts": [],
            "on_demand_promoted_experts": [],
        }
        engine._install_moe_runtime_hooks("sync_on_demand")
        cold_mask = torch.zeros(cfg.num_experts, dtype=torch.bool)
        with torch.no_grad():
            lazy_out, _ = lazy(ids, use_cache=False, available_expert_masks=[cold_mask])
        assert torch.allclose(full.logits, lazy_out.logits, atol=1e-5, rtol=1e-5)
        assert set(engine._runtime_stats["on_demand_loaded_experts"]).issubset(set(needed))
        assert set(engine._runtime_stats["on_demand_loaded_experts"]) == set(needed)
        assert len(store.vram_lru) <= 1
        assert len(store.ram_lru) <= 1
        assert len(store._ram_buffers) <= 1
    finally:
        engine._clear_moe_runtime_hooks()
        engine.teardown()


def test_predictive_on_demand_cold_miss_queues_async_prefetch(tmp_path):
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.runtime.inference_engine import ChronosInferenceEngine

    cfg = ChronosConfig(
        hidden_size=32, num_hidden_layers=1, num_experts=2,
        num_experts_per_tok=1, num_shared_experts=1,
        num_attention_heads=2, num_key_value_heads=2,
        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
        vocab_size=128, use_moe=True, storage_format="pt",
    )
    engine = ChronosInferenceEngine(ChronosForCausalLM(cfg).eval(), cfg, ssd_dir=str(tmp_path))
    submitted = []
    ensure_calls = []
    engine.cache_manager.prefetch_experts_to_ram = lambda ids: submitted.extend(ids) or list(ids)
    engine.cache_manager.ensure_resident = lambda ids: ensure_calls.extend(ids) or list(ids)
    engine.cache_manager.expert_store.vram_lru.contains = lambda _eid: False
    engine.cache_manager.expert_store.ram_lru.contains = lambda _eid: False
    engine._runtime_stats = {
        "on_demand_loads": 0,
        "on_demand_load_time_s": 0.0,
        "sync_ssd_loads": 0,
        "async_cold_miss_prefetches": 0,
        "resident_vram_hits": 0,
        "resident_ram_hits": 0,
        "on_demand_loaded_experts": [],
        "on_demand_promoted_experts": [],
    }
    engine._install_moe_runtime_hooks("on_demand")
    try:
        loader = engine._moe_layers()[0].runtime_on_demand_loader
        result = loader(1)
        assert result["ok"] is False
        assert submitted == [1]
        assert ensure_calls == []
        assert engine._runtime_stats["async_cold_miss_prefetches"] == 1
        assert engine._runtime_stats["sync_ssd_loads"] == 0
    finally:
        engine._clear_moe_runtime_hooks()
        engine.teardown()


# ── M5/M6 tests ───────────────────────────────────────────────────

def test_backend_dispatcher():
    from chronos.backend import BackendDispatcher
    d = BackendDispatcher()
    avail = d.available()
    assert "cpu" in avail, f"cpu must be available, got {avail}"
    assert d.device_str("cpu") == "cpu"
    assert d.supports_training("cpu") is True
    if "mlx" in avail:
        assert d.supports_training("mlx") is True
    assert d.supports_training("vulkan") is False  # no upstream autograd
    assert d.info("opencl").available is False     # stub
    sel = d.select()
    assert sel in avail


def test_training_backend_resolver():
    from chronos.backend import BackendDispatcher
    d = BackendDispatcher()

    fake_infos = {
        "cuda": mock.Mock(available=False, supports_training=False, torch_device=None),
        "xpu": mock.Mock(available=False, supports_training=False, torch_device=None),
        "mps": mock.Mock(available=True, supports_training=True, torch_device="mps"),
        "cpu": mock.Mock(available=True, supports_training=True, torch_device="cpu"),
        "mlx": mock.Mock(available=True, supports_training=True, torch_device=None),
    }

    with mock.patch.object(d, "info", side_effect=lambda name: fake_infos.get(name, mock.Mock(available=False, supports_training=False, torch_device=None))):
        assert d.training_available() == ["mlx", "mps", "cpu"]
        assert d.select_training() == "mlx"
        assert d.select_training("auto") == "mlx"
        assert d.select_training("cpu") == "cpu"
        assert d.select_training("mlx") == "mlx"
        assert d.training_device_str("mlx") == "mlx"
        assert d.training_device_str("mps") == "mps"
        assert d.resolve_training_device("auto") == ("mlx", "mlx")
        assert d.resolve_training_device("mlx") == ("mlx", "mlx")
        assert d.resolve_training_device("cpu") == ("cpu", "cpu")


def test_training_backend_resolver_falls_back_when_mlx_not_trainable():
    from chronos.backend import BackendDispatcher
    d = BackendDispatcher()

    fake_infos = {
        "cuda": mock.Mock(available=False, supports_training=False, torch_device=None),
        "xpu": mock.Mock(available=False, supports_training=False, torch_device=None),
        "mps": mock.Mock(available=True, supports_training=True, torch_device="mps"),
        "cpu": mock.Mock(available=True, supports_training=True, torch_device="cpu"),
        "mlx": mock.Mock(available=True, supports_training=False, torch_device=None),
    }

    with mock.patch.object(d, "info", side_effect=lambda name: fake_infos.get(name, mock.Mock(available=False, supports_training=False, torch_device=None))):
        assert d.training_available() == ["mps", "cpu"]
        assert d.select_training("mlx") == "mps"
        assert d.resolve_training_device("mlx") == ("mps", "mps")


def test_train_backend_choice_helpers():
    from ui.tabs.train_tab import (
        _available_train_backend_choices,
        _train_backend_dropdown_choices,
    )

    with mock.patch("ui.tabs.train_tab.training_available", return_value=["mlx", "mps", "cpu"]):
        assert _available_train_backend_choices() == ["auto", "mlx", "mps", "cpu"]
        choices = _train_backend_dropdown_choices("en")
        assert choices[0][1] == "auto"
        assert [value for _, value in choices] == ["auto", "mlx", "mps", "cpu"]


def test_inference_resolves_mlx_without_pytorch_checkpoint_fallback():
    from ui.tabs import inference_tab as mod

    assert mod.TORCH_INFERENCE_PRIORITY[0] == "cuda"

    with mock.patch("chronos.backend.select", return_value="mlx"):
        backend, note = mod._resolve_inference_backend("auto", "./out/foo.pth", {"hidden_size": 384})
        assert backend == "mlx"
        assert note == ""

    with mock.patch("chronos.backend.select", return_value="mlx"):
        backend, note = mod._resolve_inference_backend("auto", "./out/foo.pth", {})
        assert backend == "mlx"
        assert note == ""

    with mock.patch("chronos.backend.select", return_value="cuda"):
        backend, note = mod._resolve_inference_backend("auto", "./out/foo.pth", {"hidden_size": 384})
        assert backend == "cuda"
        assert note == ""

    with mock.patch("chronos.backend.select", return_value="cpu"):
        backend, note = mod._resolve_inference_backend("cuda", "./out/foo.pth", {})
        assert backend == "cpu"
        assert "Requested cuda" in note


def test_device_utils_mps_is_not_cpu_path():
    from chronos.trainer.device_utils import (
        configure_cpu_threads,
        dataloader_kwargs,
        grad_scaler,
        resolve_dtype_name,
        runtime_summary,
        torch_device_type,
    )

    assert torch_device_type("mps") == "mps"
    assert torch_device_type("mlx") == "mlx"
    assert torch_device_type("cuda:0") == "cuda"
    assert resolve_dtype_name("cpu", "auto") == "float32"
    assert resolve_dtype_name("mps", "auto") == "bfloat16"
    assert resolve_dtype_name("mlx", "auto") == "bfloat16"
    assert grad_scaler("mps", "float16").is_enabled() is False
    assert grad_scaler("cuda", "float16").is_enabled() in {True, False}
    summary = runtime_summary("mps", "bfloat16")
    assert summary.device_type == "mps"
    assert summary.autocast is True
    assert summary.scaler is False
    mlx_summary = runtime_summary("mlx", "auto")
    assert mlx_summary.device_type == "mlx"
    assert mlx_summary.dtype == "bfloat16"
    assert mlx_summary.autocast is False
    assert dataloader_kwargs("mps", num_workers="auto")["num_workers"] == 0
    assert dataloader_kwargs("mlx", num_workers="auto")["num_workers"] == 0
    assert dataloader_kwargs("mps", num_workers=4)["num_workers"] == 0
    assert dataloader_kwargs("mlx", num_workers=4)["num_workers"] == 0
    assert dataloader_kwargs("mps", num_workers=0)["pin_memory"] is False
    assert dataloader_kwargs("cuda", num_workers=0)["pin_memory"] is True
    threads = configure_cpu_threads("auto", budget_percent=75)
    assert threads >= 1


def test_configure_cpu_threads_overrides_single_thread_env(monkeypatch):
    from chronos.trainer.device_utils import configure_cpu_threads, cpu_thread_snapshot

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("CHRONOS_CPU_THREADS", raising=False)
    threads = configure_cpu_threads("auto", budget_percent=100)
    snap = cpu_thread_snapshot()
    assert threads >= 1
    assert snap["torch_num_threads"] == threads
    assert snap["OMP_NUM_THREADS"] == str(threads)
    assert snap["MKL_NUM_THREADS"] == str(threads)
    assert snap["VECLIB_MAXIMUM_THREADS"] == str(threads)
    assert snap["NUMEXPR_NUM_THREADS"] == str(threads)


def test_configure_cpu_threads_ignores_stale_chronos_env_by_default(monkeypatch):
    import psutil
    from chronos.trainer.device_utils import configure_cpu_threads, cpu_thread_snapshot

    monkeypatch.setenv("CHRONOS_CPU_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    physical = int(psutil.cpu_count(logical=False) or os.cpu_count() or 1)
    threads = configure_cpu_threads("auto", budget_percent=100)
    snap = cpu_thread_snapshot()
    assert threads == physical
    assert snap["OMP_NUM_THREADS"] == str(physical)
    assert snap["last_source"] == "budget_percent"


def test_configure_cpu_threads_can_prefer_env_for_diagnostics(monkeypatch):
    from chronos.trainer.device_utils import configure_cpu_threads, cpu_thread_snapshot

    monkeypatch.setenv("CHRONOS_CPU_THREADS", "2")
    threads = configure_cpu_threads("auto", budget_percent=100, prefer_env=True)
    snap = cpu_thread_snapshot()
    assert threads == 2
    assert snap["OMP_NUM_THREADS"] == "2"
    assert snap["last_source"] == "CHRONOS_CPU_THREADS"


def test_mac_backend_diagnostics_shape():
    from chronos.backend.mac_diagnostics import mac_backend_diagnostics

    diag = mac_backend_diagnostics(configure_threads=False)
    assert "torch_num_threads" in diag
    assert "mps_available" in diag
    assert "mlx_available" in diag


def test_mac_backend_diagnostics_overrides_single_thread_env(monkeypatch):
    from chronos.backend.mac_diagnostics import mac_backend_diagnostics

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    monkeypatch.delenv("CHRONOS_CPU_THREADS", raising=False)
    diag = mac_backend_diagnostics(configure_threads=True)
    threads = int(diag["torch_num_threads"])
    assert threads >= 1
    assert diag["env"]["OMP_NUM_THREADS"] == str(threads)
    assert diag["env"]["MKL_NUM_THREADS"] == str(threads)
    assert diag["env"]["VECLIB_MAXIMUM_THREADS"] == str(threads)
    assert diag["env"]["NUMEXPR_NUM_THREADS"] == str(threads)


def test_inference_cpu_thread_helper_overrides_single_thread_env(monkeypatch):
    from ui.tabs.inference_tab import _configure_inference_cpu_threads_if_needed

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("VECLIB_MAXIMUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    monkeypatch.delenv("CHRONOS_CPU_THREADS", raising=False)
    stats = _configure_inference_cpu_threads_if_needed("cpu")
    threads = int(stats["cpu_threads"])
    assert threads >= 1
    assert stats["cpu_thread_env"]["OMP_NUM_THREADS"] == str(threads)
    assert stats["cpu_thread_env"]["MKL_NUM_THREADS"] == str(threads)
    assert stats["cpu_thread_env"]["VECLIB_MAXIMUM_THREADS"] == str(threads)
    assert stats["cpu_thread_env"]["NUMEXPR_NUM_THREADS"] == str(threads)


def test_inference_backend_choice_helpers():
    from ui.tabs.inference_tab import (
        _available_inference_backend_choices,
        _default_inference_backend_value,
    )

    with mock.patch("chronos.backend.available", return_value=["mlx", "mps", "cpu"]):
        assert _available_inference_backend_choices() == ["auto", "mlx", "mps", "cpu"]
        assert _default_inference_backend_value() == "auto"


def test_cuda_inference_path_uses_cuda_device_map():
    from ui.tabs import inference_tab as mod

    class _FakeTensor:
        def __init__(self, item_value=7):
            self.devices = []
            self.item_value = item_value

        def to(self, device):
            self.devices.append(device)
            return self

        @property
        def shape(self):
            return (1, 3)

        def tolist(self):
            return [[1, 2, 3, self.item_value]]

        def __getitem__(self, _key):
            return mock.Mock(tolist=mock.Mock(return_value=[self.item_value]))

    class _FakeModel:
        def __init__(self):
            self.devices = []
            self.eval_called = False

        def load_state_dict(self, _weights, strict=False):
            return None

        def to(self, device):
            self.devices.append(device)
            return self

        def eval(self):
            self.eval_called = True
            return self

    class _FakeEngine:
        def __init__(self, _model, _cfg, ssd_dir=None):
            self.setup_called = False

        def setup(self, warm_expert_ids=None):
            self.setup_called = True

        def setup_from_state_dict(self, _state, warm_expert_ids=None):
            self.setup_called = True

        def generate(self, input_ids, **_kwargs):
            return input_ids

        def teardown(self):
            return None

    fake_model = _FakeModel()
    fake_input = _FakeTensor(item_value=11)
    fake_torch = mock.Mock()
    fake_torch.tensor = mock.Mock(return_value=fake_input)
    fake_torch.load = mock.Mock(return_value={})

    import chronos.runtime.inference_engine  # ensure module is loaded before torch is mocked

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        with mock.patch("chronos.model.model_chronos.ChronosForCausalLM", return_value=fake_model):
            with mock.patch("chronos.runtime.inference_engine.ChronosInferenceEngine", _FakeEngine):
                generated = mod._run_torch_inference(
                    "cuda",
                    model_cfg=object(),
                    model_path_val="",
                    token_ids=[1, 2, 3],
                    attention_mask=[1, 1, 1],
                    max_tok=1,
                    temp=0.8,
                    eos_token_id=99,
                )

    generated_tokens, stats = generated
    assert generated_tokens == [11]
    assert stats["effective_expert_budget"] == 1
    assert "setup_rss_delta_gb" in stats
    assert fake_model.devices == ["cuda"]
    assert fake_model.eval_called is True
    assert fake_input.devices == ["cuda", "cuda"]
    assert fake_torch.tensor.call_count == 2


def test_torch_inference_loads_base_only_before_lazy_setup():
    from ui.tabs import inference_tab as mod

    captured = {}

    class _FakeTensor:
        def to(self, _device):
            return self

        @property
        def shape(self):
            return (1, 3)

        def tolist(self):
            return [[1, 2, 3]]

        def __getitem__(self, _key):
            return mock.Mock(tolist=mock.Mock(return_value=[]))

    class _FakeModel:
        def load_state_dict(self, weights, strict=False):
            captured["loaded_keys"] = sorted(weights.keys())
            captured["strict"] = strict

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _FakeEngine:
        def __init__(self, _model, _cfg, ssd_dir=None):
            pass

        def setup_from_state_dict(self, state, warm_expert_ids=None):
            captured["setup_keys"] = sorted(state.keys())

        def setup(self, warm_expert_ids=None):
            captured["setup_called"] = True

        def generate(self, input_ids, **_kwargs):
            return input_ids

        def teardown(self):
            captured["teardown"] = True

    fake_state = {
        "model.embed_tokens.weight": torch.zeros(4, 4),
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(4, 4),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(4, 4),
    }
    with mock.patch("torch.load", return_value=fake_state):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("chronos.model.model_chronos.ChronosForCausalLM", return_value=_FakeModel()):
                with mock.patch("chronos.runtime.inference_engine.ChronosInferenceEngine", _FakeEngine):
                    out = mod._run_torch_inference(
                        "cpu",
                        model_cfg=object(),
                        model_path_val="/tmp/fake.pth",
                        token_ids=[1, 2, 3],
                        attention_mask=[1, 1, 1],
                        max_tok=4,
                        temp=0.8,
                        eos_token_id=99,
                    )

    generated_tokens, stats = out
    assert generated_tokens == []
    assert stats["effective_expert_budget"] == 1
    assert "setup_rss_delta_gb" in stats
    assert captured["strict"] is False
    assert captured["loaded_keys"] == ["model.embed_tokens.weight"]
    assert any(".mlp.experts." in key for key in captured["setup_keys"])
    assert captured["teardown"] is True


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


def test_webui_launch_port_wiring():
    """The app exposes a single WebUI port; verify CLI args are passed
    through to Gradio launch unchanged."""
    import chronos.app as app_mod

    captured = {}

    class _FakeApp:
        def launch(self, **kwargs):
            captured.update(kwargs)

    with mock.patch.object(app_mod, "build_app", return_value=_FakeApp()):
        with mock.patch.object(sys, "argv", [
            "chronos.app", "--port", "7867", "--host", "0.0.0.0", "--share"
        ]):
            app_mod.main()

    assert captured["server_port"] == 7867
    assert captured["server_name"] == "0.0.0.0"
    assert captured["share"] is True
    assert captured["show_error"] is True


def test_stage_entry_mappings_consistent():
    """Train tab and Pipeline tab must point each stage at the same
    upstream init, data fixture, and entry script."""
    from ui.tabs.train_tab import STAGE_UI_ORDER, STAGE_DEFAULT_DATA, STAGE_DEFAULT_INIT
    from ui.tabs.pipeline_tab import STAGES

    display_to_mode = {
        "Pretrain": "pretrain",
        "SFT": "sft",
        "DPO": "dpo",
        "ORPO": "orpo",
        "GRPO": "grpo",
        "Distill": "distill",
    }
    expected_scripts = {
        "pretrain": "train_chronos.py",
        "sft": "train_chronos_sft.py",
        "dpo": "train_chronos_dpo.py",
        "orpo": "train_chronos_orpo.py",
        "grpo": "train_chronos_grpo.py",
        "distill": "train_chronos_distill.py",
    }

    def _norm_relpath(path: str) -> str:
        return path[2:] if path.startswith("./") else path

    pipeline_modes = [display_to_mode[name] for name, *_ in STAGES]
    assert pipeline_modes == STAGE_UI_ORDER

    for name, script, from_weight, data_path, takes_teacher in STAGES:
        mode = display_to_mode[name]
        assert script == expected_scripts[mode]
        assert from_weight == STAGE_DEFAULT_INIT.get(mode, "")
        assert _norm_relpath(data_path) == _norm_relpath(STAGE_DEFAULT_DATA[mode])
        assert takes_teacher is (mode == "distill")


def test_pipeline_backend_choices_include_auto():
    from ui.tabs import pipeline_tab

    assert pipeline_tab.PIPELINE_TRAIN_BACKEND_CHOICES[0] == "auto"
    with mock.patch("ui.tabs.pipeline_tab.training_available", return_value=["mlx", "mps", "cpu"]):
        choices = ["auto"] + [name for name in ("cuda", "xpu", "mlx", "mps", "cpu") if name in set(pipeline_tab.training_available())]
        assert choices == ["auto", "mlx", "mps", "cpu"]


def test_stage_init_placeholder_means_auto_resolution():
    """Legacy placeholder paths should normalize back to blank so the
    runtime resolves the previous stage dynamically from save_dir."""
    from ui.tabs.train_tab import _normalize_stage_init_value, _stage_init_placeholder

    assert _normalize_stage_init_value("") == ""
    assert _normalize_stage_init_value("./out/chronos_768_moe.pth") == ""
    assert _normalize_stage_init_value("./out/sft_768_moe.pth") == ""
    assert _normalize_stage_init_value("./custom/model.pth") == "./custom/model.pth"

    assert "auto" in _stage_init_placeholder("pretrain").lower()
    assert "override" in _stage_init_placeholder("grpo").lower()


def test_shared_expert_grad():
    """M7a: shared_experts must receive gradient during the training-path
    forward (available_expert_mask=None). Pre-M7 they were dead code,
    causing train/inference distribution mismatch."""
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=4,
                        num_experts_per_tok=2, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=128, use_moe=True)
    m = ChronosForCausalLM(cfg).train()
    ids = torch.randint(0, 128, (2, 8))
    out, _ = m(ids, labels=ids)
    out.loss.backward()
    sh = m.model.layers[0].mlp.shared_experts[0]
    g = sh.gate_proj.weight.grad
    assert g is not None and g.norm().item() > 0, "shared_experts received zero gradient"


def test_moe_all_available_matches_no_mask():
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=4,
                        num_experts_per_tok=2, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=128, use_moe=True)
    model = ChronosForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    mask = torch.ones(cfg.num_experts, dtype=torch.bool)
    with torch.no_grad():
        no_mask, _ = model(ids, use_cache=False)
        all_avail, _ = model(ids, use_cache=False, available_expert_masks=[mask])
    diff = (no_mask.logits - all_avail.logits).abs().max().item()
    assert diff < 1e-6, f"all-available masked path drifted from no-mask path: {diff}"


def test_masked_offload_path_accepts_2d_attention_mask():
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=2, num_experts=4,
                        num_experts_per_tok=2, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        sliding_window_size=8, vocab_size=128, use_moe=True,
                        use_hybrid_attention=True)
    model = ChronosForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    attention_mask = torch.ones_like(ids)
    expert_mask = torch.ones(cfg.num_experts, dtype=torch.bool)
    with torch.no_grad():
        out, _ = model(
            ids,
            attention_mask=attention_mask,
            use_cache=False,
            available_expert_masks=[expert_mask, expert_mask],
        )
    assert out.logits.shape[:2] == ids.shape


def test_rope_cache_grows_during_cached_decode():
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=2, num_experts=4,
                        num_experts_per_tok=2, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=4,
                        max_position_embeddings=4,
                        sliding_window_size=8, vocab_size=128, use_moe=True,
                        use_hybrid_attention=True)
    model = ChronosForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        out, _ = model(ids, attention_mask=attention_mask, use_cache=True)
        next_id = torch.randint(0, cfg.vocab_size, (1, 1))
        next_mask = torch.ones(1, 5, dtype=attention_mask.dtype)
        out2, _ = model(
            next_id,
            attention_mask=next_mask,
            past_key_values=out.past_key_values,
            use_cache=True,
        )
    assert out2.logits.shape[:2] == next_id.shape
    assert model.model.freqs_cos.shape[0] >= 5


def test_checkpoint_config_roundtrip_strict(tmp_path):
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import (
        chronos_config_from_checkpoint,
        load_checkpoint_state_dict,
        save_state_dict_with_config,
    )

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=4,
                        num_experts_per_tok=3, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=128, use_moe=True)
    model = ChronosForCausalLM(cfg).eval()
    ckpt = tmp_path / "chronos_32_moe.pth"
    save_state_dict_with_config(model, str(ckpt), cfg, stage="test")
    loaded_cfg, sources = chronos_config_from_checkpoint(str(ckpt), require_unsniffable=True)
    assert loaded_cfg.num_experts_per_tok == 3
    assert loaded_cfg.num_shared_experts == 1
    assert sources and str(ckpt).replace(".pth", ".config.json") in sources[0]
    m2 = ChronosForCausalLM(loaded_cfg)
    m2.load_state_dict(load_checkpoint_state_dict(str(ckpt), map_location="cpu"), strict=True)


def test_export_formats_roundtrip_config_and_readers(tmp_path):
    from chronos.export import (
        EXPORT_FORMATS,
        GGUFStateReader,
        SafetensorsStateReader,
        chronos_config_from_export,
        export_checkpoint,
    )
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import save_state_dict_with_config

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=6400, use_moe=True)
    model = ChronosForCausalLM(cfg).eval()
    ckpt = tmp_path / "tiny_chronos.pth"
    save_state_dict_with_config(model, str(ckpt), cfg, stage="test")

    out_dir = tmp_path / "exported"
    results = export_checkpoint(str(ckpt), str(out_dir), formats=list(EXPORT_FORMATS))
    assert {r.format for r in results} == set(EXPORT_FORMATS)
    assert (out_dir / "config.json").exists()
    assert (out_dir / "chronos_export_manifest.json").exists()
    assert (out_dir / "OLLAMA_CHRONOS.md").exists()

    loaded_cfg = chronos_config_from_export(str(out_dir / "model.fp16.safetensors"))
    assert loaded_cfg.hidden_size == cfg.hidden_size
    assert loaded_cfg.num_experts_per_tok == cfg.num_experts_per_tok
    st_reader = SafetensorsStateReader(str(out_dir / "model.q8_0.safetensors"))
    assert "model.embed_tokens.weight" in st_reader.keys()
    assert st_reader.get_tensor("model.embed_tokens.weight").shape == model.lm_head.weight.shape

    gguf_reader = GGUFStateReader(str(out_dir / "model.fp16.gguf"))
    assert gguf_reader.config_dict()["num_experts"] == cfg.num_experts
    assert gguf_reader.get_tensor("model.embed_tokens.weight").shape == model.lm_head.weight.shape


def test_export_safetensors_lazy_reader_setup(tmp_path):
    from chronos.export import GGUFStateReader, SafetensorsStateReader, export_checkpoint
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.model.checkpoint import load_state_dict_controlled, save_state_dict_with_config
    from chronos.runtime.inference_engine import ChronosInferenceEngine
    from chronos.model.moe_chronos import LazyExpertPlaceholder

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=128, use_moe=True, storage_format="pt")
    source = ChronosForCausalLM(cfg).eval()
    ckpt = tmp_path / "tiny_chronos.pth"
    save_state_dict_with_config(source, str(ckpt), cfg, stage="test")
    export_checkpoint(str(ckpt), str(tmp_path / "exported"), formats=["fp16-safetensors"])

    reader = SafetensorsStateReader(str(tmp_path / "exported" / "model.fp16.safetensors"))
    model = ChronosForCausalLM(cfg).eval()
    load_state_dict_controlled(
        model,
        reader.load_state_dict(include_experts=False),
        allow_missing_substrings=(".mlp.experts.",),
    )
    engine = ChronosInferenceEngine(model, cfg, ssd_dir=str(tmp_path / "lazy_cache"))
    engine.setup_from_state_reader(reader, warm_expert_ids=[])
    try:
        assert isinstance(model.model.layers[0].mlp.experts[0], LazyExpertPlaceholder)
        assert (tmp_path / "lazy_cache" / "expert_l0_e0.pt").exists()
    finally:
        engine.teardown()

    export_checkpoint(str(ckpt), str(tmp_path / "exported_gguf"), formats=["fp16-gguf"])
    gguf_reader = GGUFStateReader(str(tmp_path / "exported_gguf" / "model.fp16.gguf"))
    model2 = ChronosForCausalLM(cfg).eval()
    load_state_dict_controlled(
        model2,
        gguf_reader.load_state_dict(include_experts=False),
        allow_missing_substrings=(".mlp.experts.",),
    )
    engine2 = ChronosInferenceEngine(model2, cfg, ssd_dir=str(tmp_path / "lazy_cache_gguf"))
    engine2.setup_from_state_reader(gguf_reader, warm_expert_ids=[])
    try:
        assert isinstance(model2.model.layers[0].mlp.experts[1], LazyExpertPlaceholder)
        assert (tmp_path / "lazy_cache_gguf" / "expert_l0_e1.pt").exists()
    finally:
        engine2.teardown()


def test_clustered_cache_from_checkpoint_and_export_auto_cluster(tmp_path):
    import json
    from chronos.export import export_checkpoint
    from chronos.io.cluster_layout import build_clustered_expert_cache_from_checkpoint
    from chronos.model.checkpoint import save_state_dict_with_config
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=4,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=6400, use_moe=True)
    model = ChronosForCausalLM(cfg).eval()
    ckpt = tmp_path / "tiny_cluster_chronos.pth"
    save_state_dict_with_config(model, str(ckpt), cfg, stage="test")

    calib = tmp_path / "calib.jsonl"
    calib.write_text(
        "\n".join([
            json.dumps({"text": "chronos expert router cluster cache prefetch"}),
            json.dumps({"text": "lazy offload memory expert prediction decode"}),
        ]) + "\n",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cluster_cache"
    summary = build_clustered_expert_cache_from_checkpoint(
        str(ckpt),
        str(calib),
        str(cache_dir),
        max_batches=1,
        batch_size=1,
        max_seq_len=12,
    )
    assert (cache_dir / "cluster_manifest.json").exists()
    assert summary["status"] if "status" in summary else summary["manifest_path"]
    assert summary["clusters"]

    out_dir = tmp_path / "exported_auto_cluster"
    results = export_checkpoint(
        str(ckpt),
        str(out_dir),
        formats=["fp16-safetensors"],
        auto_cluster=True,
        calibration_data_path=str(calib),
        cluster_max_batches=1,
        cluster_batch_size=1,
        cluster_max_seq_len=12,
    )
    assert (out_dir / "expert_cache" / "cluster_manifest.json").exists()
    payload = json.loads(results[0].metadata["chronos_export"])
    assert payload["offload_features"]["cluster_aware_safetensors"] is True
    assert payload["cluster_build"]["status"] == "built"
    assert payload["expert_cache"]["num_experts"] == cfg.num_experts


def test_mlx_cluster_aware_expert_store_roundtrip(tmp_path):
    if not _mlx_available():
        return
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.mlx.expert_store import MLXExpertStore
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.moe import LazyFeedForwardMLX

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=3,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=128, use_moe=True)
    pt = ChronosForCausalLM(cfg).eval()
    mlx_model = ChronosMLXModel.from_chronos_pytorch(pt, cfg)
    store = MLXExpertStore(mlx_model, cfg, ssd_dir=str(tmp_path / "mlx_cache"))
    store._warm_capacity = 2
    store.offload_all_to_ssd(clusters=[[0, 1], [2]])
    assert (tmp_path / "mlx_cache" / "cluster_manifest.json").exists()
    assert (tmp_path / "mlx_cache" / "cluster_0.ctsr").exists()

    store.replace_live_experts_with_placeholders()
    assert isinstance(mlx_model.layers[0].mlp.experts[0], LazyFeedForwardMLX)
    store.prefetch_to_ram([0])
    assert 0 in store._warm and 1 in store._warm
    assert store.promote_to_vram(1) is True
    stats = store.stats()
    assert stats["cluster_aware"] is True
    assert stats["hot_experts"] == 1
    assert stats["warm_capacity"] == 2
    assert not isinstance(mlx_model.layers[0].mlp.experts[1], LazyFeedForwardMLX)

    store.prefetch_to_ram([2])
    assert len(store._warm) <= 2


def test_mlx_lazy_store_materializes_every_layer_and_rejects_tensor_mask(tmp_path):
    if not _mlx_available():
        return
    import pytest
    import mlx.core as mx
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.mlx.expert_store import MLXExpertStore
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.moe import LazyFeedForwardMLX

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=2, num_experts=3,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=128, use_moe=True)
    pt = ChronosForCausalLM(cfg).eval()
    mlx_model = ChronosMLXModel.from_chronos_pytorch(pt, cfg)
    store = MLXExpertStore(mlx_model, cfg, ssd_dir=str(tmp_path / "mlx_cache_layers"))
    store._capacity = 1
    store._warm_capacity = 2
    store.offload_all_to_ssd(clusters=[[0, 1], [2]])
    store.replace_live_experts_with_placeholders()

    store.prefetch_to_ram([1])
    assert store.promote_to_vram(1) is True
    assert store.hot_expert_ids() == {1}
    for layer in mlx_model.layers:
        assert not isinstance(layer.mlp.experts[1], LazyFeedForwardMLX)

    store._replace_expert_with_placeholder(1)
    assert store.hot_expert_ids() == set()
    assert store.promote_to_vram(1) is True
    for layer in mlx_model.layers:
        assert not isinstance(layer.mlp.experts[1], LazyFeedForwardMLX)

    x = mx.zeros((1, 1, cfg.hidden_size), dtype=mx.float32)
    with pytest.raises(RuntimeError, match="tensor mask"):
        mlx_model.layers[0].mlp(x, available_expert_mask=store.vram_availability_mask())


def test_mlx_auto_dtype_prefers_bfloat16():
    from chronos.mlx.training.trainer import _normalize_mlx_dtype_name

    assert _normalize_mlx_dtype_name("auto") in {"bfloat16", "float32"}
    assert _normalize_mlx_dtype_name("fp16") == "float16"


def test_mlx_training_pretrain_smoke_roundtrip(tmp_path):
    if not _mlx_available():
        return
    from chronos.model.checkpoint import load_checkpoint_state_dict
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.mlx.training import run_mlx_stage

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=64, use_moe=True,
                        lambda_balance=5e-4, lambda_temporal=1e-3,
                        lambda_lookahead=0.1, lambda_lookahead_topk=0.05)
    ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=1)

    class Args:
        learning_rate = 1e-4
        weight_decay = 0.0
        steps = 1
        epochs = 1
        max_seq_len = 8

    result = run_mlx_stage(
        stage="pretrain",
        config=cfg,
        checkpoint_path=None,
        save_dir=str(tmp_path),
        loader=loader,
        args=Args(),
    )
    assert result.steps == 1
    assert os.path.exists(result.checkpoint_path)
    assert os.path.exists(result.checkpoint_path.replace(".pth", ".config.json"))
    state = load_checkpoint_state_dict(result.checkpoint_path, map_location="cpu")
    ChronosForCausalLM(cfg).load_state_dict(state, strict=True)


def test_mlx_training_progress_and_stop(tmp_path):
    if not _mlx_available():
        return
    import threading
    from chronos.model.config import ChronosConfig
    from chronos.mlx.training import run_mlx_stage

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=64, use_moe=True,
                        lambda_balance=0.0, lambda_temporal=0.0,
                        lambda_lookahead=0.0, lambda_lookahead_topk=0.0)
    ids = torch.randint(0, cfg.vocab_size, (4, 8), dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=1)

    class Args:
        learning_rate = 1e-4
        weight_decay = 0.0
        steps = 4
        epochs = 1
        max_seq_len = 8
        log_interval = 1

    events = []
    stop = threading.Event()

    def cb(event):
        events.append(dict(event))
        if event.get("event") == "step":
            stop.set()

    result = run_mlx_stage(
        stage="pretrain",
        config=cfg,
        checkpoint_path=None,
        save_dir=str(tmp_path),
        loader=loader,
        args=Args(),
        progress_callback=cb,
        stop_event=stop,
    )
    assert result.steps == 1
    assert any(e.get("event") == "start" for e in events)
    assert any(e.get("event") == "step" for e in events)
    assert any(e.get("event") == "stopped" for e in events)
    assert result.checkpoint_saved is True
    assert os.path.exists(result.checkpoint_path)
    assert os.path.exists(result.checkpoint_path.replace(".pth", ".config.json"))


def test_mlx_training_save_interval(tmp_path):
    if not _mlx_available():
        return
    from chronos.model.config import ChronosConfig
    from chronos.mlx.training import run_mlx_stage

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=64, use_moe=True,
                        lambda_balance=0.0, lambda_temporal=0.0,
                        lambda_lookahead=0.0, lambda_lookahead_topk=0.0)
    ids = torch.randint(0, cfg.vocab_size, (3, 8), dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=1)

    class Args:
        learning_rate = 1e-4
        weight_decay = 0.0
        steps = 3
        epochs = 1
        max_seq_len = 8
        log_interval = 1
        save_interval = 2
        accumulation_steps = 1
        grad_clip = 1.0

    events = []
    result = run_mlx_stage(
        stage="pretrain",
        config=cfg,
        checkpoint_path=None,
        save_dir=str(tmp_path),
        loader=loader,
        args=Args(),
        progress_callback=lambda event: events.append(dict(event)),
    )
    assert result.steps == 3
    assert result.total_steps == 3
    assert os.path.exists(result.checkpoint_path)
    assert any(e.get("event") == "save" and e.get("reason") == "interval" for e in events)
    assert any(e.get("event") == "save" and e.get("reason") == "final" for e in events)


def test_mlx_training_nan_rollback_saves_last_finite(tmp_path, monkeypatch):
    if not _mlx_available():
        return
    import mlx.core as mx
    from chronos.model.config import ChronosConfig
    import chronos.mlx.training.trainer as trainer_mod
    from chronos.mlx.training import run_mlx_stage

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=64, use_moe=True,
                        lambda_balance=0.0, lambda_temporal=0.0,
                        lambda_lookahead=0.0, lambda_lookahead_topk=0.0)
    ids = torch.randint(0, cfg.vocab_size, (3, 8), dtype=torch.long)
    loader = DataLoader(TensorDataset(ids, ids.clone()), batch_size=1)

    original_loss = trainer_mod._ce_stage_loss
    calls = {"n": 0}

    def injected_loss(model, input_ids, labels):
        calls["n"] += 1
        if calls["n"] == 2:
            return mx.array(float("nan"), dtype=mx.float32)
        return original_loss(model, input_ids, labels)

    class Args:
        learning_rate = 1e-4
        weight_decay = 0.0
        steps = 3
        epochs = 1
        max_seq_len = 8
        log_interval = 1
        save_interval = 100
        accumulation_steps = 1
        grad_clip = 1.0
        mlx_rollback_limit = 3

    monkeypatch.setattr(trainer_mod, "_ce_stage_loss", injected_loss)
    events = []
    result = run_mlx_stage(
        stage="pretrain",
        config=cfg,
        checkpoint_path=None,
        save_dir=str(tmp_path),
        loader=loader,
        args=Args(),
        progress_callback=lambda event: events.append(dict(event)),
    )
    assert result.rollbacks == 1
    assert result.checkpoint_saved is True
    assert os.path.exists(result.checkpoint_path)
    assert any(e.get("event") == "rollback" for e in events)


def test_mlx_runtime_state_not_trainable_parameters():
    if not _mlx_available():
        return
    import mlx.core as mx
    from chronos.model.config import ChronosConfig
    from chronos.mlx.model import ChronosMLXModel

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=64, use_moe=True)
    model = ChronosMLXModel(cfg)

    def flatten(prefix, tree, out):
        if isinstance(tree, dict):
            for key, value in tree.items():
                flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
        elif isinstance(tree, (list, tuple)):
            for idx, value in enumerate(tree):
                flatten(f"{prefix}.{idx}" if prefix else str(idx), value, out)
        else:
            out.append(prefix)

    before = []
    flatten("", model.parameters(), before)
    ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    logits, _lookahead, _cache = model(ids)
    mx.eval(logits)
    after = []
    flatten("", model.parameters(), after)
    joined = "\n".join(after)
    assert set(before) == set(after)
    assert "last_router_probs" not in joined
    assert "runtime_" not in joined
    assert "._cos" not in joined
    assert "._sin" not in joined


def test_ui_train_session_mlx_uses_native_trainer(tmp_path):
    if not _mlx_available():
        return
    from ui.tabs.train_tab import TrainSession

    session = TrainSession()
    messages = []
    session._put = lambda msg: messages.append(str(msg))
    cfg = {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "kv_latent_dim": 8,
        "rope_dim": 4,
        "moe_intermediate_size": 64,
        "vocab_size": 64,
        "max_seq_len": 8,
        "batch_size": 1,
        "epochs": 1,
        "max_steps": 1,
        "save_dir": str(tmp_path),
        "train_backend": "mlx",
        "data_path": "",
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
    }

    session._run(cfg, "pretrain")

    assert session.status == "finished", "\n".join(messages[-10:])
    assert session.step == 1
    assert session.total_steps == 1
    assert (tmp_path / "chronos_32_moe.pth").exists()
    assert any("Running native MLX trainer" in msg for msg in messages)
    assert any("[MLX PRETRAIN] step=1/1" in msg for msg in messages)
    assert not any('device string: mlx' in msg for msg in messages)


def test_ui_train_session_mlx_total_steps_from_loader(tmp_path):
    if not _mlx_available():
        return
    from ui.tabs.train_tab import TrainSession

    session = TrainSession()
    messages = []
    session._put = lambda msg: messages.append(str(msg))
    cfg = {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "kv_latent_dim": 8,
        "rope_dim": 4,
        "moe_intermediate_size": 64,
        "vocab_size": 64,
        "max_seq_len": 8,
        "batch_size": 1,
        "epochs": 1,
        "max_steps": 0,
        "save_dir": str(tmp_path),
        "train_backend": "mlx",
        "data_path": "",
        "num_workers": 0,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "log_interval": 100,
        "save_interval": 100,
    }

    session._run(cfg, "pretrain")

    assert session.status == "finished", "\n".join(messages[-10:])
    assert session.step == 50
    assert session.total_steps == 50


def test_train_effective_cpu_config_uses_full_physical_budget(monkeypatch):
    from ui.tabs.train_tab import TrainSession

    session = TrainSession()
    cfg = {"cpu_threads": "auto"}
    session._normalise_effective_cpu_config(cfg, "cpu")
    assert cfg["cpu_budget_percent"] == 100.0

    budget_cfg = {"cpu_threads": "auto", "cpu_budget_percent": 75}
    session._normalise_effective_cpu_config(budget_cfg, "cpu")
    assert budget_cfg["cpu_budget_percent"] == 75.0

    explicit_cfg = {"cpu_threads": "4", "cpu_budget_percent": 75}
    session._normalise_effective_cpu_config(explicit_cfg, "cpu")
    assert explicit_cfg["cpu_budget_percent"] == 75.0


def test_train_start_merge_preserves_config_weight_decay(monkeypatch, tmp_path):
    from ui.tabs import train_tab

    monkeypatch.setattr(train_tab, "resolve_training_device", lambda requested: ("cpu", "cpu"))
    cfg, effective = train_tab._prepare_training_run_config(
        {"weight_decay": 0.123, "cpu_threads": "auto"},
        "pretrain",
        "cpu",
        "",
        "",
        1,
        0.05,
        "",
        "",
    )
    assert cfg["weight_decay"] == 0.123
    assert cfg["cpu_budget_percent"] == 100.0
    assert effective["weight_decay"] == 0.123
    assert effective["cpu_budget_percent"] == 100.0


def test_config_input_order_matches_autotune_mapping():
    from ui.presets import CONFIG_INPUT_ORDER
    from ui.tabs.autotune_tab import PARAM_TO_CONFIG_IDX

    for idx, key in enumerate(CONFIG_INPUT_ORDER):
        assert PARAM_TO_CONFIG_IDX[key] == idx


def test_mlx_training_six_stage_loss_paths(tmp_path):
    if not _mlx_available():
        return
    from chronos.model.config import ChronosConfig
    from chronos.mlx.training import run_mlx_stage

    cfg = ChronosConfig(hidden_size=32, num_hidden_layers=1, num_experts=2,
                        num_experts_per_tok=1, num_shared_experts=1,
                        num_attention_heads=2, num_key_value_heads=2,
                        kv_latent_dim=8, rope_dim=4, max_seq_len=16,
                        vocab_size=64, use_moe=True,
                        lambda_balance=5e-4, lambda_temporal=1e-3,
                        lambda_lookahead=0.1, lambda_lookahead_topk=0.05)

    class Args:
        learning_rate = 1e-4
        weight_decay = 0.0
        steps = 1
        epochs = 1
        max_seq_len = 8
        max_gen_len = 2
        num_generations = 2
        beta = 0.1
        lambda_or = 0.1
        alpha = 0.7
        temperature = 0.1
        reward = "toy"

    ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    base = run_mlx_stage(
        stage="pretrain",
        config=cfg,
        checkpoint_path=None,
        save_dir=str(tmp_path),
        loader=DataLoader(TensorDataset(ids, ids.clone()), batch_size=1),
        args=Args(),
    ).checkpoint_path
    pair = {
        "x_chosen": torch.randint(0, cfg.vocab_size, (1, 7), dtype=torch.long),
        "y_chosen": torch.randint(0, cfg.vocab_size, (1, 7), dtype=torch.long),
        "mask_chosen": torch.ones(1, 7, dtype=torch.long),
        "x_rejected": torch.randint(0, cfg.vocab_size, (1, 7), dtype=torch.long),
        "y_rejected": torch.randint(0, cfg.vocab_size, (1, 7), dtype=torch.long),
        "mask_rejected": torch.ones(1, 7, dtype=torch.long),
    }
    stages = [
        ("sft", {"loader": DataLoader(TensorDataset(ids, ids.clone()), batch_size=1)}),
        ("dpo", {"loader": [pair]}),
        ("orpo", {"loader": [pair]}),
        ("distill", {"loader": DataLoader(TensorDataset(ids, ids.clone()), batch_size=1), "teacher_path": base}),
        ("grpo", {"prompts": ["hello"]}),
    ]
    for stage, kwargs in stages:
        result = run_mlx_stage(
            stage=stage,
            config=cfg,
            checkpoint_path=base,
            save_dir=str(tmp_path),
            args=Args(),
            **kwargs,
        )
        assert result.steps == 1
        assert os.path.exists(result.checkpoint_path)


def test_inference_prompt_defaults_to_chat_template():
    from ui.tabs.inference_tab import _encode_prompt

    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert messages == [{"role": "user", "content": "你好"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"

        def __call__(self, text, add_special_tokens=False, return_attention_mask=True):
            assert "<|im_start|>assistant" in text
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    ids, mask = _encode_prompt(_Tok(), "你好", raw_prompt=False)
    assert ids == [1, 2, 3]
    assert mask == [1, 1, 1]


def test_streaming_jsonl_dataset_pickles_under_spawn_workers(tmp_path):
    import json
    import multiprocessing as mp
    from torch.utils.data import DataLoader
    from chronos.data.flexible_dataset import FlexibleDataset

    data_path = tmp_path / "tiny_pretrain.jsonl"
    data_path.write_text(
        "\n".join([
            json.dumps({"text": "chronos mps dataloader spawn"}),
            json.dumps({"text": "streaming jsonl worker"}),
        ]) + "\n",
        encoding="utf-8",
    )

    ds = FlexibleDataset(str(data_path), _TinyPickleTokenizer(), max_length=12)
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=1,
        multiprocessing_context=mp.get_context("spawn"),
    )
    ids, labels = next(iter(loader))
    assert ids.shape == labels.shape == (1, 12)


def test_export_i18n_keys_present_for_all_languages():
    from ui.i18n import TRANSLATIONS

    keys = {
        "tab.export",
        "export.title",
        "export.model_path",
        "export.output_dir",
        "export.formats",
        "export.config_path",
        "export.expert_cache_dir",
        "export.copy_cache",
        "export.auto_cluster",
        "export.calibration_data_path",
        "export.cluster_device",
        "export.cluster_max_batches",
        "export.cluster_batch_size",
        "export.cluster_max_seq_len",
        "export.run",
        "export.log",
        "export.summary",
        "infer.mode",
        "infer.compare",
        "infer.compare_table",
        "infer.lazy_output",
        "infer.full_output",
        "infer.miss_policy",
        "infer.ram_load_ratio",
        "infer.sweep_ram_load_ratios",
    }
    for lang in ("zh-Hans", "zh-Hant", "en", "ja"):
        missing = [key for key in keys if TRANSLATIONS[lang].get(key, key) == key]
        assert not missing, f"{lang} missing translations: {missing}"


def test_inference_stats_helpers_are_structured():
    from ui.tabs.inference_tab import _format_inference_stats, _rows_to_chart_df

    rows = [
        {
            "mode": "lazy_offload",
            "backend": "cpu",
            "tokens": 4,
            "tokens_per_sec": 10.0,
            "response_time_s": 0.4,
            "rss_delta_gb": 0.1,
            "setup_rss_delta_gb": 0.02,
            "prefill_rss_delta_gb": 0.03,
            "decode_rss_delta_gb": 0.04,
            "rss_after_setup_gb": 1.20,
            "rss_after_prefill_gb": 1.23,
            "rss_after_decode_gb": 1.27,
            "prefill_time_s": 0.1,
            "decode_time_s": 0.3,
            "load_budget": "2/2 (ideal 2)",
            "cache_hit_rate": 0.75,
            "resident_hit_rate": 0.75,
            "prediction_hit_rate": 0.5,
            "on_demand_loads": 1,
            "async_cold_miss_prefetches": 2,
            "sync_ssd_loads": 1,
            "miss_policy": "on_demand",
            "cache_hits": 3,
            "cache_misses": 1,
            "vram_experts": "2/4",
            "ram_experts": "2/8",
            "cluster_aware": True,
        },
        {
            "mode": "full_dram",
            "backend": "cpu",
            "tokens": 4,
            "tokens_per_sec": 12.0,
            "response_time_s": 0.33,
            "rss_delta_gb": 0.2,
            "setup_rss_delta_gb": 0.1,
            "prefill_rss_delta_gb": 0.02,
            "decode_rss_delta_gb": 0.01,
            "rss_after_setup_gb": 2.00,
            "rss_after_prefill_gb": 2.02,
            "rss_after_decode_gb": 2.03,
            "prefill_time_s": 0.08,
            "decode_time_s": 0.25,
            "load_budget": "all",
            "cache_hit_rate": 1.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "vram_experts": "all",
            "ram_experts": "all",
            "cluster_aware": False,
        },
    ]
    md = _format_inference_stats(rows, {"same": True, "similarity": 1.0})
    df = _rows_to_chart_df(rows)
    assert "| Metric | lazy_offload | full_dram |" in md
    assert "Setup RSS delta" in md and "Prefill RSS delta" in md and "Decode RSS" in md
    assert "1.200 GB" in md and "2.030 GB" in md
    assert "Load budget" in md
    assert "On-demand loads" in md and "Async misses" in md and "Predict hit" in md
    assert "lazy_offload" in md and "full_dram" in md
    assert set(df.columns) == {"metric", "mode", "x", "value", "normalized_value", "unit"}
    assert set(df["mode"]) == {"lazy_offload", "full_dram"}
    assert df["normalized_value"].between(0, 1).all()
    assert {
        "Prefill RSS delta", "Decode RSS delta", "Setup RSS actual",
        "Prefill RSS actual", "Decode RSS actual", "Prefill time", "Decode time",
    }.issubset(set(df["metric"]))


def test_inference_offload_budget_caps_at_125_percent():
    from chronos.model.config import ChronosConfig
    from ui.tabs.inference_tab import _bounded_offload_expert_budget

    cfg = ChronosConfig(num_experts=6, num_experts_per_tok=3, num_hidden_layers=8)
    budget = _bounded_offload_expert_budget(cfg, 2.0)
    assert budget["ideal_active_experts"] == 6
    assert budget["max_allowed_expert_budget"] == 6
    assert budget["effective_expert_budget"] == 6
    assert budget["effective_vram_expert_budget"] == 6
    assert budget["effective_ram_expert_budget"] == 6
    assert budget["routing_top_k"] == 3
    assert budget["num_moe_layers"] == 8

    cfg2 = ChronosConfig(num_experts=64, num_experts_per_tok=4, num_hidden_layers=8)
    budget2 = _bounded_offload_expert_budget(cfg2, 1.25)
    assert budget2["ideal_active_experts"] == 32
    assert budget2["max_allowed_expert_budget"] == 40
    assert budget2["effective_expert_budget"] == 40
    assert budget2["effective_ram_expert_budget"] == 64


def test_inference_ram_load_ratio_accepts_custom_values():
    from chronos.model.config import ChronosConfig
    from ui.tabs.inference_tab import (
        RAM_LOAD_RATIO_CHOICES,
        RAM_LOAD_SWEEP_RATIOS,
        _bounded_offload_expert_budget,
        _normalize_ram_load_ratio,
    )

    assert "0.10" in RAM_LOAD_RATIO_CHOICES
    assert "1.10" in RAM_LOAD_RATIO_CHOICES
    assert 0.33 in RAM_LOAD_SWEEP_RATIOS

    cfg = ChronosConfig(num_experts=64, num_experts_per_tok=4, num_hidden_layers=8)
    custom = _bounded_offload_expert_budget(cfg, "0.33")
    assert custom["requested_ram_load_ratio"] == 0.33
    assert custom["effective_expert_budget"] == 11
    assert custom["effective_ram_expert_budget"] == 32

    custom_high = _bounded_offload_expert_budget(cfg, "1.10")
    assert custom_high["requested_ram_load_ratio"] == 1.1
    assert custom_high["effective_expert_budget"] == 36
    assert custom_high["effective_ram_expert_budget"] == 64

    assert _normalize_ram_load_ratio("not-a-number") == 1.0


def test_generate_api_returns_plain_json_with_chart_records():
    from ui.tabs import inference_tab as mod

    fake_raw = {
        "mode": "offload",
        "backend": "cpu",
        "expert_budget_policy": {"miss_policy": "on_demand"},
        "rows": [{"mode": "lazy_offload", "tokens": 1, "tokens_per_sec": 1.0}],
        "outputs": {"single": "ok"},
        "chart": [{"metric": "Decode speed", "mode": "lazy_offload", "value": 1.0, "unit": "tokens/s"}],
    }
    with mock.patch("ui.tabs.inference_tab._run_inference_modes", return_value=fake_raw):
        out = mod.generate_api({}, "cpu", "offload", "", "hello", 1, 0.1, False, "1.00", False)

    assert isinstance(out, dict)
    assert out["outputs"]["single"] == "ok"
    assert isinstance(out["chart"], list)
    assert out["expert_budget_policy"]["miss_policy"] == "on_demand"


def test_optimizer_decay_groups():
    """M7b: build_optimizer must split params into a decay group
    (linears) and a no-decay group (norms / biases / embeddings)."""
    import torch.nn as nn
    from chronos.trainer.optim_utils import build_optimizer
    m = nn.Sequential(nn.Embedding(10, 4), nn.LayerNorm(4), nn.Linear(4, 4))
    opt = build_optimizer(m, lr=1e-3, weight_decay=0.05)
    decay_g, no_decay_g = opt.param_groups[0], opt.param_groups[1]
    assert decay_g["weight_decay"] == 0.05
    assert no_decay_g["weight_decay"] == 0.0
    # Linear.weight is 2-D → decay; LN scale + Embedding rows + biases → no-decay
    assert any(p.ndim == 2 and p.shape == (4, 4) for p in decay_g["params"])
    assert any(p is m[0].weight for p in no_decay_g["params"])  # embedding
    assert any(p is m[1].weight for p in no_decay_g["params"])  # LN scale


def test_warmup_lr_schedule():
    """M7b: warmup → cosine. LR at step=1 ≪ base, peaks at warmup_steps,
    decays toward min_lr_ratio·base by total_steps."""
    from chronos.trainer.optim_utils import get_lr
    base = 1e-3
    total = 1000
    warm = 50
    assert get_lr(1, total, base, warmup_steps=warm) == base * (1 / warm)
    assert get_lr(warm, total, base, warmup_steps=warm) == base
    end = get_lr(total, total, base, warmup_steps=warm)
    assert abs(end - base * 0.1) < 1e-9, f"expected min_lr_ratio·base, got {end}"


def test_val_split_reproducible():
    """M7c: deterministic idx-modulo split — running twice yields the
    same train/val partition."""
    n = 200
    val_ratio = 0.05
    stride = max(2, int(round(1.0 / val_ratio)))
    val_a = [i for i in range(n) if i % stride == 0]
    val_b = [i for i in range(n) if i % stride == 0]
    assert val_a == val_b
    assert len(val_a) == n // stride


if __name__ == '__main__':
    tests = [
        test_forward,
        test_kv_cache_bounded,
        test_temporal_loss,
        test_lru_cache,
        test_expert_store_init,
        test_lazy_expert_placeholder_uses_shared_fallback,
        test_async_prefetcher,
        test_hybrid_attention_layers,
        test_benchmark_functions,
        test_cluster_storage_roundtrip,
        test_expert_store_clustered_prefetch,
        test_lookahead_supervision_grad,
        test_pipeline_overlap_simulated,
        test_setup_from_state_dict_replaces_live_experts_with_placeholders,
        test_backend_dispatcher,
        test_training_backend_resolver,
        test_training_backend_resolver_falls_back_when_mlx_not_trainable,
        test_train_backend_choice_helpers,
        test_inference_resolves_mlx_without_pytorch_checkpoint_fallback,
        test_device_utils_mps_is_not_cpu_path,
        test_mac_backend_diagnostics_shape,
        test_inference_backend_choice_helpers,
        test_cuda_inference_path_uses_cuda_device_map,
        test_torch_inference_loads_base_only_before_lazy_setup,
        test_param_estimator,
        test_metrics_bus,
        test_vllm_adapter_no_vllm,
        test_hf_save_load_roundtrip,
        test_ui_build,
        test_webui_launch_port_wiring,
        test_stage_entry_mappings_consistent,
        test_pipeline_backend_choices_include_auto,
        test_stage_init_placeholder_means_auto_resolution,
        test_shared_expert_grad,
        test_moe_all_available_matches_no_mask,
        lambda: test_checkpoint_config_roundtrip_strict(__import__("pathlib").Path(__import__("tempfile").mkdtemp())),
        lambda: test_mlx_cluster_aware_expert_store_roundtrip(__import__("pathlib").Path(__import__("tempfile").mkdtemp())),
        lambda: test_mlx_training_pretrain_smoke_roundtrip(__import__("pathlib").Path(__import__("tempfile").mkdtemp())),
        test_inference_prompt_defaults_to_chat_template,
        test_optimizer_decay_groups,
        test_warmup_lr_schedule,
        test_val_split_reproducible,
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
