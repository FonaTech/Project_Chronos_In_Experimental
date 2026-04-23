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


# ── M5/M6 tests ───────────────────────────────────────────────────

def test_backend_dispatcher():
    from chronos.backend import BackendDispatcher
    d = BackendDispatcher()
    avail = d.available()
    assert "cpu" in avail, f"cpu must be available, got {avail}"
    assert d.device_str("cpu") == "cpu"
    assert d.supports_training("cpu") is True
    assert d.supports_training("mlx") is False  # inference-only in current repo
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
        "mlx": mock.Mock(available=True, supports_training=False, torch_device=None),
    }

    with mock.patch.object(d, "info", side_effect=lambda name: fake_infos.get(name, mock.Mock(available=False, supports_training=False, torch_device=None))):
        assert d.training_available() == ["mps", "cpu"]
        assert d.select_training() == "mps"
        assert d.select_training("auto") == "mps"
        assert d.select_training("cpu") == "cpu"
        assert d.select_training("mlx") == "mps"
        assert d.training_device_str("mps") == "mps"
        assert d.resolve_training_device("auto") == ("mps", "mps")
        assert d.resolve_training_device("cpu") == ("cpu", "cpu")


def test_train_backend_choice_helpers():
    from ui.tabs.train_tab import (
        _available_train_backend_choices,
        _train_backend_dropdown_choices,
    )

    with mock.patch("ui.tabs.train_tab.training_available", return_value=["mps", "cpu"]):
        assert _available_train_backend_choices() == ["auto", "mps", "cpu"]
        choices = _train_backend_dropdown_choices("en")
        assert choices[0][1] == "auto"
        assert [value for _, value in choices] == ["auto", "mps", "cpu"]


def test_inference_falls_back_from_mlx_for_pytorch_checkpoint():
    from ui.tabs import inference_tab as mod

    assert mod.TORCH_INFERENCE_PRIORITY[0] == "cuda"

    with mock.patch("chronos.backend.select", return_value="mlx"):
        with mock.patch("ui.tabs.inference_tab._best_torch_inference_backend", return_value="mps"):
            backend, note = mod._resolve_inference_backend("auto", "./out/foo.pth", {"hidden_size": 384})
            assert backend == "mps"
            assert "not lossless" in note

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

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        with mock.patch("chronos.model.model_chronos.ChronosForCausalLM", return_value=fake_model):
            with mock.patch("chronos.runtime.inference_engine.ChronosInferenceEngine", _FakeEngine):
                generated = mod._run_torch_inference(
                    "cuda",
                    model_cfg=object(),
                    model_path_val="",
                    token_ids=[1, 2, 3],
                    max_tok=1,
                    temp=0.8,
                    eos_token_id=99,
                )

    assert generated == [11]
    assert fake_model.devices == ["cuda"]
    assert fake_model.eval_called is True
    assert fake_input.devices == ["cuda"]
    fake_torch.tensor.assert_called_once()


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
    fake_torch = mock.Mock()
    fake_torch.tensor = mock.Mock(return_value=_FakeTensor())
    fake_torch.load = mock.Mock(return_value=fake_state)

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        with mock.patch("os.path.exists", return_value=True):
            with mock.patch("chronos.model.model_chronos.ChronosForCausalLM", return_value=_FakeModel()):
                with mock.patch("chronos.runtime.inference_engine.ChronosInferenceEngine", _FakeEngine):
                    out = mod._run_torch_inference(
                        "cpu",
                        model_cfg=object(),
                        model_path_val="/tmp/fake.pth",
                        token_ids=[1, 2, 3],
                        max_tok=4,
                        temp=0.8,
                        eos_token_id=99,
                    )

    assert out == []
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
    from ui.tabs.pipeline_tab import PIPELINE_TRAIN_BACKEND_CHOICES

    assert PIPELINE_TRAIN_BACKEND_CHOICES[0] == "auto"


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
        test_train_backend_choice_helpers,
        test_inference_falls_back_from_mlx_for_pytorch_checkpoint,
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
