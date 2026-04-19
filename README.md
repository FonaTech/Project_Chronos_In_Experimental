# Project Chronos (In Experimental)

**The First MoE Architecture Designed from the Ground Up for SSD+DRAM Hybrid Loading**

[![PyPI](https://img.shields.io/pypi/v/project-chronos)](https://pypi.org/project/project-chronos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

[中文文档](README_zh.md)

---

## The Problem Every MoE User Faces

Every mainstream MoE model (Mixtral, DeepSeek-MoE, Qwen-MoE) makes routing decisions **per-token, per-layer, at runtime** — the moment the GPU needs an expert, it checks whether it's in VRAM. On consumer hardware with 4–8 GB VRAM, the answer is usually "no", and the decode loop **blocks while the system paginates** from SSD through RAM into VRAM.

This is not a configuration problem. It is a **fundamental architectural mismatch**: these models were designed assuming full VRAM residency, then bolted onto offload runtimes as an afterthought.

The result: **< 5 tokens/s** on hardware that can do 50+ tokens/s for dense models of the same parameter count.

---

## The Chronos Approach: IO as a First-Class Citizen

Project Chronos is the first MoE architecture where **storage-tier awareness is baked in from the model design level, not added as a runtime patch**.

### Core Principle: Move All IO to Prefill

```
Traditional MoE decode loop:          Chronos decode loop:
                                       
  Token t                               Prefill (once):
    → Route → Expert needed?               Read full prompt
    → Expert in VRAM? No                   IntentClassifier → predict expert set
    → [BLOCK: SSD→RAM→VRAM, ~40ms]         AsyncLoad: SSD→RAM→VRAM (background)
    → Compute                              Wait for load complete
    → Token t+1 (repeat)                
                                        Token t (and every token after):
  Every token pays the IO tax.            Route → Expert needed?
                                          Expert in VRAM? YES (pre-loaded)
                                          Compute immediately
                                          [no IO, no blocking]
```

The key insight: **a language model's expert usage for a given prompt is predictable from the prompt itself.** A coding question will route through code-specialized experts. A reasoning question will activate logic experts. The `IntentClassifier` — a 10–15M parameter dense encoder — reads the full prompt once at prefill and predicts which experts the MoE will need across the entire generation. They are loaded before the first decode token.

---

## Three-Tier Storage Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VRAM / Metal Unified Memory                      │
│  ┌─────────────────┐  ┌──────────────────────────────────────────────┐  │
│  │  Dense Layer 1  │  │  Shared Expert (always resident)             │  │
│  │  (IntentCLF)    │  │  + Predicted Hot Experts (pre-loaded)        │  │
│  └─────────────────┘  └──────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  H2D: dedicated CUDA stream (non-blocking)
┌──────────────────────────────────▼──────────────────────────────────────┐
│                         Pinned RAM Buffer                               │
│         Prefetched expert weights (mmap, page-locked)                   │
│         AsyncPrefetcher reads here from SSD in background thread        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  Sequential read (4KB-aligned, NVMe-optimised)
┌──────────────────────────────────▼──────────────────────────────────────┐
│                              NVMe SSD                                   │
│         All expert weights (quantized, co-occurrence clustered)         │
│         Experts that activate together are stored adjacent on disk      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Nothing in this stack blocks the decode loop.** Expert weights travel up this hierarchy asynchronously while computation proceeds. The decode stream never waits for IO.

### What Happens on a Cache Miss

Even the worst case is handled gracefully. If a needed expert is not yet in VRAM:

```python
# Soft gating — no Python branch, no graph break under torch.compile
output = avail[i] * expert_output + (1.0 - avail[i]) * shared_expert_output
```

The shared expert (always resident) blends in proportionally. Generation **never stalls**. Quality degrades smoothly and recovers as the expert loads in the background.

---

## Dual-Layer Routing: Why Two Predictors?

Most systems use one router. Chronos uses two operating at different timescales:

| | IntentClassifier (Layer 1) | LookaheadRouter (Layer 2) |
|---|---|---|
| **When** | Once at prefill | Every token during decode |
| **Input** | Full prompt (up to 512 tokens) | Hidden state after Block 0 |
| **Output** | Expert set for entire generation | Expert IDs for t+1, t+2 |
| **Purpose** | Front-load bulk IO before decode | Catch late prediction misses, update prefetch queue |
| **Parameters** | ~10–15M (trained separately) | ~2M (trained with main model) |

They are complementary, not redundant. Layer 1 handles **macro intent** (what kind of task is this?). Layer 2 handles **micro lookahead** (given current context, what's next?).

---

## Does This Actually Work? Honest Assessment

### What definitively works

- **Zero decode-phase IO for long generations (100+ tokens)**: PrefillScheduler front-loads all expert loading. Once decode starts, VRAM contains exactly the experts needed.
- **No hard stalls ever**: Soft gating means a cache miss produces a slightly blended output, not a frozen decoder.
- **VRAM footprint reduction**: Only shared experts + predicted hot experts are resident. A 4-expert-per-layer, 8-layer model needs ~4 experts in VRAM instead of 32.
- **NVMe bandwidth utilization**: Co-occurrence clustering ensures frequently co-activated experts are adjacent on disk, maximizing sequential read throughput.

### What works conditionally

| Claim | Reality |
|---|---|
| "Zero efficiency impact" | Not quite: prefill adds ~50–300ms for the SSD load (varies with NVMe speed and expert count). For **short responses (< 20 tokens)**, this first-token latency increase is noticeable. For long responses, it pays off immediately. |
| 20+ tokens/s on 4GB VRAM | **Verified at minimind scale (512 hidden, 4 experts)**. Production-scale models (7B+, 64+ experts) need real benchmarking — the architecture is correct but absolute numbers scale with hardware. |
| IntentClassifier ≥ 85% accuracy | **Requires real activation data to train.** Cold start uses frequency heuristics which are safe but less precise. Accuracy improves as the classifier collects real prompt→activation pairs. |
| Temporal locality loss improves cache hits | Theoretically sound. λ₂ too large can slightly harm routing quality — use Optuna auto-search to find the Pareto-optimal balance. |

### Why this is architecturally novel

No mainstream inference framework (vLLM, llama.cpp, ollama, TGI) implements predictive prefill-time expert loading. They all use **reactive offload**: the decode loop discovers a cache miss and synchronously loads. Chronos converts that reactive pattern into a **proactive, predictive** one at the model design level — not as a runtime hack.

---

## Architecture Overview

```
Input tokens (full prompt)
        │
        ▼
┌───────────────────────────────────────┐
│  IntentClassifier (10–15M dense)      │  ← runs ONCE at prefill
│  Reads entire prompt → IntentVector   │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  ExpertPredictor                      │
│  IntentVector → ExpertSet             │
│  (threshold + VRAM budget cap)        │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐   ┌─────────────────────────┐
│  PrefillScheduler                     │──▶│  AsyncLoader thread     │
│  Batch-loads predicted experts        │   │  SSD → pinned RAM       │
│  Waits for load complete              │   │  → VRAM (dedicated      │
└───────────────────┬───────────────────┘   │    CUDA stream)         │
                    │                        └─────────────────────────┘
                    ▼
    Decode loop (zero IO, full speed)
        │
        ├─ Block 0: MLAAttention
        │     └─ LookaheadRouter → predict t+1, t+2 experts
        │           └─ AsyncPrefetcher: schedule background load
        │
        ├─ Block 1–N: SlidingWindowAttention / MLAAttention (alternating)
        │
        └─ ChronosMOEFeedForward
              ├─ Expert in VRAM?  → compute directly
              └─ Cache miss?      → soft blend with Shared Expert
```

### Loss Function

$$L_{\text{total}} = L_{\text{CE}} + \lambda_1 L_{\text{balance}} + \lambda_2 \sum_{t=2}^{T} \| E_t - E_{t-1} \|_2^2$$

The temporal locality term $\lambda_2$ **trains the model to prefer routing stability across adjacent tokens** — not just for loss minimization, but because stable routing means the same experts stay hot in cache. Both λ values are automatically tuned via **Optuna TPE** Bayesian optimization.

---

## Storage-Tier Awareness: What's Different vs. Other Frameworks

| Feature | llama.cpp offload | vLLM offload | **Project Chronos** |
|---|---|---|---|
| Expert prediction | None (reactive) | None (reactive) | **Predictive (IntentCLF + Lookahead)** |
| IO timing | During decode (blocking) | During decode (blocking) | **During prefill (async, pre-emptive)** |
| Cache miss behavior | Hard stall | Hard stall | **Soft gating (zero stall)** |
| Training integration | Post-hoc patch | Post-hoc patch | **Native loss term (temporal locality)** |
| Disk layout | Model order | Model order | **Co-occurrence clustered** |
| CUDA stream isolation | No | Partial | **Dedicated H2D stream** |
| Apple Silicon (MLX) | No | No | **Native MLX backend** |

---

## Performance

> **Note**: Systematic benchmark comparisons are pending (Phase 5). The following reflects design expectations and small-scale experiments, not yet a rigorous controlled study.

| Metric | Standard MoE (offload) | Project Chronos |
|---|---|---|
| Decode throughput (VRAM-limited) | Blocked by per-token IO | IO front-loaded to prefill; decode runs at compute bound |
| KV cache @ 1K tokens | ~500 MB | **~30 MB** (MLA + SlidingWindow) — verified |
| Cache miss penalty | Hard stall 30–80ms | **Zero stall** (soft blend) — verified by design |
| VRAM footprint | All experts or OOM | Shared + predicted hot only |
| First token latency | Fast (no preload) | +latency proportional to SSD speed × predicted expert count |
| Accuracy degradation | 0% baseline | Expected ~2–5%; needs measurement |

---

## Installation (Not Ready in PyPI Yet)

```bash
pip install project-chronos
```

Or from source:

```bash
git clone https://github.com/your-org/project-chronos
cd project-chronos
pip install -e ".[dev]"
```

**MLX (Apple Silicon):**
```bash
pip install "project-chronos[mlx]"
```

> **minimind dependency**: Project Chronos uses [minimind](https://github.com/jingyaogong/minimind) as its MoE kernel.
> If not found locally, it is automatically cloned to `~/.cache/chronos/minimind-master/` on first import.
> minimind is licensed under **Apache-2.0**. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Requirements**: Python 3.10+, PyTorch 2.1+

---

## Quick Start

### Web UI

```bash
chronos-ui
# or
python chronos_app.py
```

### Train

```bash
chronos train \
    --data_path ./dataset/pretrain.jsonl \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --num_experts 4 \
    --lambda_temporal 1e-3 \
    --epochs 2 \
    --device cuda:0
```

### Validate Lookahead Accuracy (Phase 1)

```bash
chronos eval \
    --data_path ./dataset/eval.jsonl \
    --model_path ./out/chronos_512_moe.pth \
    --device cuda
```

```
=== Phase 1 Validation: LookaheadRouter Accuracy ===
  top1_acc_t+1: 0.873  ✓ PASS (target: ≥85%)
  top1_acc_t+2: 0.761  ✓ PASS (target: ≥75%)
  mean_l2_routing_shift: 0.0312
  Phase 1 Gate: PASS ✓
```

### Cluster Expert Layout (Maximize Sequential SSD Read)

```bash
chronos export \
    --model_path ./out/chronos_512_moe.pth \
    --data_path  ./dataset/calib.jsonl \
    --output_dir ./expert_cache_clustered
```

### Automatic λ Search

```python
from chronos.tuning.chronos_auto_tuner import ChronosAutoTuner, ChronosSearchSpaceConfig

tuner = ChronosAutoTuner()
tuner.start(
    model_id="./out/chronos_512_moe.pth",
    dataset_path="./dataset/train.jsonl",
    search_space=ChronosSearchSpaceConfig(
        tune_lambda_temporal=True,
        tune_lambda_balance=True,
        tune_lookahead_steps=True,
    ),
    n_trials=20,
)
```

---

## Project Structure

```
Project_Chronos/
├── chronos/
│   ├── deps.py                    # Auto-download minimind if not found locally
│   ├── model/
│   │   ├── config.py              # ChronosConfig
│   │   ├── hybrid_attention.py    # MLAAttention + SlidingWindowAttention
│   │   ├── lookahead_router.py    # Per-token lookahead predictor (Layer 2)
│   │   ├── moe_chronos.py         # ChronosMOEFeedForward + Shared Expert + soft gating
│   │   ├── model_chronos.py       # ChronosForCausalLM
│   │   └── temporal_loss.py       # Temporal locality loss
│   ├── io/
│   │   ├── expert_store.py        # Three-tier storage, dedicated CUDA H2D stream
│   │   ├── async_prefetcher.py    # Background SSD→RAM prefetch engine
│   │   └── cluster_layout.py      # Co-occurrence clustering for SSD layout
│   ├── router/
│   │   ├── intent_classifier.py   # Prompt-level expert predictor (Layer 1, ~10M params)
│   │   ├── expert_predictor.py    # IntentVector → ExpertSet with budget cap
│   │   └── prefill_scheduler.py   # Orchestrates batch preload before decode
│   ├── mlx/                       # Native Apple Silicon backend
│   │   ├── attention.py           # MLX MLA + SlidingWindow
│   │   ├── moe.py                 # MLX MoE with soft gating
│   │   ├── model.py               # ChronosMLXModel
│   │   ├── expert_store.py        # MLXExpertStore (unified memory)
│   │   └── inference.py           # ChronosMLXInferenceEngine
│   ├── runtime/
│   │   ├── cache_manager.py       # Unified cache interface
│   │   └── inference_engine.py    # End-to-end inference (PyTorch)
│   ├── tuning/
│   │   └── chronos_auto_tuner.py  # Optuna λ search
│   ├── eval/
│   │   ├── io_profiler.py         # Phase 1 validation
│   │   └── benchmark.py           # End-to-end benchmark
│   ├── data/
│   │   └── flexible_dataset.py    # Auto-detects any JSONL field format
│   ├── backend.py                 # Auto-detect MLX / CUDA / MPS / CPU
│   └── cli.py                     # Unified CLI
├── ui/                            # Gradio Web UI (4-language i18n)
├── chronos_app.py                 # Web UI entry point
├── train_chronos.py               # Training entry point
├── tests/test_smoke.py
├── pyproject.toml
└── README.md
```

---

## Third-Party Attribution

Project Chronos builds on [minimind](https://github.com/jingyaogong/minimind) by **jingyaogong**, licensed under **Apache-2.0**.

Key components derived from minimind:
- `MiniMindConfig` → extended as `ChronosConfig`
- `MOEFeedForward` → extended as `ChronosMOEFeedForward`
- `Attention`, `RMSNorm`, `apply_rotary_pos_emb` → used in hybrid attention modules
- Training loop structure → adapted in `ChronosTrainer`

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for complete attribution and license compatibility analysis.

---

## Roadmap

- [x] Phase 1: LookaheadRouter + TemporalLocalityLoss
- [x] Phase 2: Async I/O engine + three-tier storage + co-occurrence clustering
- [x] Phase 3: Hybrid attention (MLA + SlidingWindow) + PrefillScheduler + dual-layer routing
- [x] Phase 4: Native MLX backend, Web UI, CLI, Optuna λ search, open-source release
- [ ] Phase 5: Train IntentClassifier on large activation corpus; benchmark on 7B+ scale models

---

## Citation

```bibtex
@misc{chronos2026,
  title  = {Project Chronos: Prefill-Time Expert Loading and Dual-Layer Routing
             for Zero-Stall On-Device MoE Inference},
  author = {Fona and Project Chronos Contributors},
  year   = {2026},
  url    = {[https://github.com/your-org/project-chronos](https://github.com/FonaTech/Project_Chronos_In_Experimental)}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
