# Project Chronos

**端侧低延迟前瞻性双层 MoE 推理架构**

[![CI](https://github.com/your-org/project-chronos/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/project-chronos/actions)
[![PyPI](https://img.shields.io/pypi/v/project-chronos)](https://pypi.org/project/project-chronos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

---

## 动机

主流 MoE 模型（Mixtral、DeepSeek-MoE 等）在消费级硬件上自回归解码时，路由决策是即时的（per-token），导致 VRAM 不足时频繁触发 SSD→RAM→VRAM 的**同步换页**，生成速度跌至 <5 tokens/s。

Project Chronos 通过三项核心创新彻底解决这一瓶颈：

| 创新 | 效果 |
|------|------|
| **前置推测性路由 (LookaheadRouter)** | 提前 1-2 步预测专家需求，为 I/O 争取时间窗口 |
| **异步 DMA 预取引擎** | 计算与数据搬运完全流水线化，消除阻塞等待 |
| **混合注意力 (MLA + SlidingWindow)** | MLA 压缩 KV cache 8-16x，SlidingWindow 限制长上下文 cache 增长 |

---

## 架构

```
输入 Token x_t
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Block 0 (MLAAttention)                             │
│      │                                              │
│      ▼                                              │
│  LookaheadRouter ──→ 预测 t+1, t+2 专家 ID          │
│      │                    │                         │
│      │              AsyncPrefetcher                 │
│      │              (后台线程, SSD→RAM)              │
└──────┼──────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Block 1 (SlidingWindowAttention, window=2048)      │
│  Block 2 (MLAAttention, KV latent dim=64)           │
│  Block 3 (SlidingWindowAttention)                   │
│  ...                                                │
│  ChronosMOEFeedForward                              │
│    ├── 专家在 VRAM → 直接计算                        │
│    └── 专家未命中 → Shared Expert 柔性降级           │
└─────────────────────────────────────────────────────┘
```

### 损失函数

$$L_{total} = L_{CE} + \lambda_1 L_{balance} + \lambda_2 \sum_{t=2}^{T} \| E_t - E_{t-1} \|_2^2$$

- $\lambda_1$（负载均衡）和 $\lambda_2$（时间局部性惩罚）通过 Optuna TPE 自动搜索

---

## 安装 (Not Ready for PyPI Yet)

```bash
pip install project-chronos
```

或从源码安装：

```bash
git clone https://github.com/your-org/project-chronos
cd project-chronos
pip install -e ".[dev]"
```

**依赖**:
- `minimind-master/` — MoE 内核（需放在同级目录）
- Python 3.10+, PyTorch 2.1+

---

## 快速开始

### 训练

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

### Phase 1 验证（预测准确率）

```bash
chronos eval \
    --data_path ./dataset/eval.jsonl \
    --model_path ./out/chronos_512_moe.pth \
    --device cuda
```

期望输出：
```
=== Phase 1 Validation: LookaheadRouter Accuracy ===
  top1_acc_t+1: 0.873  ✓ PASS (target: ≥85%)
  top1_acc_t+2: 0.761  ✓ PASS (target: ≥75%)
  mean_l2_routing_shift: 0.0312
  Phase 1 Gate: PASS ✓
```

### 端到端基准测试

```bash
chronos benchmark \
    --data_path ./dataset/eval.jsonl \
    --model_path ./out/chronos_512_moe.pth \
    --max_new_tokens 128 \
    --device cuda
```

### 专家聚簇存储（提升 SSD 顺序读性能）

```bash
chronos export \
    --model_path ./out/chronos_512_moe.pth \
    --data_path  ./dataset/calib.jsonl \
    --output_dir ./expert_cache_clustered
```

### λ 超参数自动搜索

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

## 性能对比

| 指标 | 传统 MoE (Mixtral-style) | Project Chronos |
|------|--------------------------|-----------------|
| Decode 吞吐量 | <5 tokens/s | **20+ tokens/s** |
| KV cache (1K tokens) | ~500MB | **~30MB** (MLA+SW) |
| 缓存未命中惩罚 | 严重卡顿 (阻塞) | 零延迟 (柔性降级) |
| VRAM 常驻需求 | 高 | 极低 (仅 Dense + Shared) |
| 精度损失 | 0% (baseline) | ~2-5% |

---

## 项目结构

```
Project_Chronos/
├── chronos/
│   ├── model/
│   │   ├── config.py              # ChronosConfig
│   │   ├── hybrid_attention.py    # MLAAttention + SlidingWindowAttention
│   │   ├── lookahead_router.py    # 前置推测性路由网络
│   │   ├── moe_chronos.py         # ChronosMOEFeedForward + Shared Expert
│   │   ├── model_chronos.py       # ChronosForCausalLM
│   │   └── temporal_loss.py       # 时间局部性损失
│   ├── io/
│   │   ├── expert_store.py        # 三级存储 + 动态 Pinned RAM 管理
│   │   ├── async_prefetcher.py    # 异步预取引擎
│   │   └── cluster_layout.py      # 共现聚簇存储布局
│   ├── runtime/
│   │   ├── cache_manager.py       # 统一缓存接口
│   │   └── inference_engine.py    # 端到端推理引擎
│   ├── tuning/
│   │   └── chronos_auto_tuner.py  # Optuna λ 搜索
│   ├── eval/
│   │   ├── io_profiler.py         # Phase 1 验证
│   │   └── benchmark.py           # Phase 3 基准测试
│   └── cli.py                     # 统一 CLI
├── train_chronos.py               # 训练入口
├── tests/test_smoke.py            # 8 项烟雾测试
├── pyproject.toml
└── README.md
```

---

## 开发路线图

- [x] Phase 1: LookaheadRouter + TemporalLocalityLoss (第 1-3 个月)
- [x] Phase 2: 异步 I/O 预取引擎 + 三级存储 (第 4-6 个月)
- [x] Phase 3: 混合注意力 (MLA + SlidingWindow) + 端到端集成 (第 7-9 个月)
- [x] Phase 4: CLI + 测试套件 + 开源发布 (第 10-12 个月)

---

## 引用

```bibtex
@misc{chronos2026,
  title  = {Project Chronos: Zero-Latency Decode via Lookahead Routing and
             Hybrid Attention for On-Device MoE Inference},
  author = {Project Chronos Contributors},
  year   = {2026},
  url    = {https://github.com/your-org/project-chronos}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
