# Project Chronos (In Experimental)

**一套从架构层原生支持 SSD+DRAM 混合加载推理的 MoE 框架**

[![PyPI](https://img.shields.io/pypi/v/project-chronos)](https://pypi.org/project/project-chronos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

[English](README.md)

---

## 现有方案的根本性缺陷

主流 MoE 模型（Mixtral、DeepSeek-MoE、Qwen-MoE 等）在消费级硬件上运行时，路由决策是**即时、被动、逐 token** 的——每一步 decode 才去检查所需专家是否在 VRAM 中。当 VRAM 不足时，系统在 decode 流中同步等待 SSD→RAM→VRAM 的搬运，生成速度跌至 **< 5 tokens/s**。

这不是参数调优问题，而是**架构层的根本缺陷**：这些模型设计时假设全量 VRAM 驻留，offload 是事后打的补丁。

---

## Chronos 的核心思路：把 IO 挪到 Prefill

```
传统 MoE 解码循环：                    Chronos 解码循环：

  Token t                              Prefill（一次性）：
    → 路由 → 需要哪个专家？                读取完整 Prompt
    → 专家在 VRAM？不在                    IntentClassifier → 预测专家集合
    → [阻塞：SSD→RAM→VRAM，~40ms]          AsyncLoad: SSD→RAM→VRAM（后台）
    → 计算                                 等待加载完毕
    → Token t+1（重复）
                                       Token t（以及之后每个 token）：
  每个 token 都在等 IO。                   路由 → 需要哪个专家？
                                           专家已在 VRAM（已预加载）
                                           直接计算
                                           [无 IO，无阻塞]
```

核心洞察：**给定 Prompt，MoE 模型在整次生成中会用到哪些专家，是可以从 Prompt 本身预测出来的。** 编程类问题会激活代码专家，推理类问题激活逻辑专家。`IntentClassifier`——一个约 10–15M 参数的小型 Dense 编码器——在 prefill 阶段一次性读取完整 Prompt，预测整次生成所需的专家集合，在第一个 decode token 开始之前全部加载完毕。

---

## 三级存储架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VRAM / Metal 统一内存                            │
│  ┌─────────────────┐  ┌──────────────────────────────────────────────┐  │
│  │  Dense 第一层   │  │  共享专家（常驻）                            │  │
│  │  (IntentCLF)    │  │  + 预测热门专家（预加载完毕）                │  │
│  └─────────────────┘  └──────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  H2D：独立 CUDA Stream（非阻塞）
┌──────────────────────────────────▼──────────────────────────────────────┐
│                         Pinned RAM 缓冲区                               │
│         已预取的专家权重（mmap，页锁定内存）                             │
│         AsyncPrefetcher 后台线程从 SSD 读入此处                         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  顺序读取（4KB 对齐，NVMe 优化）
┌──────────────────────────────────▼──────────────────────────────────────┐
│                              NVMe SSD                                   │
│         全量专家权重（量化存储，共现聚簇布局）                           │
│         共激活的专家在磁盘上相邻存放，最大化顺序读带宽                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**这条链路上没有任何环节阻塞 decode 循环。** 专家权重在后台异步向上搬运，计算流不等待 IO。

### 缓存未命中时的处理

即使最坏情况也不会卡顿——柔性降级（Soft Gating）：

```python
# 纯张量乘法，无 Python 分支，torch.compile 不会图断裂
output = avail[i] * expert_output + (1.0 - avail[i]) * shared_expert_output
```

共享专家（常驻 VRAM）按比例混入，生成流**从不中断**，精度平滑降级，专家后台加载完毕后自动恢复。

---

## 双层路由：为什么是两个预测器？

| | IntentClassifier（第一层） | LookaheadRouter（第二层） |
|---|---|---|
| **触发时机** | Prefill 阶段一次 | Decode 每个 token |
| **输入** | 完整 Prompt（最多 512 token） | Block 0 的 hidden state |
| **输出** | 整次生成的专家集合 | t+1、t+2 步的专家 ID |
| **职责** | 宏观意图识别，批量预加载 | 捕获晚期预测偏差，更新预取队列 |
| **参数量** | ~10–15M（单独训练） | ~2M（随主模型训练） |

两者互补：第一层处理**宏观任务类型**，第二层处理**微观上下文转折**。

---

## 这套架构真的能实现吗？诚实评估

### 确定可以实现的

- **长生成（100+ tokens）的解码阶段零 IO**：PrefillScheduler 将所有专家加载前置，decode 流中 VRAM 已包含所需专家。
- **永不硬阻塞**：Soft Gating 确保 cache miss 只带来轻微精度混合，不冻结解码器。
- **VRAM 占用大幅降低**：只需常驻共享专家 + 预测热门专家，而非全量专家。
- **NVMe 带宽高效利用**：共现聚簇存储确保高频共激活专家相邻，最大化顺序读性能。

### 有条件成立的

| 主张 | 实际情况 |
|---|---|
| "不影响效率" | 不完全准确：prefill 阶段增加 SSD 加载耗时（具体时长取决于 NVMe 带宽和专家数量）。**对短回答（< 20 tokens）首 token 延迟会有感知上升。** 对长回答，这笔开销在第一个 token 后就被均摊掉了。 |
| IntentClassifier 预测准确率 ≥ 85% | **需要真实激活数据训练后才能达到。** 冷启动阶段使用频率启发式，安全但精度较低。运行几百轮采集真实 prompt→激活日志后准确率持续提升。 |
| 时间局部性损失提升命中率 | 理论上成立。λ₂ 过大会轻微损害路由质量，需通过 Optuna 自动搜索找最优平衡点。 |

> **注意**：目前尚未完成系统性定量对比基准测试。上述性质基于架构分析和小规模实验，大规模实测数据将在 Phase 5 补充。

### 与现有框架的本质差异

主流推理框架（vLLM、llama.cpp、ollama）全部使用**被动 offload**：decode 循环发现 cache miss 才同步加载。Chronos 将被动模式转化为**主动预测性**模式——不是运行时补丁，而是从模型设计层就融入：

1. **PrefillScheduler** 读取整个 Prompt，批量预加载专家，再开始 decode——工业框架中无现成实现
2. **LookaheadRouter** 在 decode 内部提前 1–2 步预测，与 PrefillScheduler 形成双保险
3. **时间局部性损失** 从训练阶段主动鼓励相邻 token 复用同一批专家，让预测准确率的上限更高
4. **共现聚簇存储** 在磁盘布局层就优化 IO 模式，而非依赖 OS 页缓存碰运气

---

## 与现有方案对比

| 特性 | llama.cpp offload | vLLM offload | **Project Chronos** |
|---|---|---|---|
| 专家预测 | 无（被动） | 无（被动） | **主动预测（IntentCLF + LookaheadRouter）** |
| IO 时机 | Decode 期间（阻塞） | Decode 期间（阻塞） | **Prefill 期间（异步，前置）** |
| Cache miss 行为 | 硬阻塞 | 硬阻塞 | **Soft Gating（零阻塞）** |
| 训练集成 | 事后补丁 | 事后补丁 | **原生损失项（时间局部性）** |
| 磁盘布局优化 | 无 | 无 | **共现聚簇，NVMe 顺序读优化** |
| CUDA Stream 隔离 | 无 | 部分 | **独立 H2D Stream** |
| Apple Silicon 原生 | 部分 | 无 | **完整 MLX 后端** |

---

## 架构概览

```
输入 Token（完整 Prompt）
        │
        ▼
┌───────────────────────────────────────┐
│  IntentClassifier（10–15M Dense）     │  ← Prefill 阶段执行一次
│  读取完整 Prompt → IntentVector       │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  ExpertPredictor                      │
│  IntentVector → ExpertSet             │
│  （阈值筛选 + VRAM 预算上限）          │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐   ┌─────────────────────────┐
│  PrefillScheduler                     │──▶│  AsyncLoader 后台线程   │
│  批量加载预测专家                     │   │  SSD → Pinned RAM       │
│  等待加载完毕                         │   │  → VRAM（独立 CUDA 流） │
└───────────────────┬───────────────────┘   └─────────────────────────┘
                    │
                    ▼
    Decode 循环（零 IO，全速运行）
        │
        ├─ Block 0: MLAAttention
        │     └─ LookaheadRouter → 预测 t+1、t+2 专家
        │           └─ AsyncPrefetcher: 调度后台加载
        │
        ├─ Block 1–N: SlidingWindowAttention / MLAAttention（交替）
        │
        └─ ChronosMOEFeedForward
              ├─ 专家在 VRAM？→ 直接计算
              └─ Cache miss？ → Soft Gating 混合共享专家
```

### 损失函数

$$L_{\text{total}} = L_{\text{CE}} + \lambda_1 L_{\text{balance}} + \lambda_2 \sum_{t=2}^{T} \| E_t - E_{t-1} \|_2^2$$

时间局部性项 $\lambda_2$ **在训练阶段主动鼓励相邻 token 路由到同一批专家**——不只是为了降低 loss，而是因为稳定的路由让相同专家持续驻留 VRAM，缓存命中率天然提高。λ₁ 和 λ₂ 均通过 **Optuna TPE 贝叶斯搜索**自动调优。

---

## 安装  (Not Ready in PyPI Yet)

```bash
pip install project-chronos
```

或从源码安装：

```bash
git clone https://github.com/your-org/project-chronos
cd project-chronos
pip install -e ".[dev]"
```

**MLX（Apple Silicon）：**
```bash
pip install "project-chronos[mlx]"
```

> **minimind 依赖**：Project Chronos 使用 [minimind](https://github.com/jingyaogong/minimind) 作为 MoE 内核。
> 若本地未找到，首次 import 时自动克隆至 `~/.cache/chronos/minimind-master/`。
> minimind 采用 **Apache-2.0** 授权，完整归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

**环境要求**：Python 3.10+，PyTorch 2.1+

---

## 快速开始

### Web UI

```bash
chronos-ui
# 或
python chronos_app.py
```

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

### Phase 1 验证（前瞻准确率）

```bash
chronos eval \
    --data_path ./dataset/eval.jsonl \
    --model_path ./out/chronos_512_moe.pth \
    --device cuda
```

### 专家聚簇存储（最大化 SSD 顺序读）

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

## 项目结构

```
Project_Chronos/
├── chronos/
│   ├── deps.py                    # 自动下载 minimind（若本地未找到）
│   ├── model/
│   │   ├── config.py              # ChronosConfig
│   │   ├── hybrid_attention.py    # MLAAttention + SlidingWindowAttention
│   │   ├── lookahead_router.py    # 逐 token 前瞻预测器（第二层）
│   │   ├── moe_chronos.py         # ChronosMOEFeedForward + 共享专家 + Soft Gating
│   │   ├── model_chronos.py       # ChronosForCausalLM
│   │   └── temporal_loss.py       # 时间局部性损失
│   ├── io/
│   │   ├── expert_store.py        # 三级存储，独立 CUDA H2D Stream
│   │   ├── async_prefetcher.py    # 异步预取引擎
│   │   └── cluster_layout.py      # 共现聚簇存储布局生成
│   ├── router/
│   │   ├── intent_classifier.py   # Prompt 级专家预测器（第一层，~10M 参数）
│   │   ├── expert_predictor.py    # IntentVector → ExpertSet（含预算上限）
│   │   └── prefill_scheduler.py   # 编排 prefill 阶段批量预加载
│   ├── mlx/                       # Apple Silicon 原生后端
│   │   ├── attention.py
│   │   ├── moe.py
│   │   ├── model.py
│   │   ├── expert_store.py
│   │   └── inference.py
│   ├── runtime/
│   │   ├── cache_manager.py       # 统一缓存接口
│   │   └── inference_engine.py    # 端到端推理引擎（PyTorch）
│   ├── tuning/
│   │   └── chronos_auto_tuner.py  # Optuna λ 搜索
│   ├── eval/
│   │   ├── io_profiler.py         # Phase 1 验证
│   │   └── benchmark.py           # 端到端基准测试
│   ├── data/
│   │   └── flexible_dataset.py    # 自动识别任意 JSONL 字段格式
│   ├── backend.py                 # 自动检测 MLX / CUDA / MPS / CPU
│   └── cli.py                     # 统一 CLI
├── ui/                            # Gradio Web UI（4 语言 i18n）
├── chronos_app.py                 # Web UI 入口
├── train_chronos.py               # 训练入口
├── tests/test_smoke.py
├── pyproject.toml
└── README.md
```

---

## 开发路线图

- [x] Phase 1: LookaheadRouter + 时间局部性损失
- [x] Phase 2: 异步 IO 引擎 + 三级存储 + 共现聚簇布局
- [x] Phase 3: 混合注意力（MLA + SlidingWindow）+ PrefillScheduler + 双层路由
- [x] Phase 4: MLX 原生后端、Web UI、CLI、Optuna λ 搜索、开源发布
- [ ] Phase 5: 在大规模激活语料上训练 IntentClassifier；7B+ 规模模型基准测试

---

## 引用

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

## 第三方归属

Project Chronos 基于 **jingyaogong** 的 [minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）构建。完整归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
