# Project Chronos (In Experimental)

**一套从架构层原生支持 SSD+DRAM 混合加载推理的 MoE 框架，配套完整的 6 阶段训练链路。**

[![PyPI](https://img.shields.io/pypi/v/project-chronos)](https://pypi.org/project/project-chronos/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

[English](README.md)

---

## 现有方案的根本性缺陷

主流 MoE 模型（Mixtral、DeepSeek-MoE、Qwen-MoE 等）在消费级硬件上运行时，路由决策是**即时、被动、逐 token** 的——每一步 decode 才去检查所需专家是否在 VRAM 中。当 VRAM 不足时，系统在 decode 流中同步等待 SSD→RAM→VRAM 的搬运，生成速度跌至 **< 5 tokens/s**。

这不是参数调优问题，而是**架构层的根本缺陷**：这些模型设计时假设全量 VRAM 驻留，offload 是事后打的补丁。

---

## Chronos 的核心思路：把 IO 挪到 Prefill，把同步挪到事件级

```
传统 MoE 解码循环：                    Chronos 解码循环：

  Token t                              Prefill（一次性）：
    → 路由 → 需要哪个专家？                读取完整 Prompt
    → 专家在 VRAM？不在                    IntentClassifier → 预测专家集合
    → [阻塞：SSD→RAM→VRAM，~40ms]          AsyncLoad: SSD→RAM→VRAM（后台）
    → 计算                                 等待加载完毕
    → Token t+1（重复）
                                       Token t（之后每个 token）：
                                         Prefetch(t+1 expert) ──► (H2D 流)
  每个 token 都在等 IO。                  forward(t)             (compute 流，并行)
                                         wait_for_event(t+1 only)
                                         [无阻塞，无全局 sync]
```

两层关键改进：

1. **Prefill 时机**：`PrefillScheduler` + `IntentClassifier` 在第一个 decode token 之前批量预加载专家。
2. **事件级同步（M3）**：`promote_to_vram(blocking=False)` 在 `_h2d_stream` 上记录 `torch.cuda.Event`；compute 流通过 `current_stream.wait_event(evt)` **只等所需专家**，不再全局 `stream.synchronize()`。在 30ms 模拟 SSD 延迟下，新路径比旧路径有 **35ms+/token** 的流水线富余空间。

---

## 三级存储架构（M1：cluster-aware safetensors）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VRAM / Metal 统一内存                            │
│  ┌─────────────────┐  ┌──────────────────────────────────────────────┐  │
│  │  Dense 第一层   │  │  共享专家（常驻）                            │  │
│  │  (IntentCLF)    │  │  + 预测热门专家（预加载完毕）                │  │
│  └─────────────────┘  └──────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  H2D：独立 CUDA Stream + per-expert Event
┌──────────────────────────────────▼──────────────────────────────────────┐
│                         Pinned RAM 缓冲区                               │
│         已预取的专家权重（safetensors mmap，页锁定内存）                │
│         AsyncPrefetcher 后台线程从 SSD 读入此处                         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │  按 cluster 顺序读（一次 mmap N 个专家）
┌──────────────────────────────────▼──────────────────────────────────────┐
│                              NVMe SSD                                   │
│   一个 .ctsr 文件 = 一个 Louvain 聚簇（共激活专家打包，连续物理布局）   │
│   配套 cluster_manifest.json 描述 expert→cluster 映射                   │
└─────────────────────────────────────────────────────────────────────────┘
```

`cluster_manifest.json` 与 `.ctsr` 文件由离线 Louvain 聚类生成；运行时一次 `safetensors.safe_open(...).mmap` 把整个簇拉进 RAM，把随机读改写成顺序读。

### 缓存未命中时的处理

即使最坏情况也不会卡顿——柔性降级（Soft Gating）：

```python
# 纯张量乘法，无 Python 分支，torch.compile 不会图断裂
output = avail[i] * expert_output + (1.0 - avail[i]) * shared_expert_output
```

共享专家（常驻 VRAM）按比例混入，生成流**从不中断**，精度平滑降级，专家后台加载完毕后自动恢复。

---

## 双层路由 + 监督的 Lookahead（M2）

| | IntentClassifier（第一层） | LookaheadRouter（第二层） |
|---|---|---|
| **触发时机** | Prefill 阶段一次 | Decode 每个 token |
| **输入** | 完整 Prompt（最多 512 token） | Block 0 的 hidden state |
| **输出** | 整次生成的专家集合 | t+1、t+2 步的专家 ID |
| **训练目标** | 真实激活日志监督（Phase 5） | **L_lookahead — 真实路由 t+k 作为 stop-grad teacher（M2）** |
| **参数量** | ~10–15M（单独训练） | ~2M（随主模型训练） |

M2 之前 LookaheadRouter 没有任何监督——只是个未训练的 head。M2 引入：

```
L_lookahead = (1/K) · Σ_{k=1..K} CE( stopgrad(real_router_{t+k}), lookahead_pred_t^{(k)} )
```

让前瞻头真正学习预测未来 K 步的路由分布。

---

## 完整训练链路（Stage 1 → Stage 6）

每个阶段都是独立的 entry script，**全部继承 Chronos 损失混合器**（lookahead + temporal + balance），并在对齐阶段引入**路由 KL 锚定**防止缓存命中率被 RL/DPO 梯度毁掉。

| Stage | 脚本 | 核心损失 | Router KL 锚定 (默认 λ) |
|---|---|---|---|
| 1 Pretrain  | `train_chronos.py`         | CE + balance + temporal + lookahead | 0.0 (off) |
| 2 SFT       | `train_chronos_sft.py`     | + 上述 mix                          | 0.01 (weak) |
| 3 DPO       | `train_chronos_dpo.py`     | DPO log-σ(β·logits) + mix          | 0.10 (strong) |
| 4 ORPO      | `train_chronos_orpo.py`    | NLL + λ·OR（无 ref model）          | 0.10 |
| 5 GRPO      | `train_chronos_grpo.py`    | PG·A − β·KL（含 ToyReward / 可插 LMRewardModel）| 0.10 |
| 6 Distill   | `train_chronos_distill.py` | α·T²·KL(s‖t) + (1−α)·CE             | 0.05 |

完整 6 阶段端到端对比见 `tools/compare_minimind_chronos_v3.py`。

---

## 多后端调度（M5）

```python
from chronos.backend import BackendDispatcher
d = BackendDispatcher()
d.available()      # ['mlx', 'mps', 'cpu'] on Apple Silicon
                   # ['cuda', 'cpu']        on NVIDIA host
d.select()         # 自动选最佳；可被 CHRONOS_BACKEND 环境变量覆盖
d.describe()       # 人类可读的能力总览
```

- **一等公民（训练 + 推理）**：`cpu`、`mps`、`cuda`、`mlx`
- **推理仅 / 实验性**：`vulkan`（仅当 PyTorch `USE_VULKAN=ON` 自定义构建时存在）
- **第三方插件钩子**：`opencl`（替换 `chronos/backend/ext/opencl.py:PROBE()`）

诚实声明：上游 PyTorch 没有 OpenCL 后端、Vulkan 也仅在自定义构建中可用。Chronos 提供 dispatcher 接缝，使第三方插件无需改核心代码即可接入。

---

## HuggingFace / vLLM 兼容（M5）

- `ChronosForCausalLM` 继承 `PreTrainedModel`，已注册 `AutoConfig` / `AutoModelForCausalLM`，无需 `trust_remote_code`：

  ```python
  from transformers import AutoModelForCausalLM
  model = AutoModelForCausalLM.from_pretrained("./out_dir")
  ```

- `chronos.model.hf_io.save_chronos_pretrained` / `load_chronos_pretrained` 输出标准 `model.safetensors` + `config.json`，并把 `cluster_manifest.json` + `.ctsr` 一起带过去；roundtrip logits 0.00e+00 偏差。

- `chronos.serving.register_chronos_with_vllm()` 在已安装 vLLM 时把 Chronos 注册到 `ModelRegistry`；未安装时打印安装提示，**不报错**。worker 侧 mask 注入 hook 见 `docs/vllm_integration.md`。

---

## 与现有方案对比

| 特性 | llama.cpp offload | vLLM offload | **Project Chronos** |
|---|---|---|---|
| 专家预测 | 无（被动） | 无（被动） | **主动预测（IntentCLF + LookaheadRouter）** |
| Lookahead 训练 | n/a | n/a | **L_lookahead 真实监督（M2）** |
| IO 时机 | Decode 期间（阻塞） | Decode 期间（阻塞） | **Prefill 期间（异步，前置）** |
| Decode 流水线 | 同步 | 同步 | **双流 + per-expert event（M3）** |
| Cache miss 行为 | 硬阻塞 | 硬阻塞 | **Soft Gating（零阻塞）** |
| 磁盘格式 | gguf | safetensors | **cluster-packed safetensors（.ctsr）** |
| 训练集成 | 事后补丁 | 事后补丁 | **6 阶段全栈 + 路由 KL 锚定** |
| 后端调度 | 编译期固定 | CUDA-only | **cpu / mps / cuda / mlx 自动 + vulkan/opencl 钩子** |
| Apple Silicon 原生 | 部分 | 无 | **完整 MLX 后端** |
| HuggingFace 兼容 | 仅 GGUF | ✓ | ✓ + 携带专家缓存 |
| vLLM 兼容 | n/a | 原生 | **可选 adapter（按需注册）** |

---

## 损失函数（完整形式）

$$
L_{\text{total}} = L_{\text{base}} + \lambda_{\text{bal}} L_{\text{balance}} + \lambda_{\text{tmp}} L_{\text{temporal}} + \lambda_{\text{LA}} L_{\text{lookahead}} + \lambda_{\text{anc}} L_{\text{router-KL-anchor}}
$$

- $L_{\text{base}}$：阶段相关（CE / DPO / ORPO / GRPO / KD）
- $L_{\text{temporal}} = \frac{1}{T} \sum_{t=2}^{T} \| E_t - E_{t-1} \|_2^2$ — 鼓励相邻 token 复用同一批专家
- $L_{\text{lookahead}} = \frac{1}{K} \sum_{k=1}^{K} \mathrm{CE}\big(\text{stopgrad}(P_{t+k}),\ Q_t^{(k)}\big)$ — 监督前瞻头
- $L_{\text{router-KL-anchor}} = \mathrm{KL}\big(\pi_\theta^{\text{router}} \,\|\, \pi_{\text{pretrain}}^{\text{router}}\big)$ — 防止对齐阶段毁掉聚簇布局

`λ` 全部支持 Optuna TPE 自动搜索（包括 `hidden_size` / `num_experts` / `kv_latent_dim` 等结构超参）。

---

## 安装  (Not Ready in PyPI Yet)

```bash
pip install project-chronos
```

或从源码：

```bash
git clone https://github.com/your-org/project-chronos
cd project-chronos
pip install -e ".[dev]"
```

**MLX（Apple Silicon）：**
```bash
pip install "project-chronos[mlx]"
```

**vLLM 服务（可选，仅 Linux+CUDA）：**
```bash
pip install vllm
```

> **minimind 依赖**：Project Chronos 使用 [minimind](https://github.com/jingyaogong/minimind) 作为 MoE 内核。
> 若本地未找到，首次 import 时自动克隆至 `~/.cache/chronos/minimind-master/`。
> minimind 采用 **Apache-2.0** 授权，完整归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

**环境要求**：Python 3.10+，PyTorch 2.1+

---

## 快速开始

### Web UI（M6 — 7 个 Tab，4 种语言）

```bash
chronos-ui
# 或
python chronos_app.py
```

包含：⚙️ Config（含右侧实时参数估算面板，合并了 Designer）/ 🏋️ Train（拥有 data_path）/ 🧪 6-Stage Pipeline（每阶段独立数据路径）/ 💬 Inference / 📊 Benchmark（Markdown 表 + BarPlot）/ 🔬 Auto-Tune（持久化日志 + 一键 Apply Best → Config）/ 📡 IO Monitor。i18n 支持 zh-Hans / zh-Hant / en / ja。

### Stage 1：预训练

```bash
python train_chronos.py \
    --data_path ./tests/fixtures/tiny_pretrain.jsonl \
    --hidden_size 256 --num_hidden_layers 4 --num_experts 4 \
    --epochs 1 --device cpu --save_dir ./out
```

### Stage 2-5：对齐链路

```bash
python train_chronos_sft.py   --data_path ./tests/fixtures/tiny_sft.jsonl   --from_weight chronos --save_dir ./out --device cpu
python train_chronos_dpo.py   --data_path ./tests/fixtures/tiny_dpo.jsonl   --from_weight sft     --save_dir ./out --device cpu
python train_chronos_orpo.py  --data_path ./tests/fixtures/tiny_dpo.jsonl   --from_weight sft     --save_dir ./out --device cpu
python train_chronos_grpo.py  --data_path ./tests/fixtures/tiny_grpo.jsonl  --from_weight orpo    --save_dir ./out --device cpu \
    --reward toy   # 或 lm:/path/to/reward-model
```

### Stage 6：蒸馏

```bash
python train_chronos_distill.py \
    --data_path ./tests/fixtures/tiny_sft.jsonl \
    --teacher_path ./out/sft_192_moe.pth \
    --from_weight grpo --save_dir ./out --device cpu \
    --alpha 0.7 --temperature 4.0
```

### 端到端对比（minimind vs Chronos）

```bash
python tools/compare_minimind_chronos_v3.py \
    --pretrain_steps 150 --align_steps 30 --distill_steps 30 \
    --simulated_ssd_ms 30 --device cpu \
    --output results/compare_results_v3.json
```

输出包括：每阶段 loss、HF roundtrip Δlogit、tokens/sec、激活专家比例、常驻专家字节、M3 流水线富余空间、当前主机后端列表。

### 专家聚簇存储（最大化 SSD 顺序读）

```python
from chronos.io.cluster_layout import (
    collect_activation_log, build_cooccurrence_matrix,
    try_louvain_clustering, repack_expert_weights_safetensors,
)
log = collect_activation_log(model, calib_loader, "cpu", max_batches=50)
clusters = try_louvain_clustering(build_cooccurrence_matrix(log, num_experts))
repack_expert_weights_safetensors(model, clusters, "./expert_cache_clustered")
```

### λ + 结构超参自动搜索

```python
from chronos.tuning.chronos_auto_tuner import ChronosAutoTuner, ChronosSearchSpaceConfig

tuner = ChronosAutoTuner()
tuner.start(
    model_id="./out/chronos_256_moe.pth",
    dataset_path="./dataset/train.jsonl",
    search_space=ChronosSearchSpaceConfig(
        tune_lambda_balance=True, tune_lambda_temporal=True,
        tune_lambda_lookahead=True, tune_lookahead_steps=True,
        tune_hidden_size=True, tune_num_experts=True,
        tune_num_shared_experts=True, tune_kv_latent_dim=True,
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
│   ├── __init__.py                # 注册 AutoConfig / AutoModelForCausalLM
│   ├── model/
│   │   ├── config.py              # ChronosConfig（lookahead/temporal/anchor/storage_format 等）
│   │   ├── hybrid_attention.py    # MLAAttention + SlidingWindowAttention
│   │   ├── lookahead_router.py    # 逐 token 前瞻预测器（第二层）
│   │   ├── moe_chronos.py         # ChronosMOEFeedForward + 共享专家 + Soft Gating
│   │   ├── model_chronos.py       # ChronosForCausalLM（_tied_weights_keys 已修补）
│   │   ├── temporal_loss.py       # 时间局部性 + lookahead 监督损失
│   │   └── hf_io.py               # save/load_chronos_pretrained + AutoModel 注册
│   ├── io/
│   │   ├── expert_store.py        # 三级存储 + per-expert Event + 非阻塞 promote
│   │   ├── async_prefetcher.py    # 异步预取（prefetch_only / promote_current 已分离）
│   │   ├── storage.py             # ClusterStorage：.ctsr safetensors + manifest
│   │   ├── cluster_layout.py      # 共现聚簇 + safetensors 重排
│   │   └── io_simulator.py        # CHRONOS_SIM_SSD_MS 测试钩子（M3）
│   ├── router/
│   │   ├── intent_classifier.py   # Prompt 级专家预测器（第一层，~10M 参数）
│   │   ├── expert_predictor.py    # IntentVector → ExpertSet（含预算上限）
│   │   └── prefill_scheduler.py   # 编排 prefill 阶段批量预加载
│   ├── mlx/                       # Apple Silicon 原生后端
│   │   ├── attention.py / moe.py / model.py / expert_store.py / inference.py
│   ├── runtime/
│   │   ├── cache_manager.py       # prefetch_for_next_step / ensure_resident（M3）
│   │   ├── inference_engine.py    # 端到端推理引擎（重排为 H2D-compute 重叠）
│   │   └── metrics.py             # MetricsBus（IO Monitor 数据源）
│   ├── trainer/
│   │   ├── loss_mixin.py          # chronos_loss_term + router_kl_anchor + capture_reference_routing
│   │   ├── chronos_trainer.py     # Pretrain
│   │   ├── sft_trainer.py         # Stage 2
│   │   ├── dpo_trainer.py         # Stage 3
│   │   ├── orpo_trainer.py        # Stage 4（无 ref model）
│   │   ├── grpo_trainer.py        # Stage 5（含 self-contained rollout）
│   │   ├── distill_trainer.py     # Stage 6（KL/T² + α 混合）
│   │   └── reward.py              # ToyReward / LMRewardModel / build_reward_fn
│   ├── tuning/
│   │   └── chronos_auto_tuner.py  # Optuna λ + 结构超参搜索
│   ├── eval/
│   │   ├── io_profiler.py         # Phase 1 验证（前瞻准确率）
│   │   └── benchmark.py           # 端到端基准测试
│   ├── data/
│   │   └── flexible_dataset.py    # 自动识别任意 JSONL 字段格式
│   ├── backend/
│   │   ├── __init__.py            # BackendDispatcher（cpu/mps/cuda/mlx）
│   │   ├── dispatcher.py          # 探针 + 优先级 + 训练能力声明
│   │   └── ext/opencl.py          # 第三方 OpenCL 插件钩子（stub）
│   ├── _backend_legacy.py         # 向后兼容 build_model() 等旧 API
│   ├── serving/
│   │   ├── __init__.py
│   │   └── vllm_adapter.py        # 可选 vLLM 注册（无 vLLM 时优雅降级）
│   └── cli.py                     # 统一 CLI
├── ui/                            # Gradio Web UI（i18n: zh-Hans/zh-Hant/en/ja）
│   ├── i18n.py
│   ├── estimator.py               # 实时参数量/内存估算（与 Config 同步）
│   └── tabs/
│       ├── config_tab.py          # 已合并 Designer，右侧实时估算面板
│       ├── train_tab.py           # 拥有 data_path（不再属于 Config）
│       ├── pipeline_tab.py        # 6 阶段，每段独立 data_path
│       ├── inference_tab.py
│       ├── benchmark_tab.py       # Markdown 表 + gr.BarPlot
│       ├── autotune_tab.py        # 持久化日志 + Apply Best → Config
│       └── iomon_tab.py           # MetricsBus 实时仪表盘
├── chronos_app.py                 # Web UI 入口
├── train_chronos.py               # Stage 1 入口
├── train_chronos_sft.py           # Stage 2 入口
├── train_chronos_dpo.py           # Stage 3 入口
├── train_chronos_orpo.py          # Stage 4 入口
├── train_chronos_grpo.py          # Stage 5 入口
├── train_chronos_distill.py       # Stage 6 入口
├── tools/
│   ├── compare_minimind_chronos.py      # v1 (M1+M2)
│   ├── compare_minimind_chronos_v2.py   # v2 (M3+M4)
│   └── compare_minimind_chronos_v3.py   # v3 (含 6 阶段 + HF roundtrip + 后端报告)
├── tests/
│   ├── test_smoke.py              # 18 个单元测试
│   ├── test_smoke_cuda.py         # 仅 CUDA 主机执行
│   └── fixtures/                  # tiny_pretrain / tiny_sft / tiny_dpo / tiny_grpo
├── docs/
│   └── vllm_integration.md
├── pyproject.toml
└── README.md / README_zh.md / THIRD_PARTY_NOTICES.md
```

---

## 开发路线图

- [x] Phase 1: LookaheadRouter + 时间局部性损失
- [x] Phase 2: 异步 IO 引擎 + 三级存储 + 共现聚簇布局
- [x] Phase 3: 混合注意力（MLA + SlidingWindow）+ PrefillScheduler + 双层路由
- [x] Phase 4: MLX 原生后端、Web UI、CLI、Optuna λ 搜索、开源发布
- [x] **M1**: cluster-aware safetensors 三级存储（替换 .pt pickle）
- [x] **M2**: 真实 lookahead 监督损失（`L_lookahead`）
- [x] **M3**: 双流解码流水线（per-expert event，非阻塞 H2D）
- [x] **M4**: 完整 SFT / DPO / ORPO / GRPO 训练器 + 路由 KL 锚定
- [x] **M5**: HF safetensors I/O + AutoModel 注册 + vLLM adapter + 多后端调度 + Stage 6 蒸馏 + 可插 reward 模型
- [x] **M6**: WebUI v2（合并 Config+Designer、6 阶段独立数据路径、autotune 持久化 & Apply Best、Benchmark BarPlot、IO Monitor）
- [ ] Phase 5（待办）: 大规模激活语料上训练 IntentClassifier；7B+ 规模基准；vLLM worker 端 mask 注入；Vulkan/OpenCL 实际 kernel

---

## 引用

```bibtex
@misc{chronos2026,
  title  = {Project Chronos: Prefill-Time Expert Loading and Dual-Layer Routing
             for Zero-Stall On-Device MoE Inference},
  author = {Fona and Project Chronos Contributors},
  year   = {2026},
  url    = {[project-chronos](https://github.com/FonaTech/Project_Chronos_In_Experimental)}
}
```

---

## 第三方归属

Project Chronos 基于 **jingyaogong** 的 [minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）构建。完整归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
