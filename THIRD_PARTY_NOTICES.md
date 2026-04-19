# Third-Party Notices — Project Chronos

Project Chronos incorporates code and concepts derived from the following
open-source projects and academic works. Full license texts are included
below or linked.

---

## 1. MiniMind

**Repository**: https://github.com/jingyaogong/minimind
**Author**: jingyaogong
**License**: Apache License, Version 2.0
**Copyright**: Copyright jingyaogong and MiniMind Contributors

**Files derived from MiniMind**:

| Chronos file | Derived from MiniMind |
|---|---|
| `chronos/model/config.py` | `model/model_minimind.py` — `MiniMindConfig` |
| `chronos/model/model_chronos.py` | `model/model_minimind.py` — `MiniMindModel`, `MiniMindForCausalLM`, `MiniMindBlock` |
| `chronos/model/moe_chronos.py` | `model/model_minimind.py` — `MOEFeedForward`, `FeedForward` |
| `chronos/model/hybrid_attention.py` | `model/model_minimind.py` — `Attention`, `apply_rotary_pos_emb`, `repeat_kv`, `RMSNorm` |
| `chronos/model/lookahead_router.py` | Architecture inspired by MiniMind's routing design |
| `chronos/trainer/chronos_trainer.py` | `trainer/train_pretrain.py` — training loop structure |
| `chronos/eval/io_profiler.py` | `eval_llm.py` — evaluation patterns |

**Nature of modifications**:
- `ChronosConfig` extends `MiniMindConfig` with new fields: lookahead steps, hybrid attention, λ loss coefficients, VRAM budget, pinned RAM fraction
- `ChronosForCausalLM` replaces `MiniMindForCausalLM` with hybrid attention, LookaheadRouter injection, and dual-layer routing hooks
- `ChronosMOEFeedForward` extends `MOEFeedForward` with always-resident shared experts and compile-safe soft gating (`avail[i] * expert_out + (1−avail[i]) * shared_out`)
- `MLAAttention` and `SlidingWindowAttention` are new implementations sharing RMSNorm and RoPE utilities from MiniMind's `Attention`
- Training loop structure adapted from `train_pretrain.py` with temporal locality loss injection

**Automatic download**: When `minimind-master` is not found locally, Project Chronos
automatically clones it from `https://github.com/jingyaogong/minimind` into
`~/.cache/chronos/minimind-master/`. This download is governed by the Apache-2.0
license of the MiniMind project.

**Apache-2.0 License Text**:

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Full license: https://github.com/jingyaogong/minimind/blob/master/LICENSE

---

## 2. Auto_Fine_Tuning

**Repository**: https://github.com/FonaTech/Auto_Fine_Tuning
**Author**: FonaTech
**License**: Apache License, Version 2.0
**Copyright**: Copyright Fona and Auto_Fine_Tuning Contributors

**Files derived from Auto_Fine_Tuning**:

| Chronos file | Derived from Auto_Fine_Tuning |
|---|---|
| `chronos/tuning/_base_tuner.py` | `core/auto_tuner.py` — `AutoTuner`, `SearchSpaceConfig`, event queue, Optuna loop, probe trial mechanism |
| `chronos/tuning/chronos_auto_tuner.py` | `core/auto_tuner.py` — `AutoTuner` subclass pattern |

**Nature of modifications**:
- `_base_tuner.py` vendors the `AutoTuner` base class skeleton (event queue, Optuna TPE loop, MedianPruner early stopping, `start`/`stop`/`poll` interface) so Chronos has no runtime dependency on the sibling `Auto_Fine_Tuning/` directory
- `ChronosAutoTuner` subclasses `AutoTuner`, extending `_sample_params()` with Chronos-specific search dimensions: `lambda_temporal`, `lambda_balance`, `lookahead_steps`, `num_shared_experts`
- `ChronosSearchSpaceConfig` extends `SearchSpaceConfig` with Chronos-specific boolean flags

**Apache-2.0 License Text**: same as Section 1 above.

Full license: https://github.com/FonaTech/Auto_Fine_Tuning/blob/master/LICENSE

---

## 3. Third-Party Python Packages

The following packages are used as runtime dependencies and are **not** bundled with
Project Chronos source. They are installed separately via `pip`.

| Package | License | URL |
|---------|---------|-----|
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| Transformers (HuggingFace) | Apache-2.0 | https://github.com/huggingface/transformers |
| Datasets (HuggingFace) | Apache-2.0 | https://github.com/huggingface/datasets |
| Optuna | MIT | https://github.com/optuna/optuna |
| Gradio | Apache-2.0 | https://github.com/gradio-app/gradio |
| safetensors | Apache-2.0 | https://github.com/huggingface/safetensors |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| matplotlib | PSF / BSD-style | https://github.com/matplotlib/matplotlib |
| MLX (optional) | MIT | https://github.com/ml-explore/mlx |
| python-louvain (optional) | BSD | https://github.com/taynaud/python-louvain |
| networkx (optional) | BSD-3-Clause | https://github.com/networkx/networkx |

---

## 4. Academic References and Algorithmic Inspiration

No source code was copied from the following works. Algorithmic concepts,
architectural patterns, and mathematical formulations were studied and
independently re-implemented.

### 4.1 Attention Mechanisms

**Transformer (original)**
> Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
> https://arxiv.org/abs/1706.03762

Foundational multi-head self-attention, positional encoding, and encoder-decoder
architecture. Project Chronos uses the decoder-only variant.

**Rotary Position Embedding (RoPE)**
> Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." 2021.
> https://arxiv.org/abs/2104.09864

Applied in `chronos/model/hybrid_attention.py` (`apply_rotary_pos_emb`) and
`chronos/mlx/attention.py` (`_apply_rope`). RoPE replaces learned absolute
positional embeddings with rotation-based relative encoding.

**Multi-head Latent Attention (MLA)**
> Dai et al. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
> Language Model." arXiv:2405.04434, 2024.
> https://arxiv.org/abs/2405.04434

The MLA mechanism introduced in DeepSeek-V2 compresses K/V projections through
a shared low-rank latent vector, reducing KV cache size by 8–16×. `MLAAttention`
in `chronos/model/hybrid_attention.py` and `chronos/mlx/attention.py` is
directly inspired by this design.

**Sliding Window Attention**
> Beltagy et al. "Longformer: The Long-Document Transformer." arXiv:2004.05150, 2020.
> https://arxiv.org/abs/2004.05150
> Also: Mistral 7B (Jiang et al., arXiv:2310.06825, 2023).

Local attention restricted to a fixed window, reducing memory complexity from
O(n²) to O(n·w). Used in `SlidingWindowAttention` in
`chronos/model/hybrid_attention.py`. The alternating MLA/SlidingWindow pattern
per layer is inspired by Mistral and Qwen2.5.

**Grouped-Query Attention (GQA)**
> Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models from
> Multi-Head Checkpoints." arXiv:2305.13245, 2023.
> https://arxiv.org/abs/2305.13245

KV head sharing via `repeat_kv` reduces memory bandwidth. Used in both attention
variants to support `num_key_value_heads < num_attention_heads`.

### 4.2 Mixture of Experts

**Sparse MoE (original Transformer MoE)**
> Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated
> Mixture-of-Experts Layer." ICLR 2017.
> https://arxiv.org/abs/1701.06538

Foundational gated MoE layer with Top-K routing and load balancing auxiliary loss.
`ChronosMOEFeedForward` builds on this routing paradigm.

**Switch Transformer load balancing**
> Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models with
> Simple and Efficient Sparsity." JMLR 2022.
> https://arxiv.org/abs/2101.03961

The auxiliary load balance loss $L_{\text{balance}}$ (coefficient λ₁ in Chronos)
follows the Switch Transformer formulation to prevent expert collapse.

**Mixtral of Experts**
> Jiang et al. "Mixtral of Experts." arXiv:2401.04088, 2024.
> https://arxiv.org/abs/2401.04088

Practical sparse MoE at production scale. Referenced as the primary performance
baseline for the "reactive offload" decode pattern that Chronos replaces.

**DeepSeek-MoE fine-grained expert segmentation**
> Dai et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in
> Mixture-of-Experts Language Models." arXiv:2401.06066, 2024.
> https://arxiv.org/abs/2401.06066

Shared expert concept (always-active experts separate from routed experts) directly
inspired the `num_shared_experts` design in `ChronosMOEFeedForward`.

### 4.3 Speculative / Lookahead Decoding

**Speculative Decoding**
> Leviathan et al. "Fast Inference from Transformers via Speculative Decoding."
> ICML 2023. https://arxiv.org/abs/2211.17192

The general principle of using a small model to predict and pre-execute future
computation. `LookaheadRouter` adapts this to predict *which experts* are needed
at t+1, t+2 rather than predicting future token logits.

**Lookahead Decoding**
> Fu et al. "Break the Sequential Dependency of LLM Inference Using Lookahead
> Decoding." arXiv:2402.02057, 2024.
> https://arxiv.org/abs/2402.02057

Conceptual inspiration for the per-token lookahead routing window.

### 4.4 Normalization and Feed-Forward Design

**RMSNorm**
> Zhang & Sennrich. "Root Mean Square Layer Normalization." NeurIPS 2019.
> https://arxiv.org/abs/1910.07467

Used throughout in place of LayerNorm for improved training stability and lower
compute. Implemented in `chronos/model/hybrid_attention.py` (RMSNorm class) and
`chronos/mlx/model.py` (RMSNormMLX).

**SwiGLU activation**
> Shazeer. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.
> https://arxiv.org/abs/2002.05202

Feed-forward layers use the SwiGLU variant (gate × SiLU(x) × W) for improved
expressivity. Applied in `FeedForward` (PyTorch) and `FeedForwardMLX`.

### 4.5 Bayesian Hyperparameter Optimization

**Optuna / TPE**
> Akiba et al. "Optuna: A Next-generation Hyperparameter Optimization Framework."
> KDD 2019. https://arxiv.org/abs/1907.10902

Tree-structured Parzen Estimator (TPE) used in `ChronosAutoTuner` to search
λ₁ (load balance), λ₂ (temporal locality), `lookahead_steps`, and `num_shared_experts`.

### 4.6 Memory and IO Optimization

**Pinned memory and async H2D transfers**
> NVIDIA CUDA Programming Guide — Asynchronous Concurrent Execution.
> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution

Dedicated CUDA streams in `ExpertStore.promote_to_vram()` allow H2D weight copies
to overlap with the default compute stream, implementing the principle of
asynchronous concurrent execution.

**MLX unified memory**
> Hannun et al. "MLX: An Array Framework for Apple Silicon." 2023.
> https://github.com/ml-explore/mlx

On Apple Silicon, CPU and GPU share physical memory. `MLXExpertStore` exploits
this by using `mx.eval()` to materialise arrays in Metal without explicit H2D
copies, eliminating the memory bandwidth bottleneck that motivates the CUDA
stream isolation above.

---

## 5. License Compatibility

Project Chronos is released under **Apache-2.0**.

| Dependency | License | Compatible with Apache-2.0? |
|---|---|---|
| MiniMind | Apache-2.0 | ✓ Yes (same license) |
| Auto_Fine_Tuning | Apache-2.0 | ✓ Yes (same license) |
| PyTorch | BSD-3-Clause | ✓ Yes |
| Transformers | Apache-2.0 | ✓ Yes |
| Optuna | MIT | ✓ Yes |
| Gradio | Apache-2.0 | ✓ Yes |
| MLX | MIT | ✓ Yes |
| psutil | BSD-3-Clause | ✓ Yes |
| matplotlib | PSF / BSD-style | ✓ Yes |
| python-louvain | BSD | ✓ Yes |
| networkx | BSD-3-Clause | ✓ Yes |

All incorporated code and dependencies carry permissive licenses compatible with
Apache-2.0 distribution. No GPL, LGPL, or copyleft code is incorporated or bundled.

---

*This file was last updated: 2026-04-20*
