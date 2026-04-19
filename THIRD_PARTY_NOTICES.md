# Third-Party Notices — Project Chronos

Project Chronos incorporates code and concepts derived from the following
open-source projects. Full license texts are included below or linked.

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
- `ChronosConfig` extends `MiniMindConfig` with new fields (lookahead, hybrid attention, λ parameters)
- `ChronosForCausalLM` replaces `MiniMindForCausalLM` with hybrid attention and LookaheadRouter
- `ChronosMOEFeedForward` extends `MOEFeedForward` with shared experts and soft gating
- `MLAAttention` and `SlidingWindowAttention` are new implementations inspired by MiniMind's `Attention` interface
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

The full Apache-2.0 license text is available at:
https://github.com/jingyaogong/minimind/blob/master/LICENSE

---

## 2. Auto_Fine_Tuning

**Repository**: https://github.com/FonaTech/Auto_Fine_Tuning
**Author**: FonaTech
**License**: Apache License, Version 2.0
**Copyright**: Copyright Fona and Auto_Fine_Tuning Contributors  

**Files derived**:

| Chronos file | Derived from Auto_Fine_Tuning |
|---|---|
| `chronos/tuning/chronos_auto_tuner.py` | `core/auto_tuner.py` — `AutoTuner`, `SearchSpaceConfig` |

**Nature of modifications**:
- `ChronosAutoTuner` subclasses `AutoTuner`, extending `_sample_params()` with λ1/λ2/lookahead_steps search dimensions
- `ChronosSearchSpaceConfig` extends `SearchSpaceConfig` with Chronos-specific fields

---

## 3. Third-Party Python Packages

The following packages are used as dependencies and are **not** bundled:

| Package | License | URL |
|---------|---------|-----|
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| Transformers (HuggingFace) | Apache-2.0 | https://github.com/huggingface/transformers |
| Datasets (HuggingFace) | Apache-2.0 | https://github.com/huggingface/datasets |
| Optuna | MIT | https://github.com/optuna/optuna |
| safetensors | Apache-2.0 | https://github.com/huggingface/safetensors |
| psutil | BSD-3-Clause | https://github.com/giampaolo/psutil |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| python-louvain (optional) | BSD | https://github.com/taynaud/python-louvain |
| networkx (optional) | BSD-3-Clause | https://github.com/networkx/networkx |

---

## 4. Architectural Inspiration

The following published works informed the design of Project Chronos.
No source code was copied; only algorithmic concepts were referenced:

- **DeepSeek-V2 / MLA**: Dai et al., "DeepSeek-V2: A Strong, Economical,
  and Efficient Mixture-of-Experts Language Model", arXiv:2405.04434.
  Multi-head Latent Attention concept used in `MLAAttention`.

- **Qwen2.5 / Sliding Window Attention**: Qwen Team, Alibaba Cloud.
  Hybrid attention layer alternation pattern used in `ChronosBlock`.

- **Mixtral of Experts**: Jiang et al., arXiv:2401.04088.
  MoE routing baseline referenced for performance comparison.

---

## 5. License Compatibility

Project Chronos is released under **Apache-2.0**.

| Dependency | License | Compatible with Apache-2.0? |
|---|---|---|
| MiniMind | Apache-2.0 | ✓ Yes (same license) |
| PyTorch | BSD-3-Clause | ✓ Yes |
| Transformers | Apache-2.0 | ✓ Yes |
| Optuna | MIT | ✓ Yes |
| psutil | BSD-3-Clause | ✓ Yes |

All incorporated dependencies are permissive licenses compatible with
Apache-2.0 distribution. No GPL or LGPL code is incorporated.

---

*This file was generated for Project Chronos v0.1.0 — 2026-04-20*
