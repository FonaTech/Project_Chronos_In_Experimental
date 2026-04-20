# vLLM Integration

Project Chronos exposes a thin adapter for [vLLM](https://github.com/vllm-project/vllm) serving. vLLM is **optional**: Chronos works without it, and it is not a pip dependency.

## When to use

You want high-throughput production serving of a Chronos checkpoint with paged-attention and continuous batching, and you have a Linux host with CUDA.

## Install

```bash
pip install vllm
```

vLLM requires CUDA; on CPU-only or macOS hosts the install will fail. That is upstream vLLM behavior, not a Chronos limitation.

## Register Chronos with vLLM

```python
from chronos.serving import register_chronos_with_vllm
register_chronos_with_vllm()
```

This adds `ChronosForCausalLM` to vLLM's `ModelRegistry` so the architecture can be found by name. It is idempotent and a no-op if vLLM is not installed.

## Serving a Chronos checkpoint

```python
from vllm import LLM, SamplingParams
from chronos.serving import register_chronos_with_vllm

register_chronos_with_vllm()

llm = LLM(model="/path/to/chronos-checkpoint", trust_remote_code=False)
outputs = llm.generate("Hello", SamplingParams(max_tokens=64))
```

The checkpoint must have been saved via `chronos.model.hf_io.save_chronos_pretrained` (HuggingFace safetensors format + `config.json`).

## Expert cache wiring — current limitations

vLLM's worker/scheduler internals change across releases. The Chronos side exposes:

- `ChronosForCausalLM` forward accepts `available_expert_masks` (one per layer) for soft-gating.
- `chronos.serving.set_available_expert_masks(model, masks)` is a hook point a custom vLLM worker can call each step.

But **wiring the async prefetcher + LookaheadRouter into vLLM's per-step loop is not included** — it requires a custom `vllm.worker.worker.Worker` subclass whose shape is version-specific. When run through vanilla vLLM, Chronos will:

- serve correctly using soft gating with whatever experts happen to be in VRAM,
- **not** benefit from lookahead prefetch / cluster-aware IO (that needs the custom worker).

A reference worker will ship in a future release. For now, if you need the full Chronos pipeline, use `chronos.runtime.inference_engine.ChronosInferenceEngine` directly.

## Gotchas

- **Safetensors checkpoints only.** Chronos legacy `.pt` checkpoints need converting via `save_chronos_pretrained` first.
- **`tie_word_embeddings`.** Enabled by default in `ChronosConfig`; HF I/O handles it, but some vLLM versions log warnings about it — harmless.
- **MoE routing.** vLLM's native MoE support varies. Chronos routes through `ChronosMOEFeedForward` which is a plain `nn.Module`; compatible with any vLLM version that can run MoE-style modules but not with vLLM's fused-MoE kernels.
