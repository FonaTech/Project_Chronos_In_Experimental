"""
ui/tabs/inference_tab.py — Real-time generation with tokens/s display
"""
import time

from ui.gradio_compat import gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable


INFERENCE_BACKEND_CHOICES = ("mlx", "cuda", "xpu", "mps", "cpu")
TORCH_INFERENCE_PRIORITY = ("cuda", "xpu", "mps", "cpu")


def _best_torch_inference_backend() -> str:
    from chronos.backend import BackendDispatcher

    d = BackendDispatcher()
    for name in TORCH_INFERENCE_PRIORITY:
        if d.info(name).available:
            return name
    return "cpu"


def _available_inference_backend_choices() -> list[str]:
    from chronos.backend import available

    detected = set(available())
    return ["auto"] + [name for name in INFERENCE_BACKEND_CHOICES if name in detected]


def _default_inference_backend_value() -> str:
    return "auto"


def _resolve_inference_backend(requested_backend: str, model_path_val: str, sniffed: dict) -> tuple[str, str]:
    from chronos.backend import select

    requested = (requested_backend or "auto").strip().lower() or "auto"
    backend = select(None if requested == "auto" else requested)
    note = ""

    if requested not in {"", "auto"} and backend != requested:
        note = (
            f"[backend fallback] Requested {requested}, but it is not available; "
            f"using {backend}.\n"
        )

    # Current MLX inference in this repo does not have a lossless
    # PyTorch-.pth loader. If we detected a standard Chronos checkpoint,
    # prefer a torch backend so the weights actually load.
    if backend == "mlx" and sniffed:
        fallback = _best_torch_inference_backend()
        if fallback != "mlx":
            backend = fallback
            note = note + (
                f"[backend fallback] Detected a PyTorch Chronos checkpoint at {model_path_val}; "
                f"using {backend} because the current MLX .pth import path is not lossless.\n"
            )
    return backend, note


def _checkpoint_expert_cache_dir(model_path_val: str, model_cfg) -> str:
    """Keep lazy expert shards isolated per checkpoint.

    Reusing a single ./expert_cache across different .pth files can silently
    load stale expert shards. The cache key uses path + stat metadata + core
    topology so repeated runs reuse the same shards, while changed checkpoints
    get a clean cache namespace.
    """
    import hashlib
    import os

    base = os.path.join(os.getcwd(), "expert_cache")
    if not model_path_val or not os.path.exists(model_path_val):
        return base
    try:
        st = os.stat(model_path_val)
    except OSError:
        return base
    topology = (
        getattr(model_cfg, "vocab_size", ""),
        getattr(model_cfg, "hidden_size", ""),
        getattr(model_cfg, "num_hidden_layers", ""),
        getattr(model_cfg, "num_experts", ""),
        getattr(model_cfg, "moe_intermediate_size", ""),
    )
    raw = f"{os.path.abspath(model_path_val)}:{st.st_size}:{st.st_mtime_ns}:{topology}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return os.path.join(base, f"ckpt_{key}")


def _replace_torch_experts_with_placeholders(model, model_cfg) -> None:
    """Remove randomly initialised live experts before model.to(device).

    For .pth inference the real expert tensors are sharded from the checkpoint
    to SSD by ChronosInferenceEngine.setup_from_state_dict(). Keeping the
    constructor's random experts alive until model.to(mps/cuda) defeats the
    SSD+DRAM lazy-load design by moving full dummy experts onto the device.
    """
    import torch
    from chronos.model.moe_chronos import ChronosMOEFeedForward, LazyExpertPlaceholder

    layers = getattr(getattr(model, "model", None), "layers", [])
    hidden = int(getattr(model_cfg, "hidden_size", 0) or 0)
    intermediate = int(getattr(model_cfg, "moe_intermediate_size", 0) or 0)
    if not hidden or not intermediate:
        return
    for layer in layers:
        moe = getattr(layer, "mlp", None)
        if not isinstance(moe, ChronosMOEFeedForward):
            continue
        for idx in range(len(moe.experts)):
            moe.experts[idx] = LazyExpertPlaceholder(hidden, intermediate, torch.float16)


def _run_torch_inference(backend: str, model_cfg, model_path_val: str, token_ids: list[int],
                         max_tok: int, temp: float, eos_token_id: int | None) -> list[int]:
    import os
    import torch
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.runtime.inference_engine import ChronosInferenceEngine

    model = ChronosForCausalLM(model_cfg)
    weights = None
    if model_path_val and os.path.exists(model_path_val):
        weights = torch.load(model_path_val, map_location="cpu")
        base_only = {
            k: v for k, v in weights.items()
            if ".mlp.experts." not in k
        }
        model.load_state_dict(base_only, strict=False)
        _replace_torch_experts_with_placeholders(model, model_cfg)

    device_map = {"cuda": "cuda", "xpu": "xpu", "mps": "mps", "cpu": "cpu"}[backend]
    model = model.to(device_map).eval()
    input_ids = torch.tensor([token_ids]).to(device_map)
    engine = ChronosInferenceEngine(
        model,
        model_cfg,
        ssd_dir=_checkpoint_expert_cache_dir(model_path_val, model_cfg),
    )
    if weights is not None:
        engine.setup_from_state_dict(weights, warm_expert_ids=[])
        del weights
        weights = None
        import gc
        gc.collect()
    else:
        engine.setup(warm_expert_ids=[])

    try:
        out = engine.generate(
            input_ids,
            max_new_tokens=int(max_tok),
            temperature=float(temp),
            eos_token_id=eos_token_id,
        )
    finally:
        engine.teardown()

    prompt_len = input_ids.shape[1]
    generated = out[0, prompt_len:].tolist()

    return generated


def build_inference_tab(config_state: gr.State):
    with gr.Tab(t("tab.inference")) as tab:
        register_translatable(tab, "tab.inference")

        model_path = gr.Textbox(
            label=t("infer.model_path"), placeholder="./out/chronos_512_moe.pth"
        )
        register_translatable(model_path, "infer.model_path")

        prompt = gr.Textbox(
            label=t("infer.prompt"), lines=4,
            placeholder="Explain how Chronos handles expert prefetch during prefill..."
        )
        register_translatable(prompt, "infer.prompt")

        with gr.Row():
            inference_backend = gr.Dropdown(
                choices=_available_inference_backend_choices(),
                value=_default_inference_backend_value(),
                label=t("infer.backend"),
            )
            max_tokens  = gr.Slider(16, 512, value=128, step=16, label=t("infer.max_tokens"))
            temperature = gr.Slider(0.1, 2.0, value=0.85, step=0.05, label=t("infer.temperature"))
            register_translatable(inference_backend, "infer.backend")
            register_translatable(max_tokens,  "infer.max_tokens")
            register_translatable(temperature, "infer.temperature")

        gen_btn = gr.Button(t("infer.generate"), variant="primary")
        register_translatable(gen_btn, "infer.generate")

        output_box = gr.Textbox(label=t("infer.output"), lines=10, interactive=False)
        tps_box    = gr.Number(label=t("infer.tps"), value=0.0, interactive=False, precision=1)
        register_translatable(output_box, "infer.output")
        register_translatable(tps_box,    "infer.tps")

        def _sniff_checkpoint(path: str) -> dict:
            """Inspect a Chronos .pth and return the topology fields encoded
            in the saved tensor shapes. Returns {} if the file is unreadable
            or doesn't look like a Chronos checkpoint.

            Sniffed fields:
                vocab_size, hidden_size, num_hidden_layers, num_experts,
                moe_intermediate_size, num_attention_heads, num_key_value_heads.
            """
            import os as _os
            if not path or not _os.path.exists(path):
                return {}
            try:
                import torch as _t
                sd = _t.load(path, map_location="cpu")
                if not isinstance(sd, dict):
                    return {}
                out = {}

                embed = sd.get("model.embed_tokens.weight")
                if embed is not None:
                    out["vocab_size"], out["hidden_size"] = int(embed.shape[0]), int(embed.shape[1])

                # Count layers via highest layer index in keys.
                layer_idxs = set()
                for k in sd.keys():
                    if k.startswith("model.layers."):
                        try:
                            layer_idxs.add(int(k.split(".")[2]))
                        except (ValueError, IndexError):
                            pass
                if layer_idxs:
                    out["num_hidden_layers"] = max(layer_idxs) + 1

                # Count experts on layer 0.
                expert_idxs = set()
                for k in sd.keys():
                    if k.startswith("model.layers.0.mlp.experts."):
                        try:
                            expert_idxs.add(int(k.split(".")[5]))
                        except (ValueError, IndexError):
                            pass
                if expert_idxs:
                    out["num_experts"] = max(expert_idxs) + 1

                # Per-expert FFN width from gate_proj shape (intermediate, hidden).
                gate = sd.get("model.layers.0.mlp.experts.0.gate_proj.weight")
                if gate is not None:
                    out["moe_intermediate_size"] = int(gate.shape[0])

                # Attention head counts from q/k_proj shapes.
                qw = sd.get("model.layers.0.self_attn.q_proj.weight")
                kw = sd.get("model.layers.0.self_attn.k_proj.weight")
                if qw is not None and kw is not None and out.get("hidden_size"):
                    h = out["hidden_size"]
                    # qw: (n_heads * head_dim, hidden); kw: (n_kv * head_dim, hidden)
                    # head_dim is typically hidden_size // n_heads, but we can't
                    # split q without knowing head_dim. Conservative: assume the
                    # ratio matches MiniMind defaults (head_dim = h // num_heads).
                    # Solve via the kw / qw ratio assuming both share head_dim.
                    out["num_attention_heads_total_dim"] = int(qw.shape[0])
                    out["num_kv_heads_total_dim"] = int(kw.shape[0])

                # MLA attention uses split query and latent KV projections:
                #   q_nope_proj: (n_heads * nope_dim, hidden)
                #   q_rope_proj: (n_heads * rope_dim, hidden)
                #   kv_down_proj: (kv_latent_dim, hidden)
                #   v_proj: (n_kv_heads * head_dim, kv_latent_dim)
                qnope = sd.get("model.layers.0.self_attn.q_nope_proj.weight")
                qrope = sd.get("model.layers.0.self_attn.q_rope_proj.weight")
                kvdown = sd.get("model.layers.0.self_attn.kv_down_proj.weight")
                vproj = sd.get("model.layers.0.self_attn.v_proj.weight")
                if qnope is not None and qrope is not None and out.get("hidden_size"):
                    total_q = int(qnope.shape[0] + qrope.shape[0])
                    h = out["hidden_size"]
                    if total_q > 0 and h % total_q == 0:
                        # For MiniMind/Chronos checkpoints head_dim is
                        # hidden_size / num_attention_heads, so total_q equals
                        # hidden_size and n_heads is recovered by divisibility.
                        pass
                    candidates = [
                        n for n in range(1, out["hidden_size"] + 1)
                        if out["hidden_size"] % n == 0
                        and int(qrope.shape[0]) % n == 0
                        and int(qnope.shape[0]) % n == 0
                    ]
                    if candidates:
                        # Prefer the configured/default 8 when valid; else
                        # choose the largest plausible head count.
                        preferred = 8 if 8 in candidates else max(candidates)
                        head_dim = out["hidden_size"] // preferred
                        rope_dim = int(qrope.shape[0]) // preferred
                        nope_dim = int(qnope.shape[0]) // preferred
                        if rope_dim + nope_dim == head_dim:
                            out["num_attention_heads"] = preferred
                            out["rope_dim"] = rope_dim
                            out["num_attention_heads_total_dim"] = preferred * head_dim
                if kvdown is not None:
                    out["kv_latent_dim"] = int(kvdown.shape[0])
                if vproj is not None and out.get("kv_latent_dim"):
                    head_dim = out.get("hidden_size", 0) // max(out.get("num_attention_heads", 8), 1)
                    if head_dim > 0:
                        out["num_key_value_heads"] = max(1, int(vproj.shape[0]) // head_dim)
                        out["num_kv_heads_total_dim"] = int(vproj.shape[0])

                # Odd layers are SlidingWindowAttention in the hybrid stack.
                # If layer 1 exists, its q/k projection shapes are a direct
                # source of head counts and should override weaker defaults.
                sw_q = sd.get("model.layers.1.self_attn.q_proj.weight")
                sw_k = sd.get("model.layers.1.self_attn.k_proj.weight")
                if sw_q is not None and sw_k is not None and out.get("hidden_size"):
                    out["num_attention_heads_total_dim"] = int(sw_q.shape[0])
                    out["num_kv_heads_total_dim"] = int(sw_k.shape[0])
                return out
            except Exception:
                return {}

        def generate(cfg, selected_backend, model_path_val, prompt_val, max_tok, temp):
            from transformers import AutoTokenizer
            from chronos.model.config import ChronosConfig

            if not prompt_val.strip():
                return "Please enter a prompt.", 0.0

            try:
                # Forward ALL Config-tab fields so the model topology built
                # here matches the topology the user actually configured.
                # Hardcoding only hidden_size/num_layers/num_experts caused
                # checkpoint↔model mismatches like:
                #   "matmul (1,1,768) × (256,6400)"
                # when a checkpoint trained at H=768 was loaded into a
                # H=256-default ChronosConfig.
                cfg = cfg or {}
                # ── Auto-derive topology from the checkpoint ──
                # If a model path is given, sniff the saved tensor shapes
                # and override the Config fields with what the checkpoint
                # actually requires. This makes inference "just work"
                # regardless of which preset is loaded in the Config tab.
                sniffed = _sniff_checkpoint(model_path_val) if model_path_val else {}
                sniff_note = ""
                if sniffed:
                    summary = ", ".join(
                        f"{k}={v}" for k, v in sniffed.items()
                        if k in ("vocab_size", "hidden_size", "num_hidden_layers",
                                 "num_experts", "moe_intermediate_size")
                    )
                    sniff_note = f"[sniffed from {model_path_val}: {summary}]\n"

                model_cfg_kwargs = {
                    "hidden_size":          sniffed.get("hidden_size",        cfg.get("hidden_size", 768)),
                    "num_hidden_layers":    sniffed.get("num_hidden_layers",  cfg.get("num_hidden_layers", 8)),
                    "num_experts":          sniffed.get("num_experts",        cfg.get("num_experts", 4)),
                    "num_experts_per_tok":  cfg.get("num_experts_per_tok", 1),
                    "num_shared_experts":   cfg.get("num_shared_experts", 1),
                    "lookahead_steps":      cfg.get("lookahead_steps", 2),
                    "kv_latent_dim":        sniffed.get("kv_latent_dim", cfg.get("kv_latent_dim", 64)),
                    "sliding_window_size":  cfg.get("sliding_window_size", 2048),
                    "use_hybrid_attention": cfg.get("use_hybrid_attention", True),
                    "vram_budget_gb":       cfg.get("vram_budget_gb", 4.0),
                    "use_moe": True,
                }
                if "rope_dim" in sniffed:
                    model_cfg_kwargs["rope_dim"] = sniffed["rope_dim"]
                if "num_attention_heads" in sniffed:
                    model_cfg_kwargs["num_attention_heads"] = sniffed["num_attention_heads"]
                if "num_key_value_heads" in sniffed:
                    model_cfg_kwargs["num_key_value_heads"] = sniffed["num_key_value_heads"]
                if "vocab_size" in sniffed:
                    model_cfg_kwargs["vocab_size"] = sniffed["vocab_size"]
                if "moe_intermediate_size" in sniffed:
                    model_cfg_kwargs["intermediate_size"] = sniffed["moe_intermediate_size"]
                    model_cfg_kwargs["moe_intermediate_size"] = sniffed["moe_intermediate_size"]

                for opt_key, cfg_key in [
                    ("num_attention_heads",   "num_attention_heads"),
                    ("num_key_value_heads",   "num_key_value_heads"),
                    ("rope_dim",              "rope_dim"),
                    ("max_position_embeddings", "max_seq_len"),
                    ("tie_word_embeddings",   "tie_word_embeddings"),
                ]:
                    if opt_key in {"num_attention_heads", "num_key_value_heads", "rope_dim"} \
                            and opt_key in model_cfg_kwargs:
                        continue
                    if cfg_key not in cfg:
                        continue
                    val = cfg[cfg_key]
                    if val in (None, "", 0):
                        continue
                    model_cfg_kwargs[opt_key] = val

                # If checkpoint sniffing revealed attention dims, adjust heads
                # to match. We prefer the sniffed q/kv totals when the derived
                # head_dim would otherwise be inconsistent.
                if "num_attention_heads_total_dim" in sniffed:
                    h = model_cfg_kwargs["hidden_size"]
                    n_heads = model_cfg_kwargs.get("num_attention_heads", cfg.get("num_attention_heads", 8))
                    head_dim = h // n_heads
                    if head_dim > 0:
                        correct_n_heads = sniffed["num_attention_heads_total_dim"] // head_dim
                        correct_n_kv    = sniffed["num_kv_heads_total_dim"]       // head_dim
                        if correct_n_heads > 0:
                            model_cfg_kwargs["num_attention_heads"] = correct_n_heads
                        if correct_n_kv > 0:
                            model_cfg_kwargs["num_key_value_heads"] = correct_n_kv

                model_cfg = ChronosConfig(**model_cfg_kwargs)

                backend, backend_note = _resolve_inference_backend(selected_backend, model_path_val, sniffed)
                tokenizer = AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())
                token_ids = tokenizer.encode(prompt_val)
                generated = []
                t0 = time.monotonic()

                if backend == "mlx":
                    import mlx.core as mx
                    from chronos.mlx.model import ChronosMLXModel
                    from chronos.mlx.inference import ChronosMLXInferenceEngine

                    model = ChronosMLXModel(model_cfg)
                    if model_path_val and __import__("os").path.exists(model_path_val):
                        import numpy as np, torch
                        sd = torch.load(model_path_val, map_location="cpu")
                        # best-effort weight copy via numpy bridge
                        for k, v in sd.items():
                            arr = mx.array(v.float().numpy())
                            # walk model attributes by key path
                            try:
                                parts = k.split(".")
                                obj = model
                                for p in parts[:-1]:
                                    obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
                                setattr(obj, parts[-1], arr)
                            except Exception:
                                pass

                    engine = ChronosMLXInferenceEngine(model, model_cfg)
                    input_arr = mx.array([token_ids])
                    for tok_id in engine.generate(input_arr, max_new_tokens=int(max_tok),
                                                  temperature=float(temp)):
                        generated.append(tok_id)
                        if tok_id == tokenizer.eos_token_id:
                            break
                    engine.stop()

                else:
                    generated = _run_torch_inference(
                        backend,
                        model_cfg,
                        model_path_val,
                        token_ids,
                        int(max_tok),
                        float(temp),
                        tokenizer.eos_token_id,
                    )

                elapsed = time.monotonic() - t0
                tps = len(generated) / max(elapsed, 1e-6)
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                return f"{backend_note}{sniff_note}[{backend}] {decoded}", round(tps, 1)

            except Exception as e:
                import traceback
                msg = str(e)
                hint = ""
                # Detect the common "config doesn't match checkpoint" case and
                # inspect the saved weights to tell the user exactly which
                # hidden_size/vocab_size the checkpoint expects.
                if ("matmul" in msg or "shape" in msg.lower() or "size mismatch" in msg.lower()) \
                        and model_path_val and __import__("os").path.exists(model_path_val):
                    try:
                        import torch as _t
                        sd = _t.load(model_path_val, map_location="cpu")
                        if isinstance(sd, dict):
                            embed = sd.get("model.embed_tokens.weight")
                            if embed is not None:
                                vocab, h = tuple(embed.shape)
                                hint = (
                                    f"\n\nHINT: the checkpoint `{model_path_val}` was "
                                    f"saved with vocab_size={vocab}, hidden_size={h}. "
                                    f"Set these in the Config tab (and click Save Config), "
                                    f"then retry."
                                )
                    except Exception:
                        pass
                return f"Error: {msg}{hint}\n{traceback.format_exc()}", 0.0

        gen_btn.click(
            fn=generate,
            inputs=[config_state, inference_backend, model_path, prompt, max_tokens, temperature],
            outputs=[output_box, tps_box],
        )

    return tab
