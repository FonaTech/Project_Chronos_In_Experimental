"""
ui/tabs/inference_tab.py — Real-time generation with tokens/s display
"""
import time

import gradio as gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable


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
            max_tokens  = gr.Slider(16, 512, value=128, step=16, label=t("infer.max_tokens"))
            temperature = gr.Slider(0.1, 2.0, value=0.85, step=0.05, label=t("infer.temperature"))
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

                # MLA: sniff rope_dim from q_rope_proj shape
                #   q_rope_proj.weight: (n_heads * rope_dim, hidden)
                qrope = sd.get("model.layers.0.self_attn.q_rope_proj.weight")
                if qrope is not None and out.get("num_attention_heads_total_dim"):
                    # We know n_heads*head_dim = total_dim, rope_dim divides that
                    # ratio cleanly. Fall back: take (qrope.shape[0] / n_heads_guess).
                    pass  # rope_dim passed in via model_cfg_kwargs below
                return out
            except Exception:
                return {}

        def generate(cfg, model_path_val, prompt_val, max_tok, temp):
            from transformers import AutoTokenizer
            from chronos.model.config import ChronosConfig
            from chronos.backend import get_backend

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
                    "kv_latent_dim":        cfg.get("kv_latent_dim", 64),
                    "sliding_window_size":  cfg.get("sliding_window_size", 2048),
                    "use_hybrid_attention": cfg.get("use_hybrid_attention", True),
                    "vram_budget_gb":       cfg.get("vram_budget_gb", 4.0),
                    "use_moe": True,
                }
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

                backend = get_backend()
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
                    import torch
                    from chronos.model.model_chronos import ChronosForCausalLM

                    model = ChronosForCausalLM(model_cfg)
                    if model_path_val and __import__("os").path.exists(model_path_val):
                        weights = torch.load(model_path_val, map_location="cpu")
                        model.load_state_dict(weights, strict=False)

                    device_map = {"cuda": "cuda", "mps": "mps", "cpu": "cpu"}[backend]
                    model = model.to(device_map).eval()
                    input_ids = torch.tensor([token_ids]).to(device_map)

                    with torch.no_grad():
                        out, lp = model(input_ids, use_cache=True)
                        past_kv = out.past_key_values
                        for _ in range(int(max_tok)):
                            logits = out.logits[:, -1, :] / float(temp)
                            next_tok = torch.multinomial(
                                torch.softmax(logits, dim=-1), num_samples=1
                            )
                            generated.append(next_tok.item())
                            if next_tok.item() == tokenizer.eos_token_id:
                                break
                            out2, lp2 = model(next_tok, past_key_values=past_kv, use_cache=True)
                            past_kv = out2.past_key_values
                            out = out2

                elapsed = time.monotonic() - t0
                tps = len(generated) / max(elapsed, 1e-6)
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                return f"{sniff_note}[{backend}] {decoded}", round(tps, 1)

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
            inputs=[config_state, model_path, prompt, max_tokens, temperature],
            outputs=[output_box, tps_box],
        )

    return tab
