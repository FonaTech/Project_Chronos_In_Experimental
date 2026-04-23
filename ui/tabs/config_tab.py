"""
ui/tabs/config_tab.py — ChronosConfig editor + live param estimator sidebar.

Merges the former Designer tab: the architecture sliders here are the
single source of truth; the right-hand "Estimator" panel reads the same
state and displays parameter counts, memory footprints and estimated
decode tokens/sec in real time.

Data-path is intentionally NOT in this tab — it lives in the Train tab
(and is per-stage in the Pipeline tab) so each workflow owns its own
dataset choice. Only save_dir is shared model-wide.
"""
from ui.gradio_compat import gr

from ui.i18n import t, register_translatable
from ui.estimator import (
    ArchConfig, total_params, active_params, memory_footprint,
    estimated_decode_tps, fmt_bytes, fmt_params,
)
from ui.presets import (
    preset_names, get_preset, values_in_input_order, save_config, load_config,
)


# Shared slider ranges (match Designer v1, which had the more permissive
# bounds). The old Config tab had stricter bounds that surprised users
# toggling between tabs and rejected values they typed manually.
RANGES = {
    "hidden_size":        (8, 16384, 8),
    "num_hidden_layers":  (1, 128, 1),
    "num_experts":        (1, 1024, 1),
    "experts_per_tok":    (1, 32, 1),
    "shared_experts":     (0, 32, 1),
    "lookahead_steps":    (0, 16, 1),
    "kv_latent_dim":      (4, 1024, 4),
    "rope_dim":           (2, 256, 2),
    "sliding_window":     (8, 65536, 8),
    "moe_intermediate":   (8, 65536, 8),
    "vocab_size":         (32, 200000, 32),
}


def _estimate(cfg_dict: dict):
    """Compute estimator outputs from a config dict.

    A moe_intermediate_size of 0 (or missing) is treated as "auto-derive
    from hidden_size", matching MiniMindConfig's default rule. This keeps
    the estimator consistent with what the trainer actually builds.
    """
    mis = int(cfg_dict.get("moe_intermediate_size", 0) or 0)
    its = int(cfg_dict.get("intermediate_size", 0) or 0)
    arch = ArchConfig(
        hidden_size=int(cfg_dict.get("hidden_size", 256)),
        num_hidden_layers=int(cfg_dict.get("num_hidden_layers", 4)),
        num_attention_heads=int(cfg_dict.get("num_attention_heads", 8)),
        num_key_value_heads=int(cfg_dict.get("num_key_value_heads", 4)),
        vocab_size=int(cfg_dict.get("vocab_size", 6400)),
        intermediate_size=its if its > 0 else None,
        moe_intermediate_size=mis if mis > 0 else None,
        num_experts=int(cfg_dict.get("num_experts", 4)),
        num_experts_per_tok=int(cfg_dict.get("num_experts_per_tok", 1)),
        num_shared_experts=int(cfg_dict.get("num_shared_experts", 1)),
        lookahead_steps=int(cfg_dict.get("lookahead_steps", 2)),
        kv_latent_dim=int(cfg_dict.get("kv_latent_dim", 64)),
        rope_dim=int(cfg_dict.get("rope_dim", 32)),
        sliding_window_size=int(cfg_dict.get("sliding_window_size", 2048)),
        use_hybrid_attention=bool(cfg_dict.get("use_hybrid_attention", True)),
        tie_word_embeddings=bool(cfg_dict.get("tie_word_embeddings", True)),
        dtype=str(cfg_dict.get("dtype", "fp16")),
    )
    mem = memory_footprint(arch)
    return (
        fmt_params(total_params(arch)),
        fmt_params(active_params(arch)),
        fmt_bytes(mem["vram_estimate_bytes"]),
        fmt_bytes(mem["ssd_estimate_bytes"]),
        fmt_bytes(mem["kv_cache_bytes"]),
        f"{estimated_decode_tps(arch):.1f}",
    )


def build_config_tab():
    """Returns (config_state, list_of_input_components, save_dir).

    Note: the pre-M6 signature returned (state, inputs, data_path, save_dir).
    data_path has been moved to the Train tab, so the tuple now has 3
    elements. Callers must be updated.
    """
    initial_preset = "Recommended-CN (≈120M)"
    initial_cfg = get_preset(initial_preset)
    config_state = gr.State(dict(initial_cfg))

    with gr.Tab(t("tab.config")) as tab:
        register_translatable(tab, "tab.config")

        # ── Presets / Save / Load (top of tab) ─────────────────
        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=preset_names(),
                value=initial_preset,
                label=t("config.preset"),
                scale=2,
            )
            register_translatable(preset_dd, "config.preset")
            apply_preset_btn = gr.Button(t("config.load_preset"), variant="secondary", scale=1)
            register_translatable(apply_preset_btn, "config.load_preset")
            reset_btn = gr.Button(t("config.reset_minimind"), variant="secondary", scale=1)
            register_translatable(reset_btn, "config.reset_minimind")
        with gr.Row():
            cfg_path = gr.Textbox(
                value="./out/chronos_config.json",
                label=t("config.cfg_path"),
                scale=3,
            )
            register_translatable(cfg_path, "config.cfg_path")
            save_btn = gr.Button(t("config.save_cfg"), variant="primary", scale=1)
            register_translatable(save_btn, "config.save_cfg")
            load_btn = gr.Button(t("config.load_cfg"), variant="primary", scale=1)
            register_translatable(load_btn, "config.load_cfg")
        save_status = gr.Markdown("", visible=True)

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"### {t('config.arch')}")

                D = initial_cfg

                with gr.Row():
                    hidden_size     = gr.Slider(*RANGES["hidden_size"],       value=D["hidden_size"],         label=t("config.hidden_size"))
                    num_layers      = gr.Slider(*RANGES["num_hidden_layers"], value=D["num_hidden_layers"],   label=t("config.num_layers"))
                    num_experts     = gr.Slider(*RANGES["num_experts"],       value=D["num_experts"],         label=t("config.num_experts"))
                    experts_per_tok = gr.Slider(*RANGES["experts_per_tok"],   value=D["num_experts_per_tok"], label=t("config.experts_per_tok"))
                    register_translatable(hidden_size,     "config.hidden_size")
                    register_translatable(num_layers,      "config.num_layers")
                    register_translatable(num_experts,     "config.num_experts")
                    register_translatable(experts_per_tok, "config.experts_per_tok")

                with gr.Row():
                    num_shared     = gr.Slider(*RANGES["shared_experts"],  value=D["num_shared_experts"],  label=t("config.shared_experts"))
                    lookahead      = gr.Slider(*RANGES["lookahead_steps"], value=D["lookahead_steps"],     label=t("config.lookahead"))
                    kv_latent_dim  = gr.Slider(*RANGES["kv_latent_dim"],   value=D["kv_latent_dim"],       label=t("config.kv_latent"))
                    sliding_window = gr.Slider(*RANGES["sliding_window"],  value=D["sliding_window_size"], label=t("config.sliding_window"))
                    register_translatable(num_shared,     "config.shared_experts")
                    register_translatable(lookahead,      "config.lookahead")
                    register_translatable(kv_latent_dim,  "config.kv_latent")
                    register_translatable(sliding_window, "config.sliding_window")

                with gr.Row():
                    num_heads        = gr.Slider(1, 256, value=D["num_attention_heads"], step=1, label=t("config.num_attn_heads"))
                    num_kv_heads     = gr.Slider(1, 256, value=D["num_key_value_heads"], step=1, label=t("config.num_kv_heads"))
                    rope_dim         = gr.Slider(*RANGES["rope_dim"],      value=D["rope_dim"], label=t("config.rope_dim"))
                    moe_intermediate = gr.Slider(
                        0, 65536, value=D["moe_intermediate_size"], step=8,
                        label=t("config.moe_inter"),
                    )
                    register_translatable(num_heads,        "config.num_attn_heads")
                    register_translatable(num_kv_heads,     "config.num_kv_heads")
                    register_translatable(rope_dim,         "config.rope_dim")
                    register_translatable(moe_intermediate, "config.moe_inter")

                with gr.Row():
                    vocab_size = gr.Number(value=D["vocab_size"], precision=0, label=t("config.vocab_size"))
                    dtype      = gr.Dropdown(
                        choices=["fp32", "bf16", "fp16", "int8", "nf4"],
                        value=D["dtype"], label=t("config.dtype"),
                    )
                    tie_word_embeddings = gr.Checkbox(value=D["tie_word_embeddings"], label=t("config.tie_emb"))
                    register_translatable(vocab_size,          "config.vocab_size")
                    register_translatable(dtype,               "config.dtype")
                    register_translatable(tie_word_embeddings, "config.tie_emb")

                gr.Markdown(f"### {t('config.loss')}")
                with gr.Row():
                    lambda_balance       = gr.Number(value=D["lambda_balance"],       label=t("config.lambda_balance"),  precision=6)
                    lambda_temporal      = gr.Number(value=D["lambda_temporal"],      label=t("config.lambda_temporal"), precision=6)
                    lambda_lookahead     = gr.Number(value=D["lambda_lookahead"],     label=t("config.lambda_la"),       precision=6)
                    lambda_router_anchor = gr.Number(value=D["lambda_router_anchor"], label=t("config.lambda_anchor"),   precision=6)
                    register_translatable(lambda_balance,        "config.lambda_balance")
                    register_translatable(lambda_temporal,       "config.lambda_temporal")
                    register_translatable(lambda_lookahead,      "config.lambda_la")
                    register_translatable(lambda_router_anchor,  "config.lambda_anchor")

                gr.Markdown(f"### {t('config.hw')}")
                with gr.Row():
                    vram_budget     = gr.Slider(0.01, 256.0, value=D["vram_budget_gb"],             step=0.01,  label=t("config.vram_budget"))
                    pinned_frac     = gr.Slider(0.01, 0.95,  value=D["pinned_memory_max_fraction"], step=0.01,  label=t("config.pinned_frac"))
                    use_hybrid_attn = gr.Checkbox(value=D["use_hybrid_attention"], label=t("config.hybrid_attn"))
                    storage_format  = gr.Dropdown(choices=["safetensors", "pt"], value=D["storage_format"],
                                                  label=t("config.storage_format"))
                    register_translatable(vram_budget,     "config.vram_budget")
                    register_translatable(pinned_frac,     "config.pinned_frac")
                    register_translatable(use_hybrid_attn, "config.hybrid_attn")
                    register_translatable(storage_format,  "config.storage_format")

                gr.Markdown(f"### {t('config.train')}")
                with gr.Row():
                    learning_rate = gr.Number(value=D["learning_rate"], label=t("config.lr"), precision=6)
                    batch_size    = gr.Slider(1, 512,    value=D["batch_size"],         step=1,   label=t("config.batch_size"))
                    accum_steps   = gr.Slider(1, 256,    value=D["accumulation_steps"], step=1,   label=t("config.accum"))
                    max_seq_len   = gr.Slider(8, 131072, value=D["max_seq_len"],        step=8,   label=t("config.max_seq_len"))
                    register_translatable(learning_rate, "config.lr")
                    register_translatable(batch_size,    "config.batch_size")
                    register_translatable(accum_steps,   "config.accum")
                    register_translatable(max_seq_len,   "config.max_seq_len")

                with gr.Row():
                    epochs        = gr.Slider(1, 1000,    value=D["epochs"],         step=1,   label=t("config.epochs"))
                    save_interval = gr.Slider(10, 100000, value=D["save_interval"],  step=10,  label=t("config.save_interval"))
                    save_dir      = gr.Textbox(label=t("config.save_dir"),           value=D["save_dir"])
                    register_translatable(epochs,        "config.epochs")
                    register_translatable(save_interval, "config.save_interval")
                    register_translatable(save_dir,      "config.save_dir")

                config_display = gr.JSON(label="Current Config (saved to config_state)", value=initial_cfg)

            with gr.Column(scale=1, min_width=260):
                gr.Markdown(f"### 🧬 {t('designer.title')}")
                total_box  = gr.Textbox(label=t("designer.total"),  interactive=False)
                active_box = gr.Textbox(label=t("designer.active"), interactive=False)
                vram_box   = gr.Textbox(label=t("designer.vram"),   interactive=False)
                ssd_box    = gr.Textbox(label=t("designer.ssd"),    interactive=False)
                kv_box     = gr.Textbox(label=t("designer.kv"),     interactive=False)
                tps_box    = gr.Textbox(label=t("designer.tps"),    interactive=False)
                register_translatable(total_box,  "designer.total")
                register_translatable(active_box, "designer.active")
                register_translatable(vram_box,   "designer.vram")
                register_translatable(ssd_box,    "designer.ssd")
                register_translatable(kv_box,     "designer.kv")
                register_translatable(tps_box,    "designer.tps")

        # IMPORTANT: this list also defines the order Autotune uses to push
        # best-params back via gr.update(). Don't reorder without updating
        # ui/tabs/autotune_tab.py.
        all_inputs = [
            hidden_size, num_layers, num_experts, experts_per_tok,
            num_shared, lookahead, kv_latent_dim, sliding_window,
            num_heads, num_kv_heads, rope_dim, moe_intermediate,
            vocab_size, dtype, tie_word_embeddings,
            lambda_balance, lambda_temporal, lambda_lookahead, lambda_router_anchor,
            vram_budget, pinned_frac, use_hybrid_attn, storage_format,
            learning_rate, batch_size, accum_steps, max_seq_len,
            epochs, save_interval, save_dir,
        ]
        estimator_outs = [total_box, active_box, vram_box, ssd_box, kv_box, tps_box]

        def update_config(*vals):
            (hs, nl, ne, ept, ns, la, kv, sw,
             nh, nkv, rd, moe_i,
             vs, dt, tie,
             lb, lt, ll, lra,
             vb, pf, ha, sf,
             lr, bs, ac, msl, ep, si, sd) = vals
            cfg = {
                "hidden_size": int(hs), "num_hidden_layers": int(nl),
                "num_experts": int(ne), "num_experts_per_tok": int(ept),
                "num_shared_experts": int(ns), "lookahead_steps": int(la),
                "kv_latent_dim": int(kv), "sliding_window_size": int(sw),
                "num_attention_heads": int(nh), "num_key_value_heads": int(nkv),
                "rope_dim": int(rd), "moe_intermediate_size": int(moe_i),
                "vocab_size": int(vs), "dtype": str(dt),
                "tie_word_embeddings": bool(tie),
                "lambda_balance": float(lb), "lambda_temporal": float(lt),
                "lambda_lookahead": float(ll), "lambda_router_anchor": float(lra),
                "vram_budget_gb": float(vb), "pinned_memory_max_fraction": float(pf),
                "use_hybrid_attention": bool(ha),
                "storage_format": str(sf),
                "learning_rate": float(lr), "batch_size": int(bs),
                "accumulation_steps": int(ac), "max_seq_len": int(msl),
                "epochs": int(ep), "save_interval": int(si),
                "save_dir": sd,
            }
            est = _estimate(cfg)
            return (cfg, cfg) + est

        for inp in all_inputs:
            inp.change(
                fn=update_config,
                inputs=all_inputs,
                outputs=[config_state, config_display] + estimator_outs,
            )

        # ── Preset / Save / Load wiring ───────────────────────
        def apply_preset(name):
            cfg = get_preset(name)
            return [dict(cfg), dict(cfg)] + [gr.update(value=v) for v in values_in_input_order(cfg)]

        def reset_minimind():
            cfg = get_preset("MiniMind-MoE (default)")
            return [dict(cfg), dict(cfg)] + [gr.update(value=v) for v in values_in_input_order(cfg)]

        def do_save(cfg, path):
            try:
                abs_path = save_config(cfg or {}, path)
                return f"✅ Saved to `{abs_path}`"
            except Exception as e:
                return f"❌ Save failed: {e}"

        def do_load(path):
            try:
                cfg = load_config(path)
                vals = values_in_input_order(cfg)
                return [f"✅ Loaded from `{path}`", dict(cfg), dict(cfg)] + [gr.update(value=v) for v in vals]
            except FileNotFoundError:
                return [f"❌ Not found: `{path}`", gr.update(), gr.update()] + [gr.update() for _ in all_inputs]
            except Exception as e:
                return [f"❌ Load failed: {e}", gr.update(), gr.update()] + [gr.update() for _ in all_inputs]

        apply_preset_btn.click(fn=apply_preset, inputs=[preset_dd], outputs=[config_state, config_display] + all_inputs)
        # Selecting a preset in the dropdown should sync immediately;
        # the explicit "Load Preset" button stays as a re-apply affordance.
        preset_dd.change(fn=apply_preset, inputs=[preset_dd], outputs=[config_state, config_display] + all_inputs)
        reset_btn.click(fn=reset_minimind, outputs=[config_state, config_display] + all_inputs)
        save_btn.click(fn=do_save, inputs=[config_state, cfg_path], outputs=[save_status])
        load_btn.click(fn=do_load, inputs=[cfg_path], outputs=[save_status, config_state, config_display] + all_inputs)

        total_box.value, active_box.value, vram_box.value, ssd_box.value, kv_box.value, tps_box.value = _estimate(initial_cfg)

    return config_state, all_inputs, save_dir
