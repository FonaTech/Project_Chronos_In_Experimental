"""
ui/tabs/autotune_tab.py — Optuna search over Chronos hyperparameters.

Improvements over the M5 version:

- **Log accumulation**: events append to a rolling buffer, persisted to
  ``./auto_tune_cache/tune_log.txt`` so history survives a UI refresh.
- **Best-params history**: every trial's params + score appended to
  ``./auto_tune_cache/trials.jsonl``.
- **Expanded search space**: structural knobs (hidden_size, num_experts,
  num_shared_experts, kv_latent_dim, λ_lookahead) can be ticked for search.
- **Apply Best → Config**: one button writes the best params back into
  the Config tab's slider widgets.
"""
import json
import os
import time

from ui.gradio_compat import gr

import chronos.deps  # noqa: F401
from ui.i18n import t, register_translatable
from chronos.tuning.chronos_auto_tuner import ChronosAutoTuner, ChronosSearchSpaceConfig


_tuner = ChronosAutoTuner()
_CACHE_DIR = os.path.abspath("./auto_tune_cache")
_LOG_FILE = os.path.join(_CACHE_DIR, "tune_log.txt")
_TRIALS_FILE = os.path.join(_CACHE_DIR, "trials.jsonl")


def _ensure_cache():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _append_log(text: str):
    _ensure_cache()
    with open(_LOG_FILE, "a") as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))


def _read_log() -> str:
    if not os.path.exists(_LOG_FILE):
        return ""
    with open(_LOG_FILE) as f:
        return f.read()


def _append_trial(record: dict):
    _ensure_cache()
    with open(_TRIALS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# Map best-params keys to indices into the Config tab's all_inputs list.
# IMPORTANT: must stay in sync with ui/tabs/config_tab.py:all_inputs order.
PARAM_TO_CONFIG_IDX = {
    "hidden_size":          0,
    "num_hidden_layers":    1,
    "num_experts":          2,
    "num_experts_per_tok":  3,
    "num_shared_experts":   4,
    "lookahead_steps":      5,
    "kv_latent_dim":        6,
    "sliding_window_size":  7,
    "lambda_balance":       15,
    "lambda_temporal":      16,
    "lambda_lookahead":     17,
    "lambda_router_anchor": 18,
    "learning_rate":        23,
    "batch_size":           24,
}


def build_autotune_tab(config_state: gr.State, config_inputs: list):
    with gr.Tab(t("tab.autotune")) as tab:
        register_translatable(tab, "tab.autotune")

        gr.Markdown("### Optuna TPE — search over Chronos hyperparameters")

        with gr.Row():
            data_path   = gr.Textbox(label=t("autotune.data_path"), scale=3,
                                     placeholder="./tests/fixtures/tiny_pretrain.jsonl")
            n_trials    = gr.Slider(2, 200, value=20, step=1,  label=t("autotune.n_trials"))
            probe_steps = gr.Slider(10, 1000, value=80, step=10, label=t("autotune.probe_steps"))
            register_translatable(data_path,   "autotune.data_path")
            register_translatable(n_trials,    "autotune.n_trials")
            register_translatable(probe_steps, "autotune.probe_steps")

        gr.Markdown(t("autotune.tip"))
        with gr.Row():
            t_lb = gr.Checkbox(value=True,  label="λ1 balance")
            t_lt = gr.Checkbox(value=True,  label="λ2 temporal")
            t_ll = gr.Checkbox(value=False, label="λ lookahead")
            t_la = gr.Checkbox(value=False, label="lookahead_steps")
            t_lr = gr.Checkbox(value=True,  label="learning_rate")
        with gr.Row():
            t_hs = gr.Checkbox(value=False, label="hidden_size")
            t_ne = gr.Checkbox(value=False, label="num_experts")
            t_ns = gr.Checkbox(value=False, label="num_shared_experts")
            t_kv = gr.Checkbox(value=False, label="kv_latent_dim")

        with gr.Row():
            start_btn = gr.Button(t("autotune.start"), variant="primary")
            stop_btn  = gr.Button(t("autotune.stop"),  variant="stop")
            clear_btn = gr.Button(t("autotune.clear_log"))
            apply_btn = gr.Button(t("autotune.apply_best"), variant="secondary")
            register_translatable(start_btn, "autotune.start")
            register_translatable(stop_btn,  "autotune.stop")
            register_translatable(clear_btn, "autotune.clear_log")
            register_translatable(apply_btn, "autotune.apply_best")

        status_box = gr.Textbox(value="idle", label="Status", interactive=False)
        best_box   = gr.JSON(label=t("autotune.best"))
        log_box    = gr.Textbox(label=t("autotune.log"), lines=20,
                                interactive=False, autoscroll=True,
                                value=_read_log())
        register_translatable(log_box,  "autotune.log")
        register_translatable(best_box, "autotune.best")

        gr.Markdown(f"Log persisted at `{_LOG_FILE}` (survives refresh).")

        def start_tune(dpath, n_tr, p_steps,
                       tlb, tlt, tll, tla, tlr,
                       ths, tne, tns, tkv,
                       cfg):
            if _tuner.is_running():
                return "already running", _read_log(), {}
            if not dpath or not os.path.exists(dpath):
                return f"error: data_path not found: {dpath!r}", _read_log(), {}

            _append_log(f"\n──── Auto-tune started @ {time.strftime('%Y-%m-%d %H:%M:%S')} ────")

            ss = ChronosSearchSpaceConfig(
                tune_lambda_balance=bool(tlb),
                tune_lambda_temporal=bool(tlt),
                tune_lambda_lookahead=bool(tll),
                tune_lookahead_steps=bool(tla),
                tune_lr=bool(tlr),
                tune_hidden_size=bool(ths),
                tune_num_experts=bool(tne),
                tune_num_shared_experts=bool(tns),
                tune_kv_latent_dim=bool(tkv),
            )
            save_dir = (cfg or {}).get("save_dir", "./out")
            hidden_size = (cfg or {}).get("hidden_size", 256)
            _tuner.start(
                model_id=os.path.join(save_dir, f"chronos_{hidden_size}_moe.pth"),
                dataset_path=dpath, train_ratio=0.95,
                prompt_template="alpaca", think_mode="keep",
                search_space=ss, n_trials=int(n_tr),
                probe_steps=int(p_steps),
                output_dir=_CACHE_DIR, seed=42,
            )
            return "running", _read_log(), {}

        def stop_tune():
            _tuner.stop()
            _append_log(f"──── stopped @ {time.strftime('%Y-%m-%d %H:%M:%S')} ────")
            return "stopped"

        def clear_log_file():
            if os.path.exists(_LOG_FILE):
                os.remove(_LOG_FILE)
            return ""

        def poll_tune():
            events = _tuner.poll()
            for ev in events:
                if ev.get("type") == "log":
                    _append_log(ev.get("line", ""))
                elif ev.get("type") == "trial_done":
                    _append_trial(ev)
            best = {}
            if _tuner.result and _tuner.result.best_params:
                best = _tuner.result.best_params
            return _tuner.status, _read_log(), best

        def apply_best(best):
            """Write best_params back into the Config tab's slider widgets.
            Returns a list of gr.update for every config_input — untouched
            inputs get gr.update() (no-op). None values are dropped so
            Gradio's numeric widgets don't reject them."""
            updates = [gr.update() for _ in config_inputs]
            if not best:
                return updates
            for k, v in best.items():
                if v is None:
                    continue
                idx = PARAM_TO_CONFIG_IDX.get(k)
                if idx is not None:
                    updates[idx] = gr.update(value=v)
            return updates

        start_btn.click(
            fn=start_tune,
            inputs=[data_path, n_trials, probe_steps,
                    t_lb, t_lt, t_ll, t_la, t_lr,
                    t_hs, t_ne, t_ns, t_kv,
                    config_state],
            outputs=[status_box, log_box, best_box],
        )
        stop_btn.click(fn=stop_tune, outputs=[status_box])
        clear_btn.click(fn=clear_log_file, outputs=[log_box])
        apply_btn.click(fn=apply_best, inputs=[best_box], outputs=config_inputs)

        timer = gr.Timer(value=2.0)
        timer.tick(fn=poll_tune, outputs=[status_box, log_box, best_box])

    return tab
