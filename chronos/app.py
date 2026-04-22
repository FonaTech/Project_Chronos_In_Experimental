"""
chronos/app.py — Project Chronos Web UI entry point (package-installable).

Exposed as `chronos-ui` CLI entry point via pyproject.toml.

Usage:
    chronos-ui
    chronos-ui --port 7861 --share
    python -m chronos.app
"""
import sys
import os
import argparse

# When run from the installed package, Project_Chronos root may not be on
# sys.path. We add it so that ui.* imports resolve correctly.
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import gradio as gr

import chronos.deps  # auto-bootstrap minimind
from ui.i18n import (
    LANGUAGE_CHOICES, DEFAULT_LANGUAGE,
    t, set_current_lang, build_language_update, get_registered_components,
)
from ui.tabs.config_tab    import build_config_tab
from ui.tabs.train_tab     import build_train_tab
from ui.tabs.inference_tab import build_inference_tab
from ui.tabs.benchmark_tab import build_benchmark_tab
from ui.tabs.autotune_tab  import build_autotune_tab
from ui.tabs.iomon_tab     import build_iomon_tab
from ui.tabs.pipeline_tab  import build_pipeline_tab
from ui.tabs.data_tab      import build_data_tab


APP_CSS = """
.app-header { padding: 24px 0 16px; border-bottom: 1px solid var(--border-color-primary); margin-bottom: 16px; }
.app-header-title { font-size: 2rem; font-weight: 700; margin: 4px 0; }
.app-header-subtitle { color: var(--body-text-color-subdued); margin: 0; }
.app-meta-line { display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
.app-meta-chip { background: var(--background-fill-secondary); border-radius: 6px; padding: 3px 10px; font-size: 12px; }
.toolbar-strip { margin-bottom: 8px; }
"""


def _hero_html() -> str:
    return (
        "<div class='app-header'>"
        "<div style='font-size:12px;color:var(--body-text-color-subdued);letter-spacing:.08em;text-transform:uppercase;'>On-Device MoE Inference Workbench</div>"
        "<h1 class='app-header-title'>Project Chronos</h1>"
        "<p class='app-header-subtitle'>"
        "Lookahead Routing · MLA + Sliding Window Attention · Async SSD Prefetch · "
        "Pretrain / SFT / RL / ORPO · Optuna Auto-Tune"
        "</p>"
        "<div class='app-meta-line'>"
        "<span class='app-meta-chip'>Apache-2.0</span>"
        "<span class='app-meta-chip'>Python 3.10+</span>"
        "<span class='app-meta-chip'>PyTorch 2.1+</span>"
        "<span class='app-meta-chip'><a href='https://github.com/jingyaogong/minimind' target='_blank'>MoE kernel: minimind</a></span>"
        "</div>"
        "</div>"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Project Chronos",
        css=APP_CSS,
        fill_width=True,
    ) as app:
        gr.HTML(_hero_html())

        with gr.Row(elem_classes="toolbar-strip"):
            language = gr.Dropdown(
                choices=LANGUAGE_CHOICES,
                value=DEFAULT_LANGUAGE,
                label="Language / 语言 / 言語",
                scale=1,
            )

        config_state, config_inputs, cfg_save_dir = build_config_tab()
        build_data_tab()
        build_train_tab(config_state, cfg_save_dir)
        build_pipeline_tab()
        build_inference_tab(config_state)
        build_benchmark_tab()
        build_autotune_tab(config_state, config_inputs)
        build_iomon_tab()

        def on_language_change(lang):
            set_current_lang(lang)
            return build_language_update(lang)

        language.change(
            fn=on_language_change,
            inputs=[language],
            outputs=get_registered_components(),
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Project Chronos Web UI")
    parser.add_argument("--port",  type=int,  default=7860)
    parser.add_argument("--host",  type=str,  default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
