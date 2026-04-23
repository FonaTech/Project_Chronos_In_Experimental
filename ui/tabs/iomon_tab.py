"""ui/tabs/iomon_tab.py — Live IO/cache metrics monitor."""
import time

from ui.gradio_compat import gr

from ui.i18n import t, register_translatable
from chronos.runtime.metrics import bus


def _poll() -> dict:
    snap = bus.snapshot()
    out = {}
    for k, v in snap.items():
        if not v:
            continue
        out[k] = {"latest": v[-1][1], "n_samples": len(v)}
    return out


def build_iomon_tab():
    with gr.Tab(t("tab.iomon")) as tab:
        register_translatable(tab, "tab.iomon")
        gr.Markdown(f"### {t('iomon.title')}")

        metrics_box = gr.JSON(label="metrics snapshot")
        refresh_btn = gr.Button(t("iomon.refresh"))
        register_translatable(refresh_btn, "iomon.refresh")

        refresh_btn.click(fn=_poll, outputs=[metrics_box])

        # Auto-refresh every 2s while tab is open
        timer = gr.Timer(2.0)
        timer.tick(fn=_poll, outputs=[metrics_box])

        # Initial value
        metrics_box.value = _poll()
    return tab
