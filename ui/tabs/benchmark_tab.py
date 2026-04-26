"""
ui/tabs/benchmark_tab.py — MiniMind vs Chronos comparison view.

Renders the comparison_results.json as:

  - a formatted Markdown table (parameters, RAM, tokens/sec)
  - per-metric bar charts (gr.BarPlot) for visual comparison
  - the raw JSON (collapsed; for power users)
  - a streaming log of the comparison subprocess
"""
import json
import os

from ui.gradio_compat import gr
import pandas as pd

import chronos.deps  # noqa: F401
from ui.i18n import t, register_translatable
from chronos.trainer.device_utils import configure_cpu_threads, cpu_thread_snapshot

RESULTS_FILE = os.path.join(os.path.dirname(__file__), '../../benchmark_results.json')


# Metrics we know how to chart. Order matters — drives the row order of
# the bar plots. Each entry: (display_name, json_key, unit_label, lower_is_better).
CHART_METRICS = [
    ("Parameters",         "params_m",      "M params",  False),
    ("Model RAM (resident)", "ram_model_gb", "GB",        True),
    ("Train RAM",          "ram_train_gb",  "GB",        True),
    ("Inference RAM",      "ram_infer_gb",  "GB",        True),
    ("Decode tokens/sec",  "tokens_per_sec","tokens/s",  False),
]


def _format_table(data: dict) -> str:
    """Render comparison results as a Markdown table."""
    if not data:
        return "_No benchmark data yet. Click **Run** to start._"

    mm = data.get("minimind", {}) or {}
    ch = data.get("chronos", {}) or {}

    rows = ["| Metric | MiniMind | Chronos | Δ (Chronos − MiniMind) | Better |",
            "|---|---|---|---|---|"]
    for label, key, unit, lower_is_better in CHART_METRICS:
        mv = mm.get(key)
        cv = ch.get(key)
        if mv is None and cv is None:
            continue
        mvs = "n/a" if mv is None else f"{mv:.4g} {unit}"
        cvs = "n/a" if cv is None else f"{cv:.4g} {unit}"
        if mv is None or cv is None:
            delta = "n/a"; better = ""
        else:
            d = cv - mv
            sign = "−" if d < 0 else ("+" if d > 0 else "±")
            delta = f"{sign}{abs(d):.4g} {unit}"
            if lower_is_better:
                better = "**Chronos**" if cv < mv else ("MiniMind" if cv > mv else "tie")
            else:
                better = "**Chronos**" if cv > mv else ("MiniMind" if cv < mv else "tie")
        rows.append(f"| {label} | {mvs} | {cvs} | {delta} | {better} |")

    extra = []
    if "kv_cache_type" in ch:
        extra.append(f"- **Chronos KV cache**: `{ch['kv_cache_type']}`")
    cs = ch.get("cache_stats") or {}
    if cs:
        extra.append("- **Cache stats**: " + ", ".join(
            f"`{k}={v}`" for k, v in cs.items()
            if k in ("vram_experts", "ram_experts", "num_clusters",
                     "storage_format", "hit_rate", "expert_size_kb",
                     "pinned_ram_used_gb")
        ))
    if extra:
        rows.append("")
        rows.extend(extra)

    return "\n".join(rows)


def _to_chart_df(data: dict) -> pd.DataFrame:
    """Convert benchmark JSON into a long-format DataFrame for gr.BarPlot."""
    rows = []
    for model_key, label_key in (("minimind", "MiniMind"), ("chronos", "Chronos")):
        d = data.get(model_key, {}) or {}
        for label, key, unit, _lower in CHART_METRICS:
            v = d.get(key)
            if v is None:
                continue
            rows.append({
                "metric": label,
                "model":  d.get("name", label_key),
                "value":  float(v),
                "unit":   unit,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["metric", "model", "value", "unit"]
    )


def _empty_df():
    return pd.DataFrame(columns=["metric", "model", "value", "unit"])


def _load_existing():
    p = os.path.abspath(RESULTS_FILE)
    if os.path.exists(p):
        try:
            with open(p) as f:
                d = json.load(f)
            return d, _format_table(d), _to_chart_df(d), "Loaded previous results."
        except Exception as e:
            return {}, f"_Could not parse {p}: {e}_", _empty_df(), str(e)
    return {}, _format_table({}), _empty_df(), "No results yet. Click Run."


def build_benchmark_tab():
    with gr.Tab(t("tab.benchmark")) as tab:
        register_translatable(tab, "tab.benchmark")

        run_btn = gr.Button(t("bench.run"), variant="primary")
        register_translatable(run_btn, "bench.run")

        # Formatted summary
        summary_md = gr.Markdown(value=_format_table({}))

        # Bar chart — one BarPlot per model so units stack cleanly per metric.
        chart = gr.BarPlot(
            value=_empty_df(),
            x="metric", y="value", color="model",
            title="MiniMind vs Chronos — comparison metrics",
            tooltip=["model", "metric", "value", "unit"],
            height=320,
        )

        with gr.Accordion("Raw JSON", open=False):
            results_box = gr.JSON(label=t("bench.results"))
            register_translatable(results_box, "bench.results")

        log_box = gr.Textbox(
            label=t("bench.log"), lines=20, interactive=False, autoscroll=True
        )
        register_translatable(log_box, "bench.log")

        def run_benchmark():
            import subprocess, sys
            import os as _os
            script = os.path.join(os.path.dirname(__file__), '../../benchmark_compare.py')
            threads = configure_cpu_threads("auto", budget_percent=100)
            env = dict(_os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = str(threads)
            log_lines = []
            yield (
                {}, _format_table({}), _empty_df(),
                f"Running benchmark... CPU threads={threads} {cpu_thread_snapshot()}\n",
            )
            try:
                proc = subprocess.Popen(
                    [sys.executable, script],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=env,
                )
                for line in proc.stdout:
                    log_lines.append(line.rstrip())
                    yield (
                        {}, _format_table({}), _empty_df(),
                        "\n".join(log_lines[-50:]),
                    )
                proc.wait()
                p = os.path.abspath(RESULTS_FILE)
                if os.path.exists(p):
                    with open(p) as f:
                        results = json.load(f)
                    yield (
                        results, _format_table(results), _to_chart_df(results),
                        "\n".join(log_lines),
                    )
                else:
                    yield (
                        {}, "_Run finished but no results file was produced._",
                        _empty_df(), "\n".join(log_lines) + "\n[No results file found]",
                    )
            except Exception as e:
                yield (
                    {}, f"_error: {e}_", _empty_df(), f"Error: {e}",
                )

        run_btn.click(
            fn=run_benchmark,
            outputs=[results_box, summary_md, chart, log_box],
        )

        # Render existing results on tab open
        d, md, df, log = _load_existing()
        results_box.value = d
        summary_md.value = md
        chart.value = df
        log_box.value = log

    return tab
