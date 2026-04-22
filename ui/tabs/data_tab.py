"""
ui/tabs/data_tab.py — Dataset inspector.

Streams a JSONL file (no full-load) and reports:
  - record count
  - char-length distribution (min / mean / median / p95 / max)
  - tokenized-length distribution if a tokenizer is available
  - unique-record ratio (cheap dedup heuristic via line hash)
  - first ~3 sample previews
  - matplotlib histogram of token lengths

The point: before training, the user can verify the corpus is what they
expect — right field name, sane lengths, low duplication. Catches
"wrong tokenizer" / "all records empty" / "huge dup ratio" before they
burn 10 hours of training.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import chronos.deps  # noqa: F401
from ui.i18n import t, register_translatable


def _extract_text(rec: dict) -> str:
    for k in ("text", "content", "story", "passage", "document", "article"):
        if k in rec and rec[k]:
            return str(rec[k])
    if "conversations" in rec and isinstance(rec["conversations"], list):
        return " ".join(str(c.get("content", c.get("value", ""))) for c in rec["conversations"])
    if "messages" in rec and isinstance(rec["messages"], list):
        return " ".join(str(m.get("content", "")) for m in rec["messages"])
    return " ".join(str(v) for v in rec.values()
                    if isinstance(v, (str, int, float)))


def _scan(path: str, max_records: int = 50_000):
    """Stream up to `max_records` lines; return summary dict + length list."""
    if not path or not os.path.exists(path):
        return {"error": f"file not found: {path!r}"}, [], []
    char_lens = []
    samples = []
    seen_hashes = set()
    dup = 0
    n_total = 0
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _extract_text(rec)
            char_lens.append(len(text))
            h = hashlib.md5(line).hexdigest()
            if h in seen_hashes:
                dup += 1
            else:
                seen_hashes.add(h)
            if len(samples) < 3:
                samples.append(text[:200])
    if not char_lens:
        return {"error": "no parseable records"}, [], samples

    summary = {
        "scanned":       n_total,
        "parseable":     len(char_lens),
        "dup_ratio":     round(dup / max(n_total, 1), 4),
        "char_min":      min(char_lens),
        "char_mean":     round(statistics.mean(char_lens), 1),
        "char_median":   int(statistics.median(char_lens)),
        "char_p95":      int(sorted(char_lens)[int(0.95 * len(char_lens))]),
        "char_max":      max(char_lens),
        "file_size_mb":  round(os.path.getsize(path) / (1024 * 1024), 1),
        "note":          (f"sampled first {max_records} records"
                          if n_total >= max_records else "full scan"),
    }
    return summary, char_lens, samples


def _hist(lens):
    if not lens:
        return None
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.hist(lens, bins=40, color="#39d98a", edgecolor="#1a1a2e")
    ax.set_xlabel("chars per record", color="#aaa")
    ax.set_ylabel("frequency", color="#aaa")
    ax.tick_params(colors="#aaa")
    for s in ax.spines.values():
        s.set_edgecolor("#333")
    return fig


def build_data_tab():
    with gr.Tab(t("tab.data")) as tab:
        register_translatable(tab, "tab.data")
        gr.Markdown(f"### {t('data.title')}")

        with gr.Row():
            data_path = gr.Textbox(
                label=t("data.path"), scale=4,
                placeholder="/Users/Fona/Downloads/Hybrid_LLM/Dataset/pretrain_t2t_mini.jsonl",
            )
            scan_btn = gr.Button(t("data.scan"), variant="primary", scale=1)
            register_translatable(data_path, "data.path")
            register_translatable(scan_btn, "data.scan")

        summary_box = gr.JSON(label=t("data.summary"), value={})
        register_translatable(summary_box, "data.summary")

        preview_box = gr.Textbox(
            label=t("data.preview"), lines=6, interactive=False, value="",
        )
        register_translatable(preview_box, "data.preview")

        hist_plot = gr.Plot(label=t("data.length_hist"))
        register_translatable(hist_plot, "data.length_hist")

        def do_scan(path):
            summary, lens, samples = _scan(path)
            preview = "\n---\n".join(samples) if samples else ""
            fig = _hist(lens)
            return summary, preview, fig

        scan_btn.click(fn=do_scan, inputs=[data_path],
                       outputs=[summary_box, preview_box, hist_plot])

    return tab
