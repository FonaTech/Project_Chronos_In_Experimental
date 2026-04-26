"""ui/tabs/export_tab.py — Chronos deployment export UI."""
from __future__ import annotations

import json

from ui.gradio_compat import gr

from chronos.export import EXPORT_FORMATS, export_checkpoint, format_export_report
from ui.i18n import t, register_translatable


def _default_formats() -> list[str]:
    return ["fp16-safetensors", "q8_0-safetensors", "fp16-gguf", "q8_0-gguf"]


def _result_payload(result) -> dict:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    payload = {}
    raw = metadata.get("chronos_export") or metadata.get("chronos.export.metadata_json")
    if raw:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            payload = {}
    elif metadata.get("cluster_build") or metadata.get("expert_cache"):
        payload = metadata
    return payload


def _export(model_path: str, output_dir: str, formats, config_path: str,
            expert_cache_dir: str, copy_cache: bool, auto_cluster: bool,
            calibration_data_path: str, cluster_max_batches: int,
            cluster_batch_size: int, cluster_max_seq_len: int,
            cluster_device: str):
    if not model_path:
        return "Model path is required.", {}
    if not output_dir:
        output_dir = "./exports/chronos"
    try:
        selected = formats or _default_formats()
        results = export_checkpoint(
            model_path,
            output_dir,
            formats=selected,
            config_path=config_path or None,
            expert_cache_dir=expert_cache_dir or None,
            copy_expert_cache=bool(copy_cache),
            auto_cluster=bool(auto_cluster),
            calibration_data_path=calibration_data_path or None,
            cluster_max_batches=int(cluster_max_batches or 50),
            cluster_batch_size=int(cluster_batch_size or 4),
            cluster_max_seq_len=int(cluster_max_seq_len or 256),
            cluster_device=cluster_device or "cpu",
        )
        summary = {
            r.format: {
                "path": r.path,
                "size_mb": round(r.bytes / (1024 * 1024), 3),
                "tensors": r.tensors,
                "expert_cache": _result_payload(r).get("expert_cache"),
                "cluster_build": _result_payload(r).get("cluster_build"),
            }
            for r in results
        }
        compatibility = (
            "\n\nCompatibility note: safetensors and GGUF outputs are standard containers "
            "with Chronos architecture metadata. Stock Ollama/llama.cpp builds need a "
            "Chronos architecture adapter to execute hybrid attention, lookahead MoE "
            "prediction, and lazy expert loading correctly."
        )
        return format_export_report(results) + compatibility, summary
    except Exception as exc:
        import traceback

        return f"Export failed: {exc}\n{traceback.format_exc()}", {}


def build_export_tab():
    with gr.Tab(t("tab.export")) as tab:
        register_translatable(tab, "tab.export")
        gr.Markdown(f"### {t('export.title')}")

        with gr.Row():
            model_path = gr.Textbox(
                value="./out/sft_384_moe.pth",
                label=t("export.model_path"),
                scale=3,
            )
            output_dir = gr.Textbox(
                value="./exports/chronos",
                label=t("export.output_dir"),
                scale=2,
            )
            register_translatable(model_path, "export.model_path")
            register_translatable(output_dir, "export.output_dir")

        formats = gr.CheckboxGroup(
            choices=list(EXPORT_FORMATS),
            value=_default_formats(),
            label=t("export.formats"),
        )
        register_translatable(formats, "export.formats")

        with gr.Row():
            config_path = gr.Textbox(
                value="",
                label=t("export.config_path"),
                placeholder="optional: ./chronos_config.json",
                scale=2,
            )
            expert_cache_dir = gr.Textbox(
                value="",
                label=t("export.expert_cache_dir"),
                placeholder="optional: ./expert_cache/ckpt_xxx",
                scale=2,
            )
            copy_cache = gr.Checkbox(value=True, label=t("export.copy_cache"), scale=1)
            register_translatable(config_path, "export.config_path")
            register_translatable(expert_cache_dir, "export.expert_cache_dir")
            register_translatable(copy_cache, "export.copy_cache")

        with gr.Row():
            auto_cluster = gr.Checkbox(
                value=True,
                label=t("export.auto_cluster"),
                scale=1,
            )
            calibration_data_path = gr.Textbox(
                value="",
                label=t("export.calibration_data_path"),
                placeholder="optional: ./dataset/calib.jsonl",
                scale=3,
            )
            cluster_device = gr.Dropdown(
                choices=["cpu", "mps", "cuda"],
                value="cpu",
                label=t("export.cluster_device"),
                scale=1,
            )
            register_translatable(auto_cluster, "export.auto_cluster")
            register_translatable(calibration_data_path, "export.calibration_data_path")
            register_translatable(cluster_device, "export.cluster_device")

        with gr.Row():
            cluster_max_batches = gr.Number(
                value=50,
                precision=0,
                label=t("export.cluster_max_batches"),
                scale=1,
            )
            cluster_batch_size = gr.Number(
                value=4,
                precision=0,
                label=t("export.cluster_batch_size"),
                scale=1,
            )
            cluster_max_seq_len = gr.Number(
                value=256,
                precision=0,
                label=t("export.cluster_max_seq_len"),
                scale=1,
            )
            register_translatable(cluster_max_batches, "export.cluster_max_batches")
            register_translatable(cluster_batch_size, "export.cluster_batch_size")
            register_translatable(cluster_max_seq_len, "export.cluster_max_seq_len")

        export_btn = gr.Button(t("export.run"), variant="primary")
        register_translatable(export_btn, "export.run")

        log_box = gr.Textbox(label=t("export.log"), lines=10, interactive=False)
        summary_box = gr.JSON(label=t("export.summary"), value={})
        register_translatable(log_box, "export.log")
        register_translatable(summary_box, "export.summary")

        export_btn.click(
            fn=_export,
            inputs=[
                model_path,
                output_dir,
                formats,
                config_path,
                expert_cache_dir,
                copy_cache,
                auto_cluster,
                calibration_data_path,
                cluster_max_batches,
                cluster_batch_size,
                cluster_max_seq_len,
                cluster_device,
            ],
            outputs=[log_box, summary_box],
        )

    return tab
