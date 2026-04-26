"""ui/tabs/pipeline_tab.py — Six-stage training pipeline view.

Each stage shells out to its top-level entry script (train_chronos*.py).
Each stage has its OWN data_path so different stages can use different
fixtures (pretrain corpus vs SFT conversations vs DPO pairs vs GRPO prompts
vs distill-from-SFT). Status updates stream back into a status board.
"""
import os
import subprocess
import sys

from ui.gradio_compat import gr

from chronos.backend import training_available, resolve_training_device
from ui.i18n import t, register_translatable
from ui.tabs.train_tab import _distill_teacher_placeholder


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fix(p: str) -> str:
    return os.path.join(_repo_root(), p)


# (display name, script, default --from_weight, default data fixture, takes_teacher_path)
STAGES = [
    ("Pretrain", "train_chronos.py",         "",       "tests/fixtures/tiny_pretrain.jsonl", False),
    ("SFT",      "train_chronos_sft.py",     "chronos","tests/fixtures/tiny_sft.jsonl",      False),
    ("DPO",      "train_chronos_dpo.py",     "sft",    "tests/fixtures/tiny_dpo.jsonl",      False),
    ("ORPO",     "train_chronos_orpo.py",    "sft",    "tests/fixtures/tiny_dpo.jsonl",      False),
    ("GRPO",     "train_chronos_grpo.py",    "orpo",   "tests/fixtures/tiny_grpo.jsonl",     False),
    ("Distill",  "train_chronos_distill.py", "grpo",   "tests/fixtures/tiny_sft.jsonl",      True),
]

PIPELINE_TRAIN_BACKEND_CHOICES = ["auto"] + [name for name in ("cuda", "xpu", "mlx", "mps", "cpu") if name in set(training_available())]


def build_pipeline_tab():
    with gr.Tab(t("tab.pipeline")) as tab:
        register_translatable(tab, "tab.pipeline")
        gr.Markdown(f"### {t('pipeline.title')}")

        with gr.Row():
            save_dir = gr.Textbox(value="out", label=t("pipeline.save_dir"), scale=2)
            steps    = gr.Number(value=30, precision=0, label=t("pipeline.steps"), scale=1)
            train_backend = gr.Dropdown(
                choices=PIPELINE_TRAIN_BACKEND_CHOICES,
                value="auto",
                label=t("train.backend"),
                scale=1,
            )
            register_translatable(save_dir, "pipeline.save_dir")
            register_translatable(steps,    "pipeline.steps")
            register_translatable(train_backend, "train.backend")

        # Per-stage rows: name + data_path + (optional) teacher_path + Run button
        per_stage_rows = []  # list of dicts {data, teacher_or_None, status, run_btn}
        for i, (name, script, from_w, default_data, takes_teacher) in enumerate(STAGES):
            with gr.Row():
                gr.Markdown(f"**{i+1}. {name}**", elem_classes=[])
                # Default to a RELATIVE path so the textbox is portable across
                # checkouts. Subprocess runs with cwd=repo_root, so the path
                # resolves against the project directory.
                data_box = gr.Textbox(value=default_data, label=f"{name} {t('pipeline.data_path')}", scale=3)
                teacher_box = (
                    gr.Textbox(value="", label=f"{name} {t('pipeline.teacher_path')}",
                               placeholder=_distill_teacher_placeholder(), scale=2)
                    if takes_teacher else None
                )
                status_box = gr.Textbox(value="pending", label=t("pipeline.status"), interactive=False, scale=1)
                run_btn = gr.Button(f"▶ {name}", scale=1)
            per_stage_rows.append({
                "name": name, "script": script, "from_weight": from_w,
                "data": data_box, "teacher": teacher_box,
                "status": status_box, "run_btn": run_btn, "takes_teacher": takes_teacher,
            })

        log_box = gr.Textbox(label="log (stage runs append)", lines=20, interactive=False, autoscroll=True)
        clear_log = gr.Button(t("pipeline.clear_log"))
        register_translatable(clear_log, "pipeline.clear_log")
        clear_log.click(fn=lambda: "", outputs=[log_box])

        def make_runner(idx):
            row = per_stage_rows[idx]
            name = row["name"]; script = row["script"]; from_w = row["from_weight"]
            takes_teacher = row["takes_teacher"]

            def run(_save_dir, _steps, _backend, _data, _teacher, current_log):
                selected_backend, resolved_device = resolve_training_device(_backend)
                cmd = [sys.executable, script,    # relative — resolved against cwd
                       "--data_path", _data,
                       "--save_dir", _save_dir,
                       "--steps", str(int(_steps)),
                       "--device", selected_backend if selected_backend == "mlx" else resolved_device]
                if from_w:
                    cmd += ["--from_weight", from_w]
                if takes_teacher:
                    if not _teacher:
                        # default to sft checkpoint at the configured save_dir
                        _teacher = os.path.join(_save_dir, "sft_192_moe.pth")
                    cmd += ["--teacher_path", _teacher]

                lines = (current_log.split("\n") if current_log else [])
                lines.append(f"\n──── [{name}] $ {' '.join(cmd)} ────")
                yield "running", "\n".join(lines[-500:])
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                                            cwd=_repo_root())
                    for line in proc.stdout:
                        lines.append(f"[{name}] {line.rstrip()}")
                        yield "running", "\n".join(lines[-500:])
                    proc.wait()
                    final = "done" if proc.returncode == 0 else f"failed ({proc.returncode})"
                    lines.append(f"──── [{name}] {final} ────")
                    yield final, "\n".join(lines[-500:])
                except Exception as e:
                    lines.append(f"[{name}] error: {e}")
                    yield "failed", "\n".join(lines[-500:])
            return run

        for i, row in enumerate(per_stage_rows):
            inputs = [save_dir, steps, train_backend, row["data"]]
            inputs.append(row["teacher"] if row["takes_teacher"] else gr.State(""))
            inputs.append(log_box)
            row["run_btn"].click(
                fn=make_runner(i),
                inputs=inputs,
                outputs=[row["status"], log_box],
            )

    return tab
