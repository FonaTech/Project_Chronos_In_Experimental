"""
ui/tabs/train_tab.py — Full training loop: Pretrain / SFT / DPO / ORPO / GRPO / Distill
"""
import os
import time
import threading
import queue
from contextlib import contextmanager
from types import SimpleNamespace

import gradio as gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable, get_current_lang
from chronos.backend import training_available, resolve_training_device


STAGE_UI_ORDER = ["pretrain", "sft", "dpo", "orpo", "grpo", "distill"]
STAGE_LABELS = {
    "pretrain": "Stage 1 · Pretrain",
    "sft": "Stage 2 · SFT",
    "dpo": "Stage 3 · DPO",
    "orpo": "Stage 4 · ORPO",
    "grpo": "Stage 5 · GRPO",
    "distill": "Stage 6 · Distill",
}
STAGE_CHECKPOINT_PREFIX = {
    "pretrain": "chronos",
    "sft": "sft",
    "dpo": "dpo",
    "orpo": "orpo",
    "grpo": "grpo",
    "distill": "distill",
}
STAGE_DEFAULT_INIT = {
    "sft": "chronos",
    "dpo": "sft",
    "orpo": "sft",
    "grpo": "orpo",
    "distill": "grpo",
}
STAGE_DEFAULT_DATA = {
    "pretrain": "./tests/fixtures/tiny_pretrain.jsonl",
    "sft": "./tests/fixtures/tiny_sft.jsonl",
    "dpo": "./tests/fixtures/tiny_dpo.jsonl",
    "orpo": "./tests/fixtures/tiny_dpo.jsonl",
    "grpo": "./tests/fixtures/tiny_grpo.jsonl",
    "distill": "./tests/fixtures/tiny_sft.jsonl",
}
STAGE_HELP_TEXT = {
    "pretrain": {
        "zh-Hans": "从通用语料继续预训练。`init_weight` 可留空；若存在同名 checkpoint，将按当前拓扑尝试恢复。",
        "zh-Hant": "從通用語料繼續預訓練。`init_weight` 可留空；若存在同名 checkpoint，將按目前拓撲嘗試恢復。",
        "en": "Continue base pretraining on LM-style data. `init_weight` may be left blank; if a matching checkpoint exists, Train will resume from it.",
        "ja": "汎用コーパスで事前学習を続けます。`init_weight` は空で構いません。同名 checkpoint があり拓撲が一致すれば再開します。",
    },
    "sft": {
        "zh-Hans": "使用 `conversations` 数据做监督微调。`init_weight` 留空时，会从当前 `save_dir` 自动接续 `chronos_<H>_moe.pth`；只有手动填路径时才覆盖。",
        "zh-Hant": "使用 `conversations` 資料做監督微調。`init_weight` 留空時，會從目前 `save_dir` 自動接續 `chronos_<H>_moe.pth`；只有手動填路徑時才覆蓋。",
        "en": "Run supervised fine-tuning on `conversations` data. If `init_weight` is blank, Train auto-resolves `chronos_<H>_moe.pth` from the current `save_dir`; only a pasted path overrides it.",
        "ja": "`conversations` データで SFT を行います。`init_weight` を空欄にすると、現在の `save_dir` から `chronos_<H>_moe.pth` を自動解決します。パスを入力した場合のみ上書きします。",
    },
    "dpo": {
        "zh-Hans": "使用 `chosen / rejected` 偏好对做 DPO。`init_weight` 留空时，会自动加载当前 `save_dir` 下的 `sft_<H>_moe.pth`；手动填写路径可覆盖。",
        "zh-Hant": "使用 `chosen / rejected` 偏好對做 DPO。`init_weight` 留空時，會自動載入目前 `save_dir` 下的 `sft_<H>_moe.pth`；手動填寫路徑可覆蓋。",
        "en": "Run DPO on `chosen / rejected` preference pairs. Leaving `init_weight` blank auto-loads `sft_<H>_moe.pth` from the current `save_dir`; paste a path only when you want to override it.",
        "ja": "`chosen / rejected` の選好ペアで DPO を実行します。`init_weight` を空欄にすると、現在の `save_dir` から `sft_<H>_moe.pth` を自動読み込みします。上書きしたい場合だけパスを入力してください。",
    },
    "orpo": {
        "zh-Hans": "使用同样的偏好对做 ORPO。`init_weight` 留空时，会自动加载当前 `save_dir` 下的 `sft_<H>_moe.pth`；手动填写路径可覆盖。",
        "zh-Hant": "使用相同的偏好對做 ORPO。`init_weight` 留空時，會自動載入目前 `save_dir` 下的 `sft_<H>_moe.pth`；手動填寫路徑可覆蓋。",
        "en": "Run ORPO on the same preference-pair format. Leaving `init_weight` blank auto-loads `sft_<H>_moe.pth` from the current `save_dir`; paste a path only when you want to override it.",
        "ja": "同じ選好ペア形式で ORPO を実行します。`init_weight` を空欄にすると、現在の `save_dir` から `sft_<H>_moe.pth` を自動読み込みします。上書きしたい場合だけパスを入力してください。",
    },
    "grpo": {
        "zh-Hans": "使用 prompt 数据做 GRPO rollout。`init_weight` 留空时，会自动加载当前 `save_dir` 下的 `orpo_<H>_moe.pth`；`reward` 可配置。",
        "zh-Hant": "使用 prompt 資料做 GRPO rollout。`init_weight` 留空時，會自動載入目前 `save_dir` 下的 `orpo_<H>_moe.pth`；`reward` 可設定。",
        "en": "Run GRPO rollouts on prompt-style data. Leaving `init_weight` blank auto-loads `orpo_<H>_moe.pth` from the current `save_dir`; `reward` remains configurable.",
        "ja": "prompt 形式のデータで GRPO rollout を行います。`init_weight` を空欄にすると、現在の `save_dir` から `orpo_<H>_moe.pth` を自動読み込みします。`reward` は設定できます。",
    },
    "distill": {
        "zh-Hans": "蒸馏阶段中，学生 `init_weight` 留空时会自动接续当前 `save_dir` 下的 `grpo_<H>_moe.pth`；`teacher_path` 留空时会自动解析 `sft_<H>_moe.pth`。只有手动填路径时才覆盖。",
        "zh-Hant": "蒸餾階段中，學生 `init_weight` 留空時會自動接續目前 `save_dir` 下的 `grpo_<H>_moe.pth`；`teacher_path` 留空時會自動解析 `sft_<H>_moe.pth`。只有手動填路徑時才覆蓋。",
        "en": "For distillation, a blank student `init_weight` auto-resolves `grpo_<H>_moe.pth` from the current `save_dir`, and a blank `teacher_path` auto-resolves `sft_<H>_moe.pth`. Paste paths only when you want to override either one.",
        "ja": "蒸留では、学生側の `init_weight` を空欄にすると現在の `save_dir` から `grpo_<H>_moe.pth` を自動解決し、`teacher_path` を空欄にすると `sft_<H>_moe.pth` を自動解決します。上書きしたい場合だけパスを入力してください。",
    },
}
STAGE_METRIC_LABELS = {
    "pretrain": ("Total Loss", "CE Loss", "Aux Loss"),
    "sft": ("SFT Loss", "CE Loss", "Lookahead"),
    "dpo": ("DPO Loss", "Preference", "Anchor"),
    "orpo": ("ORPO Loss", "Chosen NLL", "OR Term"),
    "grpo": ("GRPO Loss", "Policy", "KL"),
    "distill": ("Distill Loss", "KD Loss", "CE Loss"),
}
STAGE_CHART_META = {
    "pretrain": {
        "title": "Pretrain Metrics",
        "primary": "CE",
        "secondary": "Aux",
        "tertiary": "Temporal",
        "val": "Validation",
        "speed_title": "Training Speed (steps/s)",
        "speed_label": "steps/s",
    },
    "sft": {
        "title": "SFT Metrics",
        "primary": "CE",
        "secondary": "Lookahead",
        "tertiary": "Anchor",
        "val": "Validation",
        "speed_title": "SFT Speed (steps/s)",
        "speed_label": "steps/s",
    },
    "dpo": {
        "title": "DPO Metrics",
        "primary": "Preference",
        "secondary": "Anchor",
        "tertiary": "Lookahead",
        "val": "Validation",
        "speed_title": "Preference Speed (steps/s)",
        "speed_label": "steps/s",
    },
    "orpo": {
        "title": "ORPO Metrics",
        "primary": "Chosen NLL",
        "secondary": "OR Term",
        "tertiary": "Anchor",
        "val": "Validation",
        "speed_title": "ORPO Speed (steps/s)",
        "speed_label": "steps/s",
    },
    "grpo": {
        "title": "GRPO Metrics",
        "primary": "Policy",
        "secondary": "KL",
        "tertiary": "Reward",
        "val": "Validation",
        "speed_title": "Rollout Speed + Reward",
        "speed_label": "rollouts/s",
        "speed_overlay": "Mean Reward",
    },
    "distill": {
        "title": "Distill Metrics",
        "primary": "KD",
        "secondary": "CE",
        "tertiary": "Anchor",
        "val": "Validation",
        "speed_title": "Distill Speed (steps/s)",
        "speed_label": "steps/s",
    },
}


DEFAULT_STAGE_SAMPLE_PROMPTS = {
    "pretrain": {
        "zh-Hans": "请用简洁中文介绍 Project Chronos 如何把 MoE 的专家加载提前到 Prefill 阶段。",
        "zh-Hant": "請用精簡中文介紹 Project Chronos 如何把 MoE 的專家載入提前到 Prefill 階段。",
        "en": "Explain how Project Chronos moves MoE expert loading into the prefill stage.",
        "ja": "Project Chronos が MoE の専門家ロードを Prefill 段階へ前倒しする仕組みを説明してください。",
    },
    "sft": {
        "zh-Hans": "用户：为什么 Chronos 的双层路由比传统被动 offload 更适合消费级硬件？\n助手：",
        "zh-Hant": "使用者：為什麼 Chronos 的雙層路由比傳統被動 offload 更適合消費級硬體？\n助手：",
        "en": "User: Why is Chronos's dual-layer routing better suited to consumer hardware than reactive expert offload?\nAssistant:",
        "ja": "ユーザー: なぜ Chronos の二層ルーティングは、従来の受動的な expert offload よりも民生向けハードウェアに向いているのですか？\nアシスタント:",
    },
    "dpo": {
        "zh-Hans": "用户：请比较 Chronos 和传统 reactive offload 在推理时的关键差异。\n助手：",
        "zh-Hant": "使用者：請比較 Chronos 與傳統 reactive offload 在推理時的關鍵差異。\n助手：",
        "en": "User: Compare Chronos with conventional reactive offload during inference.\nAssistant:",
        "ja": "ユーザー: 推論時における Chronos と従来の reactive offload の主要な違いを比較してください。\nアシスタント:",
    },
    "orpo": {
        "zh-Hans": "用户：请写一段面向开发者的 Project Chronos 介绍。\n助手：",
        "zh-Hant": "使用者：請寫一段面向開發者的 Project Chronos 介紹。\n助手：",
        "en": "User: Write a developer-facing introduction to Project Chronos.\nAssistant:",
        "ja": "ユーザー: 開発者向けの Project Chronos 紹介文を書いてください。\nアシスタント:",
    },
    "grpo": {
        "zh-Hans": "请比较 Chronos 与传统 MoE offload 在吞吐、首 token 延迟和缓存未命中行为上的权衡。",
        "zh-Hant": "請比較 Chronos 與傳統 MoE offload 在吞吐、首 token 延遲和快取未命中行為上的權衡。",
        "en": "Compare Chronos with conventional MoE offload in terms of throughput, first-token latency, and cache-miss behavior.",
        "ja": "Chronos と従来の MoE offload を、スループット、初回トークン遅延、キャッシュミス時の挙動で比較してください。",
    },
    "distill": {
        "zh-Hans": "请用更现代、更有产品感的英文改写 Project Chronos 的首页介绍，保留核心技术点。",
        "zh-Hant": "請用更現代、更有產品感的英文改寫 Project Chronos 的首頁介紹，保留核心技術點。",
        "en": "Rewrite the Project Chronos landing-page intro in a more modern product voice while preserving the core technical claims.",
        "ja": "Project Chronos のトップ紹介文を、技術的な要点を保ったまま、よりモダンでプロダクト寄りの英語に書き換えてください。",
    },
}


def _stage_sample_prompt(mode: str, lang: str | None = None) -> str:
    lang = lang or get_current_lang()
    prompts = DEFAULT_STAGE_SAMPLE_PROMPTS.get(mode, DEFAULT_STAGE_SAMPLE_PROMPTS["pretrain"])
    return prompts.get(lang, prompts["en"])


def _default_sample_prompts() -> set[str]:
    vals: set[str] = set()
    for prompts in DEFAULT_STAGE_SAMPLE_PROMPTS.values():
        vals.update(v.strip() for v in prompts.values() if v and v.strip())
    return vals


DEFAULT_SAMPLE_PROMPT_VALUES = _default_sample_prompts()
DEFAULT_STAGE_DATA_VALUES = {v.strip() for v in STAGE_DEFAULT_DATA.values() if v.strip()}
DEFAULT_STAGE_INIT_VALUES = {
    f"./out/{prefix}_768_moe.pth"
    for prefix in STAGE_DEFAULT_INIT.values()
}

PRETRAIN_INIT_PLACEHOLDER = {
    "zh-Hans": "留空则按当前 save_dir 自动续训同阶段 checkpoint",
    "zh-Hant": "留空則按目前 save_dir 自動續訓同階段 checkpoint",
    "en": "Leave blank to auto-resume the current-stage checkpoint from save_dir",
    "ja": "空欄なら save_dir から同一ステージ checkpoint を自動再開",
}
UPSTREAM_INIT_PLACEHOLDER = {
    "zh-Hans": "留空则按当前 save_dir 自动载入 {prefix}_<H>_moe.pth，手动填写才覆盖",
    "zh-Hant": "留空則按目前 save_dir 自動載入 {prefix}_<H>_moe.pth，手動填寫才覆蓋",
    "en": "Leave blank to auto-load {prefix}_<H>_moe.pth from save_dir; paste a path to override",
    "ja": "空欄なら save_dir から {prefix}_<H>_moe.pth を自動読み込み。上書き時だけパスを指定",
}
DISTILL_TEACHER_PLACEHOLDER = {
    "zh-Hans": "留空则按当前 save_dir 自动载入 sft_<H>_moe.pth，手动填写才覆盖",
    "zh-Hant": "留空則按目前 save_dir 自動載入 sft_<H>_moe.pth，手動填寫才覆蓋",
    "en": "Leave blank to auto-load sft_<H>_moe.pth from save_dir; paste a path to override",
    "ja": "空欄なら save_dir から sft_<H>_moe.pth を自動読み込み。上書き時だけパスを指定",
}

TRAIN_BACKEND_CHOICES = ["auto", "cuda", "xpu", "mps", "cpu"]

TRAIN_BACKEND_LABELS = {
    "auto": {
        "zh-Hans": "自动 (默认最快)",
        "zh-Hant": "自動 (預設最快)",
        "en": "Auto (fastest default)",
        "ja": "自動 (最速を既定)",
    },
    "cuda": {
        "zh-Hans": "CUDA",
        "zh-Hant": "CUDA",
        "en": "CUDA",
        "ja": "CUDA",
    },
    "mps": {
        "zh-Hans": "MPS",
        "zh-Hant": "MPS",
        "en": "MPS",
        "ja": "MPS",
    },
    "xpu": {
        "zh-Hans": "XPU",
        "zh-Hant": "XPU",
        "en": "XPU",
        "ja": "XPU",
    },
    "cpu": {
        "zh-Hans": "CPU",
        "zh-Hant": "CPU",
        "en": "CPU",
        "ja": "CPU",
    },
}


def _stage_text(stage_map: dict[str, str], mode: str, lang: str | None = None) -> str:
    lang = lang or get_current_lang()
    per_stage = stage_map.get(mode, stage_map["pretrain"])
    return per_stage.get(lang, per_stage["en"])


def _stage_init_placeholder(mode: str, lang: str | None = None) -> str:
    lang = lang or get_current_lang()
    if mode == "pretrain":
        return PRETRAIN_INIT_PLACEHOLDER.get(lang, PRETRAIN_INIT_PLACEHOLDER["en"])
    upstream = STAGE_DEFAULT_INIT.get(mode, "chronos")
    template = UPSTREAM_INIT_PLACEHOLDER.get(lang, UPSTREAM_INIT_PLACEHOLDER["en"])
    return template.format(prefix=upstream)


def _normalize_stage_init_value(current_init_weight: str | None) -> str:
    current_init = (current_init_weight or "").strip()
    if current_init in DEFAULT_STAGE_INIT_VALUES:
        return ""
    return current_init


def _distill_teacher_placeholder(lang: str | None = None) -> str:
    lang = lang or get_current_lang()
    return DISTILL_TEACHER_PLACEHOLDER.get(lang, DISTILL_TEACHER_PLACEHOLDER["en"])


def _available_train_backend_choices() -> list[str]:
    available = set(training_available())
    return ["auto"] + [name for name in TRAIN_BACKEND_CHOICES[1:] if name in available]


def _train_backend_label(name: str, lang: str | None = None) -> str:
    lang = lang or get_current_lang()
    labels = TRAIN_BACKEND_LABELS.get(name, TRAIN_BACKEND_LABELS["auto"])
    return labels.get(lang, labels["en"])


def _train_backend_dropdown_choices(lang: str | None = None) -> list[tuple[str, str]]:
    return [
        (_train_backend_label(name, lang), name)
        for name in _available_train_backend_choices()
    ]


def _default_train_backend_value() -> str:
    return "auto"


def _normalize_dtype_for_trainer(dtype: str | None) -> str:
    dt = (dtype or "fp16").strip().lower()
    if dt in {"bf16", "bfloat16"}:
        return "bfloat16"
    return "float16"


@contextmanager
def _trainer_logger_sink(put_line):
    from trainer import trainer_utils as _tu  # type: ignore

    orig_logger = _tu.Logger

    def _hook(content):
        line = str(content)
        put_line(line)
        try:
            orig_logger(content)
        except Exception:
            pass

    _tu.Logger = _hook
    try:
        yield
    finally:
        _tu.Logger = orig_logger


def _sniff_checkpoint(path: str) -> dict:
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

        layer_idxs = set()
        for k in sd.keys():
            if k.startswith("model.layers."):
                try:
                    layer_idxs.add(int(k.split(".")[2]))
                except (ValueError, IndexError):
                    pass
        if layer_idxs:
            out["num_hidden_layers"] = max(layer_idxs) + 1

        expert_idxs = set()
        for k in sd.keys():
            if k.startswith("model.layers.0.mlp.experts."):
                try:
                    expert_idxs.add(int(k.split(".")[5]))
                except (ValueError, IndexError):
                    pass
        if expert_idxs:
            out["num_experts"] = max(expert_idxs) + 1

        gate = sd.get("model.layers.0.mlp.experts.0.gate_proj.weight")
        if gate is not None:
            out["moe_intermediate_size"] = int(gate.shape[0])

        qw = sd.get("model.layers.0.self_attn.q_proj.weight")
        kw = sd.get("model.layers.0.self_attn.k_proj.weight")
        if qw is not None and kw is not None:
            out["num_attention_heads_total_dim"] = int(qw.shape[0])
            out["num_kv_heads_total_dim"] = int(kw.shape[0])

        lookahead_proj = sd.get("model.lookahead_router.proj.2.weight")
        if lookahead_proj is not None and out.get("num_experts"):
            total = int(lookahead_proj.shape[0])
            n_exp = int(out["num_experts"])
            if n_exp > 0 and total % n_exp == 0:
                out["lookahead_steps"] = total // n_exp

        return out
    except Exception:
        return {}


# ── Background trainer thread ─────────────────────────────────────

class TrainSession:
    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._log_q: queue.Queue = queue.Queue(maxsize=5000)
        self.status = "idle"
        self.step = 0
        self.loss = None
        # Separate metric history for charts
        self._metrics: list[dict] = []   # [{step, total, ce, aux, temporal, tps}]
        self._metric_lock = threading.Lock()
        # Progress + ETA — set when training kicks off
        self.total_steps: int = 0
        self.t_start: float = 0.0
        # M8a: live generation sample (updated each save_interval).
        self.last_sample: str = ""
        self.last_sample_step: int = 0

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: dict, mode: str):
        if self.is_running():
            return
        self._stop.clear()
        self.status = "running"
        self.step = 0
        self.loss = None
        self._metrics.clear()
        self.total_steps = 0
        self.t_start = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, args=(cfg, mode), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.status = "stopped"

    def _put(self, msg: str):
        try:
            self._log_q.put_nowait(msg)
        except queue.Full:
            pass

    def _put_metric(self, m: dict):
        with self._metric_lock:
            self._metrics.append(m)

    def drain_log(self):
        """Return all pending log lines as a single string (no newline prefix)."""
        lines = []
        while True:
            try:
                lines.append(self._log_q.get_nowait())
            except queue.Empty:
                break
        return "\n".join(lines)

    def get_metrics(self):
        with self._metric_lock:
            return list(self._metrics)

    def _stage_checkpoint_path(self, save_dir: str, mode: str, hidden_size: int) -> str:
        prefix = STAGE_CHECKPOINT_PREFIX.get(mode, "chronos")
        return os.path.join(save_dir, f"{prefix}_{hidden_size}_moe.pth")

    def _default_init_path(self, save_dir: str, mode: str, hidden_size: int) -> str:
        upstream = STAGE_DEFAULT_INIT.get(mode)
        if not upstream:
            return ""
        return os.path.join(save_dir, f"{upstream}_{hidden_size}_moe.pth")

    def _topology_mismatches(self, sniffed: dict, model_cfg_kwargs: dict) -> list[str]:
        mismatches = []
        for k in [
            "hidden_size", "num_hidden_layers", "num_experts",
            "moe_intermediate_size", "vocab_size", "lookahead_steps",
        ]:
            if k in sniffed and k in model_cfg_kwargs and int(sniffed[k]) != int(model_cfg_kwargs[k]):
                mismatches.append(f"{k}: ckpt={sniffed[k]} != ui={model_cfg_kwargs[k]}")
        return mismatches

    def _build_stage_args(self, cfg: dict, mode: str, save_dir: str, hidden_size: int):
        reward_spec = (cfg.get("reward_spec") or "toy").strip() or "toy"
        teacher_path = (cfg.get("teacher_path") or "").strip()
        return SimpleNamespace(
            device=cfg.get("device", "cpu"),
            dtype=_normalize_dtype_for_trainer(cfg.get("dtype")),
            learning_rate=float(cfg.get("learning_rate", 5e-4)),
            accumulation_steps=max(1, int(cfg.get("accumulation_steps", 1))),
            grad_clip=float(cfg.get("grad_clip", 1.0)),
            log_interval=max(1, int(cfg.get("log_interval", 10))),
            save_interval=max(1, int(cfg.get("save_interval", 500))),
            save_dir=save_dir,
            epochs=max(1, int(cfg.get("epochs", 1))),
            batch_size=max(1, int(cfg.get("batch_size", 1))),
            max_seq_len=max(8, int(cfg.get("max_seq_len", 512))),
            max_gen_len=max(1, int(cfg.get("max_gen_len", 24))),
            num_generations=max(1, int(cfg.get("num_generations", 4))),
            temperature=float(cfg.get("temperature", 1.0)),
            beta=float(cfg.get("beta", 0.1)),
            alpha=float(cfg.get("alpha", 0.7)),
            lambda_or=float(cfg.get("lambda_or", 0.1)),
            lambda_router_anchor=float(cfg.get("lambda_router_anchor", 0.0)),
            reward=reward_spec,
            teacher_path=teacher_path or os.path.join(save_dir, f"sft_{hidden_size}_moe.pth"),
            hidden_size=hidden_size,
            num_hidden_layers=int(cfg.get("num_hidden_layers", 8)),
            num_experts=int(cfg.get("num_experts", 4)),
            steps=int(cfg.get("max_steps", 0)) or None,
            weight_decay=float(cfg.get("weight_decay", 0.01)),
        )

    def _record_stage_metric(self, mode: str, step: int, result, tps: float = 0.0):
        meta = STAGE_CHART_META.get(mode, STAGE_CHART_META["pretrain"])
        if isinstance(result, tuple):
            if mode == "sft":
                total, ce_loss, aux_loss, lookahead, anchor = result
                metric = {
                    "total": total,
                    "primary": ce_loss,
                    "secondary": lookahead,
                    "tertiary": anchor,
                    "ce": ce_loss,
                    "aux": lookahead,
                    "temporal": anchor,
                }
            elif mode == "dpo":
                total, pref_loss, _aux_loss, lookahead, anchor = result
                metric = {
                    "total": total,
                    "primary": pref_loss,
                    "secondary": anchor,
                    "tertiary": lookahead,
                    "ce": pref_loss,
                    "aux": anchor,
                    "temporal": lookahead,
                }
            elif mode == "orpo":
                total, chosen_nll, or_term, _lookahead, anchor = result
                metric = {
                    "total": total,
                    "primary": chosen_nll,
                    "secondary": or_term,
                    "tertiary": anchor,
                    "ce": chosen_nll,
                    "aux": or_term,
                    "temporal": anchor,
                }
            elif mode == "distill":
                total, kd_loss, ce_loss, _lookahead, anchor = result
                metric = {
                    "total": total,
                    "primary": kd_loss,
                    "secondary": ce_loss,
                    "tertiary": anchor,
                    "ce": kd_loss,
                    "aux": ce_loss,
                    "temporal": anchor,
                }
            else:
                return
        elif isinstance(result, dict) and mode == "grpo":
            metric = {
                "total": result.get("loss", 0.0),
                "primary": result.get("pg_loss", 0.0),
                "secondary": result.get("kl", 0.0),
                "tertiary": result.get("mean_reward", 0.0),
                "ce": result.get("pg_loss", 0.0),
                "aux": result.get("kl", 0.0),
                "temporal": result.get("mean_reward", 0.0),
            }
        else:
            return

        if mode == "pretrain":
            metric.setdefault("primary", metric.get("ce", 0.0))
            metric.setdefault("secondary", metric.get("aux", 0.0))
            metric.setdefault("tertiary", metric.get("temporal", 0.0))

        metric.update({"step": int(step), "tps": float(tps)})
        self.step = int(step)
        self.loss = float(metric["total"])
        self._put_metric(metric)
        self._put(
            f"[{mode.upper()} summary] step={step} total={metric['total']:.4f} "
            f"{meta['primary']}={metric.get('primary', 0.0):.4f} "
            f"{meta['secondary']}={metric.get('secondary', 0.0):.4f} "
            f"{meta['tertiary']}={metric.get('tertiary', 0.0):.4f} "
            f"tps={tps:.2f}"
        )

    def _generate_live_sample(self, model, device: str, cfg: dict, mode: str):
        import torch

        sp = (cfg.get("sample_prompt") or "").strip()
        if not sp:
            sp = _stage_sample_prompt(mode)
        tokenizer = self._load_tokenizer()
        ids = tokenizer.encode(sp, return_tensors="pt").to(device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            gen = ids.clone()
            for _ in range(80):
                o, _ = model(gen, use_cache=False)
                nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, nxt], dim=-1)
                if int(nxt.item()) == (tokenizer.eos_token_id or -1):
                    break
        txt = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
        if was_training:
            model.train()
        self.last_sample = txt
        self.last_sample_step = self.step or self.last_sample_step
        self._put(f"[sample @ step {self.last_sample_step or 0}] {txt[:200]}")

    def _run(self, cfg: dict, mode: str):
        t_start = time.monotonic()
        try:
            import torch
            from torch.utils.data import DataLoader

            from chronos.model.config import ChronosConfig
            from chronos.model.model_chronos import ChronosForCausalLM
            from chronos.model.moe_chronos import ChronosMOEFeedForward

            self._put(f"[{mode.upper()}] Building model...")
            # Pass the FULL config dict so user-set arch overrides (e.g.
            # moe_intermediate_size, num_attention_heads, vocab_size) are
            # actually honored. Previously this method only forwarded a
            # hardcoded subset, so the trainer silently used minimind's
            # auto-derived intermediate_size while the UI estimator showed
            # the user's slider — causing 10x param-count mismatches.
            model_cfg_kwargs = {
                "hidden_size":               cfg.get("hidden_size", 512),
                "num_hidden_layers":         cfg.get("num_hidden_layers", 8),
                "num_experts":               cfg.get("num_experts", 4),
                "num_experts_per_tok":       cfg.get("num_experts_per_tok", 1),
                "num_shared_experts":        cfg.get("num_shared_experts", 1),
                "lookahead_steps":           cfg.get("lookahead_steps", 2),
                "kv_latent_dim":             cfg.get("kv_latent_dim", 64),
                "sliding_window_size":       cfg.get("sliding_window_size", 2048),
                "lambda_balance":            cfg.get("lambda_balance", 5e-4),
                "lambda_temporal":           cfg.get("lambda_temporal", 1e-3),
                "vram_budget_gb":            cfg.get("vram_budget_gb", 4.0),
                "use_hybrid_attention":      cfg.get("use_hybrid_attention", True),
                "use_moe":                   True,
            }
            # Optional overrides: only forward when the user actually set them.
            # For the "auto" sentinel fields (0 = let MiniMindConfig derive
            # it), we skip forwarding when value is 0 so ceil(H·π/64)·64
            # takes effect instead of Linear(H, 0) crashing MPS init.
            AUTO_SENTINEL_KEYS = {"intermediate_size", "moe_intermediate_size"}
            for opt_key, cfg_key in [
                ("num_attention_heads",   "num_attention_heads"),
                ("num_key_value_heads",   "num_key_value_heads"),
                ("rope_dim",              "rope_dim"),
                ("vocab_size",            "vocab_size"),
                ("max_position_embeddings", "max_seq_len"),
                ("intermediate_size",     "moe_intermediate_size"),
                ("moe_intermediate_size", "moe_intermediate_size"),
                ("lambda_lookahead",      "lambda_lookahead"),
                ("lambda_router_anchor",  "lambda_router_anchor"),
                ("pinned_memory_max_fraction", "pinned_memory_max_fraction"),
                ("storage_format",        "storage_format"),
                ("tie_word_embeddings",   "tie_word_embeddings"),
            ]:
                if cfg_key not in cfg:
                    continue
                val = cfg[cfg_key]
                if val in (None, ""):
                    continue
                if opt_key in AUTO_SENTINEL_KEYS and val == 0:
                    continue
                model_cfg_kwargs[opt_key] = val
            model_cfg = ChronosConfig(**model_cfg_kwargs)
            model = ChronosForCausalLM(model_cfg)
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            self._put(f"Model: {params_m:.1f}M params  "
                      f"(H={model_cfg.hidden_size}, L={model_cfg.num_hidden_layers}, "
                      f"E={model_cfg.num_experts}, ffn={model_cfg.intermediate_size}, "
                      f"vocab={model_cfg.vocab_size})")

            save_dir = cfg.get("save_dir", "./out")
            ckp_path = self._stage_checkpoint_path(save_dir, mode, model_cfg.hidden_size)

            # Init-weight resolution:
            #   - pretrain: optionally resume from its own checkpoint if present
            #   - other stages: require an upstream weight (explicit init_weight
            #     or the stage's default predecessor checkpoint).
            init_weight = (cfg.get("init_weight") or "").strip()
            if mode == "pretrain":
                resume_path = init_weight or ckp_path
                if os.path.exists(resume_path):
                    sniffed = _sniff_checkpoint(resume_path)
                    mismatches = self._topology_mismatches(sniffed, model_cfg_kwargs)
                    if mismatches:
                        self._put(
                            "Resume checkpoint exists but does not match current UI config; "
                            "starting fresh instead.\n  " + "\n  ".join(mismatches)
                        )
                    else:
                        weights = torch.load(resume_path, map_location="cpu")
                        model.load_state_dict(weights, strict=False)
                        self._put(f"Resumed from {resume_path}")
                else:
                    self._put("Pretraining from random init")
            else:
                load_path = init_weight or self._default_init_path(save_dir, mode, model_cfg.hidden_size)
                if not os.path.exists(load_path):
                    raise FileNotFoundError(
                        f"[{mode.upper()}] requires an upstream checkpoint to initialize from. "
                        f"Tried: {load_path}\n"
                        f"Set 'init_weight' in the Train tab to point at a valid .pth, "
                        f"or run the prior stage first."
                    )

                sniffed = _sniff_checkpoint(load_path)
                mismatch_hints = self._topology_mismatches(sniffed, model_cfg_kwargs)
                if mismatch_hints:
                    raise RuntimeError(
                        f"[{mode.upper()}] init checkpoint topology does not match current UI config:\n  "
                        + "\n  ".join(mismatch_hints)
                        + "\nLoad the matching preset/config first, or switch init_weight to the correct checkpoint."
                    )

                weights = torch.load(load_path, map_location="cpu")
                model.load_state_dict(weights, strict=False)
                self._put(f"[{mode.upper()}] Initialized from {load_path}")

            requested_backend = (cfg.get("train_backend") or "auto").strip().lower() or "auto"
            resolved_backend, device = resolve_training_device(requested_backend)
            cfg["train_backend"] = requested_backend
            cfg["device"] = device
            self._put(f"Training backend: {resolved_backend} (requested: {requested_backend})")
            self._put(f"Device: {device}")
            model = model.to(device)
            from chronos.trainer.optim_utils import build_optimizer, get_lr, apply_lr

            data_path = cfg.get("data_path", "")
            max_seq_len = cfg.get("max_seq_len", 512)
            batch_size = cfg.get("batch_size", 4)
            accum = max(1, int(cfg.get("accumulation_steps", 8)))
            epochs = cfg.get("epochs", 1)
            save_interval = cfg.get("save_interval", 500)
            log_interval = max(1, cfg.get("log_interval", 10))
            val_ratio = float(cfg.get("val_ratio", 0.05))
            max_steps = int(cfg.get("max_steps", 0)) or None  # 0 => no cap
            base_lr = float(cfg.get("learning_rate", 5e-4))
            weight_decay = float(cfg.get("weight_decay", 0.01))
            optimizer = build_optimizer(model, lr=base_lr, weight_decay=weight_decay)

            if not data_path or not os.path.exists(data_path):
                self._put("No dataset found — running synthetic smoke test (50 steps)")
                loader = self._synthetic_loader(model_cfg.vocab_size, max_seq_len, batch_size, n=50)
                val_loader = None
            else:
                tokenizer = self._load_tokenizer()
                from chronos.data.flexible_dataset import (
                    FlexibleDataset, StreamingSFTDataset, StreamingDPODataset,
                )
                if mode == "pretrain":
                    full_ds = FlexibleDataset(data_path, tokenizer, max_length=max_seq_len)
                elif mode in {"sft", "distill"}:
                    full_ds = StreamingSFTDataset(data_path, tokenizer, max_length=max_seq_len)
                elif mode in {"dpo", "orpo"}:
                    full_ds = StreamingDPODataset(data_path, tokenizer, max_length=max_seq_len)
                else:
                    full_ds = None
                # Deterministic val split: every 1/val_ratio-th record goes to val
                # (idx-modulo split → reproducible without rescanning the file).
                if full_ds is not None and val_ratio > 0 and len(full_ds) >= 20:
                    stride = max(2, int(round(1.0 / val_ratio)))
                    val_idx   = [i for i in range(len(full_ds)) if i % stride == 0]
                    train_idx = [i for i in range(len(full_ds)) if i % stride != 0]
                    from torch.utils.data import Subset
                    train_ds = Subset(full_ds, train_idx)
                    val_ds   = Subset(full_ds, val_idx)
                    self._put(f"Split: train={len(train_ds)}  val={len(val_ds)}")
                elif full_ds is not None:
                    train_ds, val_ds = full_ds, None
                else:
                    train_ds, val_ds = None, None
                loader = (
                    DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
                    if train_ds is not None else None
                )
                val_loader = (
                    DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=False)
                    if val_ds is not None else None
                )

            if mode in {"grpo"}:
                tokenizer = self._load_tokenizer()
                from chronos.trainer.grpo_trainer import load_grpo_prompts
                prompts = load_grpo_prompts(data_path, max_prompts=max_steps)
                if not prompts:
                    raise ValueError(f"[{mode.upper()}] no prompts found in {data_path}")
                planned = max(1, len(prompts) * epochs)
                self.total_steps = min(planned, max_steps) if (max_steps and planned) else planned
            else:
                prompts = None

            model.train()
            global_step = 0
            step_t = time.monotonic()
            best_val = float("inf")
            best_path = ckp_path.replace(".pth", ".best.pth")
            # Steps remaining for ETA / progress bar — respect max_steps cap.
            try:
                planned = max(1, len(loader) * epochs) if loader is not None else 0
            except Exception:
                planned = 0
            if mode != "grpo":
                self.total_steps = min(planned, max_steps) if (max_steps and planned) else (planned or (max_steps or 0))

            stage_args = self._build_stage_args(cfg, mode, save_dir, model_cfg.hidden_size)
            stage_args.device = device

            if mode in {"sft", "dpo", "orpo", "grpo", "distill"}:
                tokenizer = self._load_tokenizer()

            def _eval_val() -> float:
                if val_loader is None:
                    return float("nan")
                model.eval()
                tot, n = 0.0, 0
                with torch.no_grad():
                    for k, (vx, vy) in enumerate(val_loader):
                        if k >= 50:  # cap eval cost
                            break
                        vx = vx.to(device); vy = vy.to(device)
                        vout, _ = model(vx, labels=vy)
                        tot += float(vout.loss.item()); n += 1
                model.train()
                return tot / max(n, 1)

            if mode in {"sft", "dpo", "orpo", "grpo", "distill"}:
                with _trainer_logger_sink(self._put):
                    if mode == "sft":
                        from chronos.trainer.sft_trainer import ChronosSFTTrainer

                        trainer = ChronosSFTTrainer(model, model_cfg, stage_args, tokenizer)
                        if float(getattr(model_cfg, "lambda_router_anchor", 0.0)) > 0 and loader is not None:
                            first_batch = next(iter(loader))
                            trainer.set_calibration_batch(first_batch[0])
                            self._put("[SFT] Router anchor reference captured.")
                        iters = len(loader) if stage_args.steps is None else min(stage_args.steps, len(loader))
                        self.total_steps = max(1, stage_args.epochs * iters)
                        for epoch in range(stage_args.epochs):
                            epoch_start = time.monotonic()
                            for step_idx, (ids, labels) in enumerate(loader, start=1):
                                if self._stop.is_set():
                                    break
                                if stage_args.steps is not None and step_idx > stage_args.steps:
                                    break
                                result = trainer.train_step(ids, labels, epoch * iters + step_idx, self.total_steps)
                                if step_idx % stage_args.log_interval == 0 or step_idx == iters:
                                    elapsed = max(time.monotonic() - epoch_start, 1e-6)
                                    self._record_stage_metric(mode, epoch * iters + step_idx, result, tps=step_idx / elapsed)
                                if step_idx % stage_args.save_interval == 0:
                                    trainer._save(epoch, step_idx)
                                    self._generate_live_sample(model, device, cfg, mode)
                            if self._stop.is_set() or stage_args.steps is not None:
                                break
                        trainer._save(epoch=stage_args.epochs - 1, step=iters)
                    elif mode == "dpo":
                        from chronos.trainer.dpo_trainer import ChronosDPOTrainer

                        trainer = ChronosDPOTrainer(model, model_cfg, stage_args, tokenizer)
                        if float(getattr(model_cfg, "lambda_router_anchor", 0.0)) > 0 and loader is not None:
                            first = next(iter(loader))
                            calib = torch.cat([first["x_chosen"], first["x_rejected"]], dim=0)
                            trainer.set_calibration_batch(calib)
                            self._put("[DPO] Router anchor reference captured.")
                        iters = len(loader) if stage_args.steps is None else min(stage_args.steps, len(loader))
                        self.total_steps = max(1, stage_args.epochs * iters)
                        for epoch in range(stage_args.epochs):
                            epoch_start = time.monotonic()
                            for step_idx, batch in enumerate(loader, start=1):
                                if self._stop.is_set():
                                    break
                                if stage_args.steps is not None and step_idx > stage_args.steps:
                                    break
                                result = trainer.train_step(batch, epoch * iters + step_idx, self.total_steps)
                                if step_idx % stage_args.log_interval == 0 or step_idx == iters:
                                    elapsed = max(time.monotonic() - epoch_start, 1e-6)
                                    self._record_stage_metric(mode, epoch * iters + step_idx, result, tps=step_idx / elapsed)
                                if step_idx % stage_args.save_interval == 0:
                                    trainer._save(epoch, step_idx)
                                    self._generate_live_sample(model, device, cfg, mode)
                            if self._stop.is_set() or stage_args.steps is not None:
                                break
                        trainer._save(epoch=stage_args.epochs - 1, step=iters)
                    elif mode == "orpo":
                        from chronos.trainer.orpo_trainer import ChronosORPOTrainer

                        trainer = ChronosORPOTrainer(model, model_cfg, stage_args, tokenizer)
                        if float(getattr(model_cfg, "lambda_router_anchor", 0.0)) > 0 and loader is not None:
                            first = next(iter(loader))
                            calib = torch.cat([first["x_chosen"], first["x_rejected"]], dim=0)
                            trainer.set_calibration_batch(calib)
                            self._put("[ORPO] Router anchor reference captured.")
                        iters = len(loader) if stage_args.steps is None else min(stage_args.steps, len(loader))
                        self.total_steps = max(1, stage_args.epochs * iters)
                        for epoch in range(stage_args.epochs):
                            epoch_start = time.monotonic()
                            for step_idx, batch in enumerate(loader, start=1):
                                if self._stop.is_set():
                                    break
                                if stage_args.steps is not None and step_idx > stage_args.steps:
                                    break
                                result = trainer.train_step(batch, epoch * iters + step_idx, self.total_steps)
                                if step_idx % stage_args.log_interval == 0 or step_idx == iters:
                                    elapsed = max(time.monotonic() - epoch_start, 1e-6)
                                    self._record_stage_metric(mode, epoch * iters + step_idx, result, tps=step_idx / elapsed)
                                if step_idx % stage_args.save_interval == 0:
                                    trainer._save(epoch, step_idx)
                                    self._generate_live_sample(model, device, cfg, mode)
                            if self._stop.is_set() or stage_args.steps is not None:
                                break
                        trainer._save(epoch=stage_args.epochs - 1, step=iters)
                    elif mode == "grpo":
                        from chronos.trainer.grpo_trainer import ChronosGRPOTrainer
                        from chronos.trainer.reward import build_reward_fn

                        reward_fn = build_reward_fn(stage_args.reward)
                        trainer = ChronosGRPOTrainer(model, model_cfg, stage_args, tokenizer, reward_fn=reward_fn)
                        if float(getattr(model_cfg, "lambda_router_anchor", 0.0)) > 0 and prompts:
                            calib = tokenizer(
                                prompts[0], return_tensors="pt", truncation=True, max_length=stage_args.max_seq_len
                            ).input_ids
                            trainer.set_calibration_batch(calib)
                            self._put("[GRPO] Router anchor reference captured.")
                        iters = len(prompts) if stage_args.steps is None else min(stage_args.steps, len(prompts))
                        self.total_steps = max(1, stage_args.epochs * iters)
                        for epoch in range(stage_args.epochs):
                            epoch_start = time.monotonic()
                            for step_idx, prompt in enumerate(prompts, start=1):
                                if self._stop.is_set():
                                    break
                                if step_idx > iters:
                                    break
                                result = trainer.train_step(prompt, epoch * iters + step_idx, self.total_steps)
                                if step_idx % stage_args.log_interval == 0 or step_idx == iters:
                                    elapsed = max(time.monotonic() - epoch_start, 1e-6)
                                    self._record_stage_metric(mode, epoch * iters + step_idx, result, tps=step_idx / elapsed)
                                if step_idx % stage_args.save_interval == 0:
                                    trainer._save(epoch, step_idx)
                                    self._generate_live_sample(model, device, cfg, mode)
                            if self._stop.is_set() or stage_args.steps is not None:
                                break
                        trainer._save(epoch=stage_args.epochs - 1, step=iters)
                    elif mode == "distill":
                        from chronos.trainer.distill_trainer import ChronosDistillTrainer

                        teacher_path = stage_args.teacher_path
                        if not os.path.exists(teacher_path):
                            raise FileNotFoundError(f"[DISTILL] teacher not found: {teacher_path}")
                        teacher = ChronosForCausalLM(model_cfg).to(device)
                        t_state = torch.load(teacher_path, map_location=device)
                        teacher.load_state_dict(t_state, strict=False)
                        teacher.eval().requires_grad_(False)
                        self._put(f"[DISTILL] Teacher: {teacher_path}")
                        trainer = ChronosDistillTrainer(model, teacher, model_cfg, stage_args, tokenizer)
                        if float(getattr(model_cfg, "lambda_router_anchor", 0.0)) > 0 and loader is not None:
                            first = next(iter(loader))
                            trainer.set_calibration_batch(first[0])
                            self._put("[DISTILL] Router anchor reference captured.")
                        iters = len(loader) if stage_args.steps is None else min(stage_args.steps, len(loader))
                        self.total_steps = max(1, stage_args.epochs * iters)
                        for epoch in range(stage_args.epochs):
                            epoch_start = time.monotonic()
                            for step_idx, (ids, labels) in enumerate(loader, start=1):
                                if self._stop.is_set():
                                    break
                                if stage_args.steps is not None and step_idx > stage_args.steps:
                                    break
                                result = trainer.train_step(ids, labels, epoch * iters + step_idx, self.total_steps)
                                if step_idx % stage_args.log_interval == 0 or step_idx == iters:
                                    elapsed = max(time.monotonic() - epoch_start, 1e-6)
                                    self._record_stage_metric(mode, epoch * iters + step_idx, result, tps=step_idx / elapsed)
                                if step_idx % stage_args.save_interval == 0:
                                    trainer._save(epoch, step_idx)
                                    self._generate_live_sample(model, device, cfg, mode)
                            if self._stop.is_set() or stage_args.steps is not None:
                                break
                        trainer._save(epoch=stage_args.epochs - 1, step=iters)
                self._generate_live_sample(model, device, cfg, mode)
                elapsed = time.monotonic() - t_start
                self._put(f"[{mode.upper()}] Training complete in {elapsed:.1f}s. Saved → {ckp_path}")
                self.status = "finished"
                return

            stop_outer = False
            for epoch in range(epochs):
                if self._stop.is_set() or stop_outer:
                    break
                for step, (input_ids, labels) in enumerate(loader, start=1):
                    if self._stop.is_set():
                        break
                    if max_steps and global_step >= max_steps:
                        stop_outer = True
                        break
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)

                    out, lp = model(input_ids, labels=labels)
                    moe_layers = [l.mlp for l in model.model.layers
                                  if isinstance(l.mlp, ChronosMOEFeedForward)]

                    aux_val = out.aux_loss.item() if hasattr(out, "aux_loss") else 0.0

                    # Use the shared loss helper so λ_temporal / λ_lookahead
                    # actually flow gradient back to the gate. The previous
                    # path passed the detached `last_router_probs` into
                    # total_loss, which made the regularizers inert (the
                    # gate received only CE gradient, so router-locality was
                    # never learned even after thousands of steps).
                    from chronos.trainer.loss_mixin import (
                        chronos_loss_term, collect_router_probs,
                    )
                    loss = chronos_loss_term(
                        model, out.loss, lp, model_cfg, aux_loss=out.aux_loss,
                    )
                    # Logging-only: detached probs are fine here.
                    router_4d_det = collect_router_probs(model, with_grad=False)
                    if router_4d_det is not None and router_4d_det.shape[1] > 1:
                        from chronos.model.temporal_loss import temporal_locality_loss
                        temp_val = temporal_locality_loss(
                            router_4d_det.mean(dim=2)
                        ).item()
                    else:
                        temp_val = 0.0

                    (loss / accum).backward()
                    global_step += 1
                    self.step = global_step
                    self.loss = loss.item()

                    # LR schedule (warmup → cosine). Refresh every step.
                    sched_total = self.total_steps or planned or (global_step + 1)
                    apply_lr(optimizer, get_lr(global_step, sched_total, base_lr))

                    # Optimizer.step() once per `accum` micro-batches.
                    # Off-by-one fix: trigger when (global_step % accum == 0).
                    # Previous code did the same check but had no warmup-aware
                    # accumulation count, occasionally stepping after step 1.
                    if global_step % accum == 0:
                        import torch.nn as nn
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    if global_step % log_interval == 0:
                        now = time.monotonic()
                        tps = log_interval / max(now - step_t, 1e-6)
                        step_t = now
                        ce_val = out.loss.item()
                        tot_val = loss.item()

                        self._put(
                            f"Epoch {epoch+1}/{epochs}  Step {global_step}  "
                            f"loss={tot_val:.4f}  ce={ce_val:.4f}  "
                            f"aux={aux_val:.4f}  temporal={temp_val:.4f}  "
                            f"steps/s={tps:.2f}  lr={optimizer.param_groups[0]['lr']:.2e}"
                        )
                        self._put_metric({
                            "step": global_step,
                            "total": tot_val,
                            "ce": ce_val,
                            "aux": aux_val,
                            "temporal": temp_val,
                            "tps": tps,
                        })

                    if global_step % save_interval == 0:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(
                            {k: v.half().cpu() for k, v in model.state_dict().items()},
                            ckp_path
                        )
                        self._put(f"Saved checkpoint → {ckp_path}")

                        # Validation eval + best-ckpt tracking.
                        v = _eval_val()
                        if v == v:  # not NaN
                            self._put(f"Validation loss: {v:.4f}  (best={best_val:.4f})")
                            self._put_metric({
                                "step": global_step, "total": v, "ce": v,
                                "aux": 0.0, "temporal": 0.0, "tps": 0.0,
                                "val": True,
                                "val_loss": v,
                            })
                            if v < best_val:
                                best_val = v
                                torch.save(
                                    {k: vv.half().cpu() for k, vv in model.state_dict().items()},
                                    best_path,
                                )
                                self._put(f"  ↑ new best → {best_path}")

                        # M8a: live qualitative sample so the user sees
                        # whether the model is actually learning, not just
                        # whether the loss number went down. We greedy-decode
                        # ~80 tokens from the user-supplied prompt, and fall
                        # back to a stage-aware evaluation prompt if blank.
                        try:
                            sp = (cfg.get("sample_prompt") or "").strip()
                            if not sp:
                                sp = _stage_sample_prompt(mode)
                            tokenizer = self._load_tokenizer()
                            ids = tokenizer.encode(sp, return_tensors="pt").to(device)
                            model.eval()
                            with torch.no_grad():
                                gen = ids.clone()
                                for _ in range(80):
                                    o, _ = model(gen, use_cache=False)
                                    nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                                    gen = torch.cat([gen, nxt], dim=-1)
                                    if int(nxt.item()) == (tokenizer.eos_token_id or -1):
                                        break
                            txt = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
                            model.train()
                            self.last_sample = txt
                            self.last_sample_step = global_step
                            self._put(f"[sample @ step {global_step}] {txt[:200]}")
                        except Exception as gen_err:
                            self._put(f"[sample failed: {gen_err}]")

            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {k: v.half().cpu() for k, v in model.state_dict().items()},
                ckp_path
            )
            elapsed = time.monotonic() - t_start
            self._put(f"Training complete in {elapsed:.1f}s. Saved → {ckp_path}")
            self.status = "finished"

        except Exception as e:
            import traceback
            self._put(f"[ERROR] {e}\n{traceback.format_exc()}")
            self.status = "error"

    def _synthetic_loader(self, vocab_size, seq_len, batch_size, n=50):
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        ids = torch.randint(0, vocab_size, (n * batch_size, seq_len))
        ds = TensorDataset(ids, ids.clone())
        return DataLoader(ds, batch_size=batch_size)

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(chronos.deps.get_tokenizer_path())


_session = TrainSession()


def _fmt_eta(seconds: float) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


def _make_loss_chart(metrics: list[dict], mode: str = "pretrain", total_steps: int = 0, t_start: float = 0.0):
    """Build matplotlib figure: loss curves + throughput + progress/ETA."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        meta = STAGE_CHART_META.get(mode, STAGE_CHART_META["pretrain"])

        # Close any prior figures held by this thread to avoid the
        # "More than 20 figures opened" warning Gradio's Plot component triggers.
        plt.close("all")

        if not metrics:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="#888")
            ax.set_axis_off()
            plt.tight_layout()
            return fig

        steps = [m["step"] for m in metrics]
        cur_step = steps[-1]
        elapsed = max(0.0, time.monotonic() - t_start) if t_start else 0.0
        # ETA from average step rate so far (not last point — smoother).
        rate = cur_step / elapsed if elapsed > 0 else 0.0
        remaining = (total_steps - cur_step) / rate if rate > 0 and total_steps else float("nan")
        progress_pct = min(100.0, 100.0 * cur_step / total_steps) if total_steps else 0.0

        # M8b: split val markers from train metrics.
        train_metrics = [m for m in metrics if not m.get("val")]
        val_metrics   = [m for m in metrics if m.get("val")]
        train_steps = [m["step"] for m in train_metrics]
        val_steps   = [m["step"] for m in val_metrics]

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5),
                                 gridspec_kw={"width_ratios": [4, 3, 3]})
        fig.patch.set_facecolor("#1a1a2e")

        # ── Left: loss curves
        ax = axes[0]
        ax.set_facecolor("#16213e")
        if train_metrics:
            ax.plot(train_steps, [m["total"] for m in train_metrics],    color="#e94560", lw=1.5, label="train total")
            ax.plot(train_steps, [m.get("primary", m.get("ce", 0.0)) for m in train_metrics],
                    color="#0f3460", lw=1.2, label=meta["primary"])
            ax.plot(train_steps, [m.get("secondary", m.get("aux", 0.0)) for m in train_metrics],
                    color="#533483", lw=1.0, label=meta["secondary"])
            ax.plot(train_steps, [m.get("tertiary", m.get("temporal", 0.0)) for m in train_metrics],
                    color="#e2b714", lw=0.9, label=meta["tertiary"], ls="--")
        if val_metrics:
            ax.plot(val_steps, [m.get("primary", m.get("ce", 0.0)) for m in val_metrics], color="#39d98a",
                    lw=2.0, marker="o", markersize=4, label=meta["val"])
        ax.set_title(meta["title"], color="white", fontsize=11)
        ax.set_xlabel("Step", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=8, facecolor="#16213e", labelcolor="white")

        # ── Middle: stage-aware speed panel
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        tps_steps = [m["step"] for m in train_metrics if m.get("tps", 0) > 0]
        tps_vals  = [m["tps"]  for m in train_metrics if m.get("tps", 0) > 0]
        overlay_ax = None
        if mode == "grpo":
            reward_steps = [m["step"] for m in train_metrics if "tertiary" in m]
            reward_vals = [m.get("tertiary", 0.0) for m in train_metrics if "tertiary" in m]
            if tps_steps:
                ax2.plot(
                    tps_steps,
                    tps_vals,
                    color="#00b4d8",
                    lw=1.6,
                    label=meta.get("speed_label", "steps/s"),
                )
                ax2.fill_between(tps_steps, tps_vals, alpha=0.18, color="#00b4d8")
            if reward_steps:
                overlay_ax = ax2.twinx()
                overlay_ax.set_facecolor("none")
                overlay_ax.plot(
                    reward_steps,
                    reward_vals,
                    color="#e2b714",
                    lw=1.3,
                    ls="--",
                    label=meta.get("speed_overlay", meta["tertiary"]),
                )
                overlay_ax.tick_params(colors="#f1d36b")
                overlay_ax.set_ylabel(meta.get("speed_overlay", meta["tertiary"]), color="#f1d36b", fontsize=9)
                for spine in overlay_ax.spines.values():
                    spine.set_edgecolor("#333")
        elif tps_steps:
            ax2.plot(
                tps_steps,
                tps_vals,
                color="#00b4d8",
                lw=1.5,
                label=meta.get("speed_label", "steps/s"),
            )
            ax2.fill_between(tps_steps, tps_vals, alpha=0.2, color="#00b4d8")
        ax2.set_title(meta.get("speed_title", "Training Speed (steps/s)"), color="white", fontsize=11)
        ax2.set_xlabel("Step", color="#aaa")
        ax2.set_ylabel(meta.get("speed_label", "steps/s"), color="#8ae9ff", fontsize=9)
        ax2.tick_params(colors="#aaa")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")
        if tps_steps or overlay_ax is not None:
            lines, labels = ax2.get_legend_handles_labels()
            if overlay_ax is not None:
                extra_lines, extra_labels = overlay_ax.get_legend_handles_labels()
                lines += extra_lines
                labels += extra_labels
            if labels:
                ax2.legend(lines, labels, fontsize=8, facecolor="#16213e", labelcolor="white", loc="upper right")

        # ── Right: progress bar + numeric panel
        ax3 = axes[2]
        ax3.set_facecolor("#16213e")
        # Horizontal bar
        bar_y = 0.65
        ax3.barh([bar_y], [100], height=0.18, color="#2a2a44", edgecolor="#444")
        ax3.barh([bar_y], [progress_pct], height=0.18, color="#39d98a")
        ax3.set_xlim(0, 100); ax3.set_ylim(0, 1)
        ax3.set_xticks([0, 25, 50, 75, 100])
        ax3.set_yticks([])
        ax3.tick_params(colors="#aaa")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#333")

        # Big text overlay
        ax3.text(50, 0.92, f"Progress  {progress_pct:5.1f}%",
                 ha="center", va="center", color="white", fontsize=12, weight="bold")
        eta_str = _fmt_eta(remaining)
        elapsed_str = _fmt_eta(elapsed)
        ax3.text(50, 0.42, f"step {cur_step:,} / {total_steps or '?':,}",
                 ha="center", va="center", color="#cfd8e3", fontsize=11)
        ax3.text(50, 0.25, f"elapsed {elapsed_str}    ETA {eta_str}",
                 ha="center", va="center", color="#9aa6b2", fontsize=10)
        ax3.text(50, 0.10, f"~{rate:.2f} {meta.get('speed_label', 'steps/s')} avg",
                 ha="center", va="center", color="#9aa6b2", fontsize=9)
        ax3.set_title("Progress & ETA", color="white", fontsize=11)

        plt.tight_layout(pad=1.5)
        return fig

    except Exception:
        return None


def build_train_tab(config_state: gr.State, cfg_save_dir: gr.Textbox):
    with gr.Tab(t("tab.train")) as tab:
        register_translatable(tab, "tab.train")

        # ── Train owns its dataset choice (moved out of Config tab) ─
        stage_help = gr.Markdown(_stage_text(STAGE_HELP_TEXT, "pretrain"))
        with gr.Row():
            data_path = gr.Textbox(
                label=t("config.data_path"),
                value=STAGE_DEFAULT_DATA["pretrain"],
                placeholder=STAGE_DEFAULT_DATA["pretrain"],
                scale=3,
            )
            register_translatable(data_path, "config.data_path")
            dataset_upload = gr.File(
                label=t("train.dataset_upload"),
                file_types=[".jsonl"], type="filepath", scale=2,
            )
            register_translatable(dataset_upload, "train.dataset_upload")
        dataset_upload.change(
            fn=lambda fp: fp or "",
            inputs=dataset_upload,
            outputs=data_path,
        )

        # ── Mode + controls ───────────────────────────────────────
        with gr.Row():
            mode = gr.Radio(
                STAGE_UI_ORDER,
                value="pretrain", label=t("train.mode")
            )
            register_translatable(mode, "train.mode")
            train_backend = gr.Dropdown(
                choices=_train_backend_dropdown_choices(),
                value=_default_train_backend_value(),
                label=t("train.backend"),
                scale=1,
            )
            register_translatable(train_backend, "train.backend")
            status_box = gr.Textbox(
                value="idle", label=t("train.status"),
                interactive=False, scale=1,
            )
            register_translatable(status_box, "train.status")

        # ── Init-weight / stage-specific extras ────────────────────
        with gr.Row():
            init_weight = gr.Textbox(
                label=t("train.init_weight"),
                placeholder=_stage_init_placeholder("pretrain"),
                scale=3,
            )
            register_translatable(init_weight, "train.init_weight")
            teacher_path = gr.Textbox(
                label=t("pipeline.teacher_path"),
                placeholder=_distill_teacher_placeholder(),
                visible=False,
                scale=2,
            )
            register_translatable(teacher_path, "pipeline.teacher_path")

        with gr.Row():
            max_steps_in = gr.Number(
                value=0, precision=0,
                label=t("train.max_steps"), scale=1,
            )
            val_ratio_in = gr.Slider(
                0.0, 0.5, value=0.05, step=0.01,
                label=t("train.val_ratio"), scale=2,
            )
            weight_decay_in = gr.Number(
                value=0.01, precision=4,
                label=t("train.weight_decay"), scale=1,
            )
            register_translatable(max_steps_in,    "train.max_steps")
            register_translatable(val_ratio_in,    "train.val_ratio")
            register_translatable(weight_decay_in, "train.weight_decay")

        with gr.Row():
            reward_spec_in = gr.Textbox(
                value="toy",
                label="reward",
                placeholder="toy or lm:/path/to/reward-model",
                visible=False,
                scale=2,
            )
            grpo_temp_in = gr.Number(
                value=1.0, precision=3,
                label="temperature",
                visible=False,
                scale=1,
            )
            grpo_num_gen_in = gr.Number(
                value=4, precision=0,
                label="num_generations",
                visible=False,
                scale=1,
            )

        with gr.Row():
            sample_prompt_in = gr.Textbox(
                value=_stage_sample_prompt("pretrain"),
                label=t("train.sample_prompt"),
                placeholder=_stage_sample_prompt("pretrain"),
                scale=4,
            )
            register_translatable(sample_prompt_in, "train.sample_prompt")

        with gr.Row():
            start_btn = gr.Button(t("train.start"), variant="primary")
            stop_btn  = gr.Button(t("train.stop"),  variant="stop")
            clear_btn = gr.Button(t("train.clear_log"), variant="secondary")
            register_translatable(start_btn, "train.start")
            register_translatable(stop_btn,  "train.stop")
            register_translatable(clear_btn, "train.clear_log")

        # ── Metrics chart ─────────────────────────────────────────
        chart = gr.Plot(label=t("train.chart"), show_label=True)
        register_translatable(chart, "train.chart")

        # ── Scalar readouts ───────────────────────────────────────
        with gr.Row():
            step_box     = gr.Number(label=t("train.step"),     value=0,   interactive=False, precision=0)
            loss_box     = gr.Number(label=STAGE_METRIC_LABELS["pretrain"][0], value=0.0, interactive=False, precision=4)
            ce_box       = gr.Number(label=STAGE_METRIC_LABELS["pretrain"][1], value=0.0, interactive=False, precision=4)
            aux_box      = gr.Number(label=STAGE_METRIC_LABELS["pretrain"][2], value=0.0, interactive=False, precision=4)
            tps_box      = gr.Number(label=t("train.tps"),      value=0.0, interactive=False, precision=2)
            register_translatable(step_box, "train.step")
            register_translatable(tps_box,  "train.tps")

        # ── Scrollable log ────────────────────────────────────────
        log_box = gr.Textbox(
            label=t("train.log"), lines=18, max_lines=30,
            interactive=False, autoscroll=True,
        )
        register_translatable(log_box, "train.log")

        # ── Live generation sample (updated each save_interval) ───
        sample_box = gr.Textbox(
            label=t("train.sample_output"), lines=6,
            interactive=False, autoscroll=False,
        )
        register_translatable(sample_box, "train.sample_output")

        # ── Callbacks ─────────────────────────────────────────────

        def sync_stage_fields(train_mode, current_prompt, current_data_path, current_init_weight):
            current = (current_prompt or "").strip()
            default_upstream = STAGE_DEFAULT_INIT.get(train_mode, "")
            help_md = _stage_text(STAGE_HELP_TEXT, train_mode)
            data_default = STAGE_DEFAULT_DATA.get(train_mode, "")
            metric_labels = STAGE_METRIC_LABELS.get(train_mode, STAGE_METRIC_LABELS["pretrain"])
            init_placeholder = _stage_init_placeholder(train_mode)
            current_data = (current_data_path or "").strip()
            current_init = _normalize_stage_init_value(current_init_weight)
            data_value = current_data
            if (not current_data) or current_data in DEFAULT_STAGE_DATA_VALUES:
                data_value = data_default
            init_value = current_init
            teacher_visible = train_mode == "distill"
            reward_visible = train_mode == "grpo"
            if (not current) or current in DEFAULT_SAMPLE_PROMPT_VALUES:
                prompt_update = gr.update(
                    value=_stage_sample_prompt(train_mode),
                    placeholder=_stage_sample_prompt(train_mode),
                )
            else:
                prompt_update = gr.update(placeholder=_stage_sample_prompt(train_mode))
            return (
                prompt_update,
                gr.update(value=help_md),
                gr.update(value=data_value, placeholder=data_default),
                gr.update(value=init_value, placeholder=init_placeholder),
                gr.update(visible=teacher_visible, placeholder=_distill_teacher_placeholder()),
                gr.update(visible=reward_visible),
                gr.update(visible=reward_visible),
                gr.update(visible=reward_visible),
                gr.update(label=metric_labels[0]),
                gr.update(label=metric_labels[1]),
                gr.update(label=metric_labels[2]),
            )

        def start_training(cfg, train_mode, selected_backend, dpath, iw, ms, vr, wd, reward_spec, grpo_temp, grpo_num_gen, teacher, sp):
            if _session.is_running():
                return "already running", 0, 0.0, 0.0, 0.0, 0.0, "Already running.\n", None, ""
            cfg = dict(cfg) if cfg else {}
            requested_backend = (selected_backend or "auto").strip().lower() or "auto"
            _, resolved_device = resolve_training_device(requested_backend)
            cfg["train_backend"] = requested_backend
            cfg["device"] = resolved_device
            cfg["data_path"] = dpath or cfg.get("data_path", "")
            cfg["init_weight"] = _normalize_stage_init_value(iw)
            cfg["max_steps"] = int(ms or 0)
            cfg["val_ratio"] = float(vr or 0.0)
            cfg["weight_decay"] = float(wd or 0.0)
            cfg["reward_spec"] = reward_spec or "toy"
            cfg["temperature"] = float(grpo_temp or 1.0)
            cfg["num_generations"] = int(grpo_num_gen or 4)
            cfg["teacher_path"] = teacher or ""
            cfg["sample_prompt"] = sp or ""
            _session.start(cfg, train_mode)
            return "running", 0, 0.0, 0.0, 0.0, 0.0, "Starting...\n", None, ""

        def stop_training():
            _session.stop()
            return "stopped"

        def clear_log():
            return ""

        def poll(current_log: str, current_mode: str):
            new_lines = _session.drain_log()
            # Append new lines to existing log; keep last 500 lines to avoid unbounded growth
            if new_lines:
                combined = (current_log + "\n" + new_lines).lstrip("\n")
            else:
                combined = current_log

            lines = combined.split("\n")
            if len(lines) > 500:
                combined = "\n".join(lines[-500:])

            metrics = _session.get_metrics()
            fig = _make_loss_chart(metrics, current_mode, _session.total_steps, _session.t_start)

            # Latest scalar values from last metric entry
            if metrics:
                last = metrics[-1]
                ce_v = last.get("primary", last.get("ce", 0.0))
                aux_v = last.get("secondary", last.get("aux", 0.0))
                tps_v = last.get("tps", 0.0)
            else:
                ce_v = aux_v = tps_v = 0.0

            return (
                _session.status,
                _session.step or 0,
                _session.loss or 0.0,
                ce_v,
                aux_v,
                tps_v,
                combined,
                fig,
                _session.last_sample or "",
            )

        mode.change(
            fn=sync_stage_fields,
            inputs=[mode, sample_prompt_in, data_path, init_weight],
            outputs=[
                sample_prompt_in, stage_help, data_path, init_weight,
                teacher_path, reward_spec_in, grpo_temp_in, grpo_num_gen_in,
                loss_box, ce_box, aux_box,
            ],
        )

        start_btn.click(
            fn=start_training,
            inputs=[config_state, mode, train_backend, data_path, init_weight,
                    max_steps_in, val_ratio_in, weight_decay_in,
                    reward_spec_in, grpo_temp_in, grpo_num_gen_in, teacher_path,
                    sample_prompt_in],
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box,
                     log_box, chart, sample_box],
        )
        stop_btn.click(fn=stop_training, outputs=[status_box])
        clear_btn.click(fn=clear_log, outputs=[log_box])

        timer = gr.Timer(value=2.0)
        timer.tick(
            fn=poll,
            inputs=[log_box, mode],
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box,
                     log_box, chart, sample_box],
        )

    return tab
