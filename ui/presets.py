"""
ui/presets.py

Default + preset configurations matching minimind's MoE defaults.

Each preset is a complete dict that can be:
  1. Loaded into the Config tab sliders via the "Load Preset" button.
  2. Used as the initial value when the user clicks "Reset to MiniMind".
  3. Persisted to / loaded from JSON via the Save/Load Config buttons.

The single source of truth for the *minimind* preset is what
``MiniMindConfig(use_moe=True)`` actually produces — we mirror those values
here so the UI shows the same model the trainer would build.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List


# ── MiniMind MoE — exact defaults from model/model_minimind.py ─────
# These values match MiniMindConfig(hidden_size=768, num_hidden_layers=8,
# use_moe=True) field-for-field, with Chronos-specific extensions left at
# their conservative defaults (no router-anchor by default).
MINIMIND_MOE_DEFAULTS: Dict = {
    # MiniMind core
    "hidden_size":           768,
    "num_hidden_layers":     8,
    "num_attention_heads":   8,
    "num_key_value_heads":   4,
    "vocab_size":            6400,
    "intermediate_size":     0,        # 0 = auto: ceil(H·π/64)·64 → 2432
    "moe_intermediate_size": 0,        # 0 = auto, same value
    "max_seq_len":           512,      # training sequence length (NOT max_position_embeddings)
    "tie_word_embeddings":   True,

    # MoE
    "num_experts":           4,
    "num_experts_per_tok":   2,        # M8c: top-2 doubles activated params at fixed total params
    "num_shared_experts":    1,        # Chronos extension; minimind has no shared

    # Attention shape (Chronos hybrid)
    "kv_latent_dim":         64,
    "rope_dim":              32,
    "sliding_window_size":   2048,
    "use_hybrid_attention":  True,

    # Chronos prediction
    "lookahead_steps":       2,

    # Loss coefficients
    "lambda_balance":        5e-4,
    "lambda_temporal":       1e-3,
    "lambda_lookahead":      0.1,
    "lambda_lookahead_topk": 0.05,
    "lambda_router_anchor":  0.0,
    "fallback_mask_prob":    0.0,

    # Hardware
    "vram_budget_gb":             4.0,
    "pinned_memory_max_fraction": 0.25,
    "storage_format":             "safetensors",

    # Training
    "learning_rate":         5e-4,
    "batch_size":            16,
    "accumulation_steps":    8,
    "epochs":                2,
    "save_interval":         1000,
    "weight_decay":          0.01,
    "grad_clip":             1.0,
    "log_interval":          10,
    "reward_spec":           "toy",
    "max_gen_len":           24,
    "num_generations":       4,
    "temperature":           1.0,
    "beta":                  0.1,
    "alpha":                 0.7,
    "lambda_or":             0.1,
    "save_dir":              "./out",
    "dtype":                 "auto",
    "cpu_threads":           "auto",
    "cpu_budget_percent":    100,
    "num_workers":           "auto",
}


# Order matches ui/tabs/config_tab.py::all_inputs (used by Apply-Preset
# to push values back into the slider widgets in the right slots).
CONFIG_INPUT_ORDER: List[str] = [
    "hidden_size", "num_hidden_layers", "num_experts", "num_experts_per_tok",
    "num_shared_experts", "lookahead_steps", "kv_latent_dim", "sliding_window_size",
    "num_attention_heads", "num_key_value_heads", "rope_dim", "moe_intermediate_size",
    "vocab_size", "dtype", "tie_word_embeddings",
    "lambda_balance", "lambda_temporal", "lambda_lookahead", "lambda_lookahead_topk", "lambda_router_anchor",
    "fallback_mask_prob",
    "vram_budget_gb", "pinned_memory_max_fraction", "use_hybrid_attention", "storage_format",
    "learning_rate", "batch_size", "accumulation_steps", "max_seq_len",
    "epochs", "save_interval", "log_interval", "save_dir",
    "weight_decay", "grad_clip",
    "cpu_threads", "cpu_budget_percent", "num_workers",
    "reward_spec", "max_gen_len", "num_generations", "temperature",
    "beta", "alpha", "lambda_or",
]


# Named presets — keys are user-facing labels (also used in the dropdown).
PRESETS: Dict[str, Dict] = {
    "MiniMind-MoE (default)": dict(MINIMIND_MOE_DEFAULTS),

    # M8c: "Recommended-CN" — sized for the /Users/Fona/Downloads/Hybrid_LLM/Dataset
    # 1.3 GB pretrain corpus. Previously the default 15M-param model
    # (H=256, L=4, E=4, topk=1 → ~6M activated) was far below the size
    # needed for coherent Chinese output (empirically ~100M activated).
    # This preset targets ~120M total / ~60M activated params at vocab=6400.
    "Recommended-CN (≈120M)": {
        **MINIMIND_MOE_DEFAULTS,
        "hidden_size": 512, "num_hidden_layers": 8, "num_experts": 8,
        "num_experts_per_tok": 2, "num_shared_experts": 1,
        "num_attention_heads": 8, "num_key_value_heads": 2,
        "kv_latent_dim": 64, "rope_dim": 32, "sliding_window_size": 2048,
        "max_seq_len": 512, "batch_size": 8, "accumulation_steps": 8,
        "epochs": 1, "save_interval": 500,
        "learning_rate": 3e-4,
    },

    "Tiny (smoke / CI)": {
        **MINIMIND_MOE_DEFAULTS,
        "hidden_size": 128, "num_hidden_layers": 2, "num_experts": 4,
        "num_attention_heads": 4, "num_key_value_heads": 4,
        "kv_latent_dim": 16, "rope_dim": 8, "sliding_window_size": 64,
        "max_seq_len": 64, "batch_size": 2, "accumulation_steps": 1,
        "epochs": 1, "save_interval": 100,
    },

    "Small (256H × 4L × 4E)": {
        **MINIMIND_MOE_DEFAULTS,
        "hidden_size": 256, "num_hidden_layers": 4, "num_experts": 4,
        "max_seq_len": 256, "batch_size": 8,
    },

    "Medium (512H × 8L × 8E)": {
        **MINIMIND_MOE_DEFAULTS,
        "hidden_size": 512, "num_experts": 8,
        "max_seq_len": 1024, "batch_size": 8,
    },

    "Large (1024H × 16L × 16E)": {
        **MINIMIND_MOE_DEFAULTS,
        "hidden_size": 1024, "num_hidden_layers": 16, "num_experts": 16,
        "num_attention_heads": 16, "num_key_value_heads": 4,
        "kv_latent_dim": 128, "rope_dim": 64,
        "max_seq_len": 2048, "batch_size": 4, "accumulation_steps": 16,
        "vram_budget_gb": 16.0,
    },
}


def preset_names() -> List[str]:
    return list(PRESETS.keys())


def get_preset(name: str) -> Dict:
    """Return a deep-ish copy of the named preset (avoids accidental mutation)."""
    cfg = dict(MINIMIND_MOE_DEFAULTS)
    cfg.update(PRESETS.get(name, {}))
    return cfg


def values_in_input_order(cfg: Dict) -> List:
    """Return preset values in the same order as Config tab's all_inputs.

    For keys missing from `cfg`, falls back to MINIMIND_MOE_DEFAULTS. If a
    value is ``None`` (missing in both), it is replaced by a safe-default
    per field type so Gradio's slider/number widgets don't reject it.
    """
    # Hardcoded fallbacks for fields whose MiniMind default is None or absent.
    # Values chosen to match the slider minima where applicable.
    SAFE = {
        "hidden_size": 768, "num_hidden_layers": 8,
        "num_experts": 4, "num_experts_per_tok": 2, "num_shared_experts": 1,
        "lookahead_steps": 2, "kv_latent_dim": 64, "sliding_window_size": 2048,
        "num_attention_heads": 8, "num_key_value_heads": 4, "rope_dim": 32,
        "moe_intermediate_size": 0, "vocab_size": 6400,
        "dtype": "auto", "tie_word_embeddings": True,
        "lambda_balance": 5e-4, "lambda_temporal": 1e-3,
        "lambda_lookahead": 0.1, "lambda_lookahead_topk": 0.05, "lambda_router_anchor": 0.0,
        "fallback_mask_prob": 0.0,
        "vram_budget_gb": 4.0, "pinned_memory_max_fraction": 0.25,
        "use_hybrid_attention": True, "storage_format": "safetensors",
        "learning_rate": 5e-4, "batch_size": 16, "accumulation_steps": 8,
        "max_seq_len": 512, "epochs": 2, "save_interval": 1000,
        "save_dir": "./out",
        "cpu_threads": "auto", "cpu_budget_percent": 100,
        "num_workers": "auto",
        "weight_decay": 0.01, "grad_clip": 1.0, "log_interval": 10,
        "reward_spec": "toy",
        "max_gen_len": 24, "num_generations": 4, "temperature": 1.0,
        "beta": 0.1, "alpha": 0.7, "lambda_or": 0.1,
    }
    out = []
    for k in CONFIG_INPUT_ORDER:
        v = cfg.get(k) if k in cfg else MINIMIND_MOE_DEFAULTS.get(k)
        if v is None:
            v = SAFE.get(k, 0)
        if k == "dtype":
            normalized = str(v).strip().lower()
            v = {
                "fp32": "float32",
                "bf16": "bfloat16",
                "fp16": "float16",
                "half": "float16",
                "full": "float32",
                "": "auto",
            }.get(normalized, normalized)
        out.append(v)
    return out


def save_config(cfg: Dict, path: str) -> str:
    """Write cfg as pretty JSON. Returns the absolute path written."""
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    return abs_path


def load_config(path: str) -> Dict:
    """Read JSON and return the dict. Raises FileNotFoundError if missing."""
    with open(os.path.abspath(path)) as f:
        loaded = json.load(f)
    cfg = dict(MINIMIND_MOE_DEFAULTS)
    cfg.update(loaded)
    return cfg
