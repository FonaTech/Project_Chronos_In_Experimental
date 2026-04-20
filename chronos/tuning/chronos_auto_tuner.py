"""
chronos/tuning/chronos_auto_tuner.py — ChronosAutoTuner: λ1/λ2/lookahead Optuna search.

Extends the vendored AutoTuner base with Chronos-specific hyperparameters.
Uses a lightweight training loop (no external unsloth/trl dependency).
"""
import os
import time
import dataclasses
from dataclasses import dataclass, field
from typing import List

import chronos.deps  # ensure minimind on sys.path

from chronos.tuning._base_tuner import (
    AutoTuner, SearchSpaceConfig, AutoTuneResult, _gc_collect,
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class ChronosSearchSpaceConfig(SearchSpaceConfig):
    tune_lambda_balance: bool = True
    lambda_balance_min: float = 1e-4
    lambda_balance_max: float = 1e-2

    tune_lambda_temporal: bool = True
    lambda_temporal_min: float = 1e-4
    lambda_temporal_max: float = 1e-2

    tune_lookahead_steps: bool = False
    lookahead_steps_choices: List[int] = field(default_factory=lambda: [1, 2])

    tune_lr: bool = True

    # Extended (UI fix 6): structural knobs that can also be searched.
    tune_hidden_size: bool = False
    hidden_size_choices: List[int] = field(default_factory=lambda: [128, 256, 384, 512])

    tune_num_experts: bool = False
    num_experts_choices: List[int] = field(default_factory=lambda: [2, 4, 6, 8])

    tune_num_shared_experts: bool = False
    num_shared_experts_choices: List[int] = field(default_factory=lambda: [0, 1, 2])

    tune_kv_latent_dim: bool = False
    kv_latent_dim_choices: List[int] = field(default_factory=lambda: [32, 64, 96, 128])

    tune_lambda_lookahead: bool = False
    lambda_lookahead_min: float = 1e-3
    lambda_lookahead_max: float = 1.0


class ChronosAutoTuner(AutoTuner):
    """
    Extends AutoTuner to search over Chronos-specific hyperparameters
    (λ1, λ2, lookahead_steps) using a lightweight native training loop
    (no trl/unsloth dependency required).
    """

    def _sample_params(self, trial, ss) -> dict:
        params = {}

        if ss.tune_lr:
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", ss.lr_min, ss.lr_max, log=True
            )
        else:
            params["learning_rate"] = 2e-4

        if isinstance(ss, ChronosSearchSpaceConfig):
            if ss.tune_lambda_balance:
                params["lambda_balance"] = trial.suggest_float(
                    "lambda_balance", ss.lambda_balance_min, ss.lambda_balance_max, log=True
                )
            else:
                params["lambda_balance"] = 5e-4

            if ss.tune_lambda_temporal:
                params["lambda_temporal"] = trial.suggest_float(
                    "lambda_temporal", ss.lambda_temporal_min, ss.lambda_temporal_max, log=True
                )
            else:
                params["lambda_temporal"] = 1e-3

            if ss.tune_lookahead_steps:
                params["lookahead_steps"] = trial.suggest_categorical(
                    "lookahead_steps", ss.lookahead_steps_choices
                )
            else:
                params["lookahead_steps"] = 2

            # Extended (UI fix 6) — categorical structural knobs
            if ss.tune_hidden_size:
                params["hidden_size"] = trial.suggest_categorical(
                    "hidden_size", ss.hidden_size_choices)
            if ss.tune_num_experts:
                params["num_experts"] = trial.suggest_categorical(
                    "num_experts", ss.num_experts_choices)
            if ss.tune_num_shared_experts:
                params["num_shared_experts"] = trial.suggest_categorical(
                    "num_shared_experts", ss.num_shared_experts_choices)
            if ss.tune_kv_latent_dim:
                params["kv_latent_dim"] = trial.suggest_categorical(
                    "kv_latent_dim", ss.kv_latent_dim_choices)
            if ss.tune_lambda_lookahead:
                params["lambda_lookahead"] = trial.suggest_float(
                    "lambda_lookahead", ss.lambda_lookahead_min, ss.lambda_lookahead_max, log=True)
        else:
            params["lambda_balance"] = 5e-4
            params["lambda_temporal"] = 1e-3
            params["lookahead_steps"] = 2

        return params

    def _run_probe_trial(
        self, trial, model_id, dataset_path, train_ratio,
        probe_steps, output_dir, seed, params, **kwargs
    ) -> float:
        import torch
        from chronos.model.config import ChronosConfig
        from chronos.model.model_chronos import ChronosForCausalLM
        from chronos.model.temporal_loss import total_loss
        from chronos.model.moe_chronos import ChronosMOEFeedForward

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        lr = params.get("learning_rate", 2e-4)
        lb = params.get("lambda_balance", 5e-4)
        lt = params.get("lambda_temporal", 1e-3)
        la = params.get("lookahead_steps", 2)
        hs = int(params.get("hidden_size", 256))
        ne = int(params.get("num_experts", 4))
        ns = int(params.get("num_shared_experts", 1))
        kv = int(params.get("kv_latent_dim", 64))

        cfg = ChronosConfig(
            hidden_size=hs,
            num_hidden_layers=4,
            num_experts=ne,
            num_shared_experts=ns,
            kv_latent_dim=kv,
            lookahead_steps=la,
            lambda_balance=lb,
            lambda_temporal=lt,
            use_hybrid_attention=True,
            use_moe=True,
        )
        model = ChronosForCausalLM(cfg).to(device)
        if os.path.exists(model_id):
            try:
                w = torch.load(model_id, map_location="cpu")
                model.load_state_dict(w, strict=False)
            except Exception:
                pass

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        vocab_size = cfg.vocab_size
        seq_len = 64
        batch = 2
        model.train()
        losses = []
        for step in range(probe_steps):
            ids = torch.randint(0, vocab_size, (batch, seq_len), device=device)
            out, _ = model(ids, labels=ids)
            moe_layers = [l.mlp for l in model.model.layers
                          if isinstance(l.mlp, ChronosMOEFeedForward)]
            if moe_layers and moe_layers[0].last_router_probs is not None:
                probs = torch.stack(
                    [l.last_router_probs for l in moe_layers], dim=2
                ).mean(dim=2)
                loss = total_loss(out.loss, out.aux_loss, probs, lb, lt)
            else:
                loss = out.loss + out.aux_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            trial.report(loss.item(), step=step)
            if trial.should_prune():
                del model
                _gc_collect()
                raise optuna.exceptions.TrialPruned()

        final_loss = sum(losses[-max(1, probe_steps // 5):]) / max(1, probe_steps // 5)
        del model
        _gc_collect()
        return final_loss

    def get_best_chronos_config_patch(self) -> dict:
        patch = self.get_best_config_patch()
        if self.result and self.result.best_params:
            p = self.result.best_params
            for key in ("lambda_balance", "lambda_temporal", "lookahead_steps"):
                if key in p:
                    patch[key] = p[key]
        return patch
