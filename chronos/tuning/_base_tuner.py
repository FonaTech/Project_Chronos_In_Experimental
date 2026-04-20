"""
chronos/tuning/_base_tuner.py — Vendored subset of Auto_Fine_Tuning's AutoTuner.

Only the base class skeleton (event queue, start/stop/poll, Optuna loop) is kept.
The full probe-trial machinery from Auto_Fine_Tuning is intentionally omitted —
ChronosAutoTuner uses its own lightweight training loop instead.

This file is derived from Auto_Fine_Tuning/core/auto_tuner.py (Apache-2.0).
See THIRD_PARTY_NOTICES.md for full attribution.
"""

import os
import gc
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class SearchSpaceConfig:
    tune_lora_r: bool = True
    lora_r_choices: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    tune_lr: bool = True
    lr_min: float = 5e-5
    lr_max: float = 5e-4
    tune_batch: bool = True
    batch_size_choices: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tune_grad_accum: bool = False
    grad_accum_choices: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tune_warmup: bool = True
    warmup_min: float = 0.01
    warmup_max: float = 0.15
    tune_scheduler: bool = True
    scheduler_choices: List[str] = field(default_factory=lambda: ["cosine", "linear", "constant"])
    tune_lora_alpha: bool = True
    alpha_multiplier_choices: List[int] = field(default_factory=lambda: [1, 2])


@dataclass
class TrialResult:
    trial_number: int
    params: Dict[str, Any]
    train_loss: float
    duration_s: float
    status: str
    error: str = ""


@dataclass
class AutoTuneResult:
    best_params: Dict[str, Any]
    best_loss: float
    trials: List[TrialResult]
    param_importances: Dict[str, float]
    n_completed: int
    n_pruned: int
    n_failed: int
    elapsed_s: float


class AutoTuner:
    """
    Base class: Optuna TPE loop + event queue.
    Subclasses must implement _run_probe_trial().
    """

    def __init__(self):
        self._event_queue: queue.Queue = queue.Queue(maxsize=5000)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.study = None
        self.result: Optional[AutoTuneResult] = None
        self.trials: List[TrialResult] = []
        self.status: str = "idle"

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        model_id: str,
        dataset_path: str,
        train_ratio: float = 0.95,
        prompt_template: str = "alpaca",
        think_mode: str = "keep",
        search_space: Optional[SearchSpaceConfig] = None,
        n_trials: int = 20,
        probe_steps: int = 80,
        output_dir: str = "auto_tune_cache",
        seed: int = 42,
        **kwargs,
    ) -> None:
        if self.is_running():
            raise RuntimeError("Auto-tune already running.")
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("optuna not installed: pip install optuna")

        self._stop_event.clear()
        self.trials.clear()
        self.result = None
        self.status = "loading"

        self._thread = threading.Thread(
            target=self._loop,
            kwargs=dict(
                model_id=model_id,
                dataset_path=dataset_path,
                train_ratio=train_ratio,
                search_space=search_space or SearchSpaceConfig(),
                n_trials=n_trials,
                probe_steps=probe_steps,
                output_dir=output_dir,
                seed=seed,
                **kwargs,
            ),
            daemon=True,
            name="chronos-auto-tune",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.status = "stopped"
        self._put({"type": "status", "status": "stopped"})

    def poll(self) -> List[Dict]:
        events = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def _put(self, event: Dict) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            pass

    def _loop(self, model_id, dataset_path, train_ratio, search_space,
              n_trials, probe_steps, output_dir, seed, **kwargs):
        self._put({"type": "log", "line": "=== Auto-tune started ==="})
        t0 = time.time()
        try:
            sampler = TPESampler(seed=seed, n_startup_trials=max(5, n_trials // 5))
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=probe_steps // 4)
            self.study = optuna.create_study(
                direction="minimize", sampler=sampler, pruner=pruner,
                study_name="chronos_auto_tune",
            )
            self.status = "running"
            self._put({"type": "status", "status": "running"})

            def objective(trial):
                if self._stop_event.is_set():
                    raise optuna.exceptions.OptunaError("user stopped")
                t_start = time.time()
                params = self._sample_params(trial, search_space)
                self._put({"type": "log",
                           "line": f"[Trial {trial.number+1}/{n_trials}] {_fmt(params)}"})
                try:
                    loss = self._run_probe_trial(
                        trial, model_id, dataset_path, train_ratio,
                        probe_steps, output_dir, seed, params, **kwargs,
                    )
                    status = "complete"
                except optuna.exceptions.TrialPruned:
                    loss = float("inf")
                    status = "pruned"
                except Exception as e:
                    self._put({"type": "log", "line": f"[Trial {trial.number}] failed: {e}"})
                    loss = float("inf")
                    status = "failed"

                self.trials.append(TrialResult(
                    trial_number=trial.number, params=params,
                    train_loss=loss, duration_s=round(time.time() - t_start, 1),
                    status=status,
                ))
                best = min((t.train_loss for t in self.trials if t.status == "complete"),
                           default=float("inf"))
                self._put({"type": "log",
                           "line": f"  → loss={loss:.4f}  best={best:.4f}  {status}"})
                if status == "pruned":
                    raise optuna.exceptions.TrialPruned()
                return loss

            def _stop_cb(study, _trial):
                if self._stop_event.is_set():
                    study.stop()

            self.study.optimize(objective, n_trials=n_trials,
                                callbacks=[_stop_cb], gc_after_trial=True)

            try:
                importances = optuna.importance.get_param_importances(self.study)
            except Exception:
                importances = {}

            completed = [t for t in self.trials if t.status == "complete"]
            best_params = self.study.best_params if completed else {}
            best_loss = self.study.best_value if completed else float("inf")

            self.result = AutoTuneResult(
                best_params=best_params, best_loss=best_loss,
                trials=self.trials, param_importances=importances,
                n_completed=len(completed),
                n_pruned=len([t for t in self.trials if t.status == "pruned"]),
                n_failed=len([t for t in self.trials if t.status == "failed"]),
                elapsed_s=time.time() - t0,
            )
            self.status = "finished"
            self._put({"type": "finished", "best_params": best_params, "best_loss": best_loss})
            self._put({"type": "log",
                       "line": (f"=== Done ===  best_loss={best_loss:.4f}  "
                                f"completed={self.result.n_completed}  "
                                f"elapsed={self.result.elapsed_s:.0f}s")})
        except Exception as e:
            import traceback
            self.status = "error"
            self._put({"type": "status", "status": "error"})
            self._put({"type": "log", "line": f"[ERROR] {e}\n{traceback.format_exc()}"})

    def _sample_params(self, trial, ss: SearchSpaceConfig) -> dict:
        params = {}
        params["lora_r"] = (trial.suggest_categorical("lora_r", ss.lora_r_choices)
                            if ss.tune_lora_r else 16)
        mult = (trial.suggest_categorical("alpha_mult", ss.alpha_multiplier_choices)
                if ss.tune_lora_alpha else 2)
        params["lora_alpha"] = params["lora_r"] * mult
        params["learning_rate"] = (
            trial.suggest_float("learning_rate", ss.lr_min, ss.lr_max, log=True)
            if ss.tune_lr else 2e-4)
        params["per_device_train_batch_size"] = (
            trial.suggest_categorical("batch_size", ss.batch_size_choices)
            if ss.tune_batch else 4)
        params["gradient_accumulation_steps"] = (
            trial.suggest_categorical("grad_accum", ss.grad_accum_choices)
            if ss.tune_grad_accum else 4)
        params["warmup_ratio"] = (
            trial.suggest_float("warmup_ratio", ss.warmup_min, ss.warmup_max)
            if ss.tune_warmup else 0.05)
        params["lr_scheduler_type"] = (
            trial.suggest_categorical("scheduler", ss.scheduler_choices)
            if ss.tune_scheduler else "cosine")
        return params

    def _run_probe_trial(self, trial, model_id, dataset_path, train_ratio,
                         probe_steps, output_dir, seed, params, **kwargs) -> float:
        raise NotImplementedError

    def get_best_config_patch(self) -> Dict[str, Any]:
        if not self.result or not self.result.best_params:
            return {}
        return dict(self.result.best_params)


def _fmt(p: dict) -> str:
    return ", ".join(
        f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
        for k, v in p.items()
    )


def _gc_collect():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
