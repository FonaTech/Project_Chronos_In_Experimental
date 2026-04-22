"""
ui/tabs/train_tab.py — Full training loop: Pretrain / SFT / RL / ORPO
"""
import os
import time
import threading
import queue

import gradio as gr

import chronos.deps  # auto-bootstrap minimind on sys.path
from ui.i18n import t, register_translatable, get_current_lang


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
    "rl": {
        "zh-Hans": "请比较 Chronos 与传统 MoE offload 在吞吐、首 token 延迟和缓存未命中行为上的权衡。",
        "zh-Hant": "請比較 Chronos 與傳統 MoE offload 在吞吐、首 token 延遲和快取未命中行為上的權衡。",
        "en": "Compare Chronos with conventional MoE offload in terms of throughput, first-token latency, and cache-miss behavior.",
        "ja": "Chronos と従来の MoE offload を、スループット、初回トークン遅延、キャッシュミス時の挙動で比較してください。",
    },
    "orpo": {
        "zh-Hans": "请写一段更适合 README 首页的英文介绍，突出 Predictive Prefill、Soft Gating 和 6-stage training pipeline。",
        "zh-Hant": "請寫一段更適合 README 首頁的英文介紹，突出 Predictive Prefill、Soft Gating 和 6-stage training pipeline。",
        "en": "Write a README-ready intro that highlights predictive prefill, soft gating, and the 6-stage training pipeline.",
        "ja": "predictive prefill、soft gating、6-stage training pipeline を強調した README 向けの紹介文を書いてください。",
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

    def _run(self, cfg: dict, mode: str):
        t_start = time.monotonic()
        try:
            import torch
            import torch.optim as optim
            from torch.utils.data import DataLoader

            from chronos.model.config import ChronosConfig
            from chronos.model.model_chronos import ChronosForCausalLM
            from chronos.model.temporal_loss import total_loss
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
            # Output checkpoint path differs per stage so SFT doesn't
            # clobber the pretrain weights, DPO doesn't clobber SFT, etc.
            STAGE_PREFIX = {
                "pretrain": "chronos", "sft": "sft",
                "rl": "rl", "orpo": "orpo",
            }
            out_prefix = STAGE_PREFIX.get(mode, "chronos")
            ckp_path = os.path.join(save_dir, f"{out_prefix}_{model_cfg.hidden_size}_moe.pth")

            # Init-weight resolution:
            #   - pretrain: optionally resume from its own checkpoint if present
            #   - sft/rl/orpo: REQUIRE an upstream weight (init_weight from UI,
            #     or fall back to the previous-stage default in save_dir).
            init_weight = (cfg.get("init_weight") or "").strip()
            if mode == "pretrain":
                resume_path = init_weight or ckp_path
                if os.path.exists(resume_path):
                    sniffed = _sniff_checkpoint(resume_path)
                    mismatches = []
                    for k in [
                        "hidden_size", "num_hidden_layers", "num_experts",
                        "moe_intermediate_size", "vocab_size", "lookahead_steps",
                    ]:
                        if k in sniffed and k in model_cfg_kwargs and int(sniffed[k]) != int(model_cfg_kwargs[k]):
                            mismatches.append(f"{k}: ckpt={sniffed[k]} != ui={model_cfg_kwargs[k]}")

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
                STAGE_DEFAULT_INIT = {
                    "sft":  f"chronos_{model_cfg.hidden_size}_moe.pth",
                    "rl":   f"sft_{model_cfg.hidden_size}_moe.pth",
                    "orpo": f"sft_{model_cfg.hidden_size}_moe.pth",
                }
                load_path = init_weight or os.path.join(save_dir, STAGE_DEFAULT_INIT[mode])
                if not os.path.exists(load_path):
                    raise FileNotFoundError(
                        f"[{mode.upper()}] requires an upstream checkpoint to initialize from. "
                        f"Tried: {load_path}\n"
                        f"Set 'init_weight' in the Train tab to point at a valid .pth, "
                        f"or run the prior stage first (Pretrain → SFT → DPO/ORPO/GRPO)."
                    )

                sniffed = _sniff_checkpoint(load_path)
                mismatch_hints = []
                for k in [
                    "hidden_size", "num_hidden_layers", "num_experts",
                    "moe_intermediate_size", "vocab_size", "lookahead_steps",
                ]:
                    if k in sniffed and k in model_cfg_kwargs and int(sniffed[k]) != int(model_cfg_kwargs[k]):
                        mismatch_hints.append(f"{k}: ckpt={sniffed[k]} != ui={model_cfg_kwargs[k]}")
                if mismatch_hints:
                    raise RuntimeError(
                        f"[{mode.upper()}] init checkpoint topology does not match current UI config:\n  "
                        + "\n  ".join(mismatch_hints)
                        + "\nLoad the matching preset/config first, or switch init_weight to the correct checkpoint."
                    )

                weights = torch.load(load_path, map_location="cpu")
                model.load_state_dict(weights, strict=False)
                self._put(f"[{mode.upper()}] Initialized from {load_path}")

            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self._put(f"Device: {device}")
            model = model.to(device)
            from chronos.trainer.optim_utils import build_optimizer, get_lr, apply_lr
            base_lr = float(cfg.get("learning_rate", 5e-4))
            weight_decay = float(cfg.get("weight_decay", 0.01))
            optimizer = build_optimizer(model, lr=base_lr, weight_decay=weight_decay)

            data_path = cfg.get("data_path", "")
            max_seq_len = cfg.get("max_seq_len", 512)
            batch_size = cfg.get("batch_size", 4)
            accum = max(1, int(cfg.get("accumulation_steps", 8)))
            epochs = cfg.get("epochs", 1)
            save_interval = cfg.get("save_interval", 500)
            log_interval = max(1, cfg.get("log_interval", 10))
            val_ratio = float(cfg.get("val_ratio", 0.05))
            max_steps = int(cfg.get("max_steps", 0)) or None  # 0 => no cap

            if not data_path or not os.path.exists(data_path):
                self._put("No dataset found — running synthetic smoke test (50 steps)")
                loader = self._synthetic_loader(model_cfg.vocab_size, max_seq_len, batch_size, n=50)
                val_loader = None
            else:
                tokenizer = self._load_tokenizer()
                # Streaming JSONL — O(N · 8B) RSS regardless of corpus size.
                # Avoids the HuggingFace `load_dataset('json', ...)` Arrow
                # mmap path which kept blowing up to 100s of GB resident
                # under continuous random access on macOS.
                from chronos.data.flexible_dataset import (
                    FlexibleDataset, StreamingSFTDataset,
                )
                if mode == "pretrain":
                    full_ds = FlexibleDataset(data_path, tokenizer, max_length=max_seq_len)
                else:
                    full_ds = StreamingSFTDataset(data_path, tokenizer, max_length=max_seq_len)
                # Deterministic val split: every 1/val_ratio-th record goes to val
                # (idx-modulo split → reproducible without rescanning the file).
                if val_ratio > 0 and len(full_ds) >= 20:
                    stride = max(2, int(round(1.0 / val_ratio)))
                    val_idx   = [i for i in range(len(full_ds)) if i % stride == 0]
                    train_idx = [i for i in range(len(full_ds)) if i % stride != 0]
                    from torch.utils.data import Subset
                    train_ds = Subset(full_ds, train_idx)
                    val_ds   = Subset(full_ds, val_idx)
                    self._put(f"Split: train={len(train_ds)}  val={len(val_ds)}")
                else:
                    train_ds, val_ds = full_ds, None
                loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                    num_workers=0, pin_memory=False)
                val_loader = (
                    DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=False)
                    if val_ds is not None else None
                )

            model.train()
            global_step = 0
            step_t = time.monotonic()
            best_val = float("inf")
            best_path = ckp_path.replace(".pth", ".best.pth")
            # Steps remaining for ETA / progress bar — respect max_steps cap.
            try:
                planned = max(1, len(loader) * epochs)
            except Exception:
                planned = 0
            self.total_steps = min(planned, max_steps) if (max_steps and planned) else (planned or (max_steps or 0))

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


def _make_loss_chart(metrics: list[dict], total_steps: int = 0, t_start: float = 0.0):
    """Build matplotlib figure: loss curves + throughput + progress/ETA."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
            ax.plot(train_steps, [m["ce"] for m in train_metrics],       color="#0f3460", lw=1.2, label="train ce")
            ax.plot(train_steps, [m["aux"] for m in train_metrics],      color="#533483", lw=1.0, label="aux")
            ax.plot(train_steps, [m["temporal"] for m in train_metrics], color="#e2b714", lw=0.9, label="temporal", ls="--")
        if val_metrics:
            ax.plot(val_steps, [m["ce"] for m in val_metrics], color="#39d98a",
                    lw=2.0, marker="o", markersize=4, label="val ce")
        ax.set_title("Loss Curves", color="white", fontsize=11)
        ax.set_xlabel("Step", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(fontsize=8, facecolor="#16213e", labelcolor="white")

        # ── Middle: throughput (only train rows have tps)
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        tps_steps = [m["step"] for m in train_metrics if m.get("tps", 0) > 0]
        tps_vals  = [m["tps"]  for m in train_metrics if m.get("tps", 0) > 0]
        if tps_steps:
            ax2.plot(tps_steps, tps_vals, color="#00b4d8", lw=1.5)
            ax2.fill_between(tps_steps, tps_vals, alpha=0.2, color="#00b4d8")
        ax2.set_title("Throughput (steps/s)", color="white", fontsize=11)
        ax2.set_xlabel("Step", color="#aaa")
        ax2.tick_params(colors="#aaa")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#333")

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
        ax3.text(50, 0.10, f"~{rate:.2f} steps/s avg",
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
        with gr.Row():
            data_path = gr.Textbox(
                label=t("config.data_path"),
                placeholder="./tests/fixtures/tiny_pretrain.jsonl",
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
                ["pretrain", "sft", "rl", "orpo"],
                value="pretrain", label=t("train.mode")
            )
            register_translatable(mode, "train.mode")
            status_box = gr.Textbox(
                value="idle", label=t("train.status"),
                interactive=False, scale=1,
            )
            register_translatable(status_box, "train.status")

        # ── Init-weight: required for SFT/RL/ORPO, optional for pretrain ──
        with gr.Row():
            init_weight = gr.Textbox(
                label=t("train.init_weight"),
                placeholder="./out/chronos_768_moe.pth",
                scale=4,
            )
            register_translatable(init_weight, "train.init_weight")

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
            loss_box     = gr.Number(label=t("train.loss"),     value=0.0, interactive=False, precision=4)
            ce_box       = gr.Number(label=t("train.ce_loss"),  value=0.0, interactive=False, precision=4)
            aux_box      = gr.Number(label=t("train.aux_loss"), value=0.0, interactive=False, precision=4)
            tps_box      = gr.Number(label=t("train.tps"),      value=0.0, interactive=False, precision=2)
            register_translatable(step_box, "train.step")
            register_translatable(loss_box, "train.loss")
            register_translatable(ce_box,   "train.ce_loss")
            register_translatable(aux_box,  "train.aux_loss")
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

        def sync_sample_prompt(train_mode, current_prompt):
            current = (current_prompt or "").strip()
            if (not current) or current in DEFAULT_SAMPLE_PROMPT_VALUES:
                return gr.update(
                    value=_stage_sample_prompt(train_mode),
                    placeholder=_stage_sample_prompt(train_mode),
                )
            return gr.update(placeholder=_stage_sample_prompt(train_mode))

        def start_training(cfg, train_mode, dpath, iw, ms, vr, wd, sp):
            if _session.is_running():
                return "already running", 0, 0.0, 0.0, 0.0, 0.0, "Already running.\n", None, ""
            cfg = dict(cfg) if cfg else {}
            cfg["data_path"] = dpath or cfg.get("data_path", "")
            cfg["init_weight"] = iw or ""
            cfg["max_steps"] = int(ms or 0)
            cfg["val_ratio"] = float(vr or 0.0)
            cfg["weight_decay"] = float(wd or 0.0)
            cfg["sample_prompt"] = sp or ""
            _session.start(cfg, train_mode)
            return "running", 0, 0.0, 0.0, 0.0, 0.0, "Starting...\n", None, ""

        def stop_training():
            _session.stop()
            return "stopped"

        def clear_log():
            return ""

        def poll(current_log: str):
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
            fig = _make_loss_chart(metrics, _session.total_steps, _session.t_start)

            # Latest scalar values from last metric entry
            if metrics:
                last = metrics[-1]
                ce_v = last.get("ce", 0.0)
                aux_v = last.get("aux", 0.0)
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
            fn=sync_sample_prompt,
            inputs=[mode, sample_prompt_in],
            outputs=[sample_prompt_in],
        )

        start_btn.click(
            fn=start_training,
            inputs=[config_state, mode, data_path, init_weight,
                    max_steps_in, val_ratio_in, weight_decay_in,
                    sample_prompt_in],
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box,
                     log_box, chart, sample_box],
        )
        stop_btn.click(fn=stop_training, outputs=[status_box])
        clear_btn.click(fn=clear_log, outputs=[log_box])

        timer = gr.Timer(value=2.0)
        timer.tick(
            fn=poll,
            inputs=[log_box],
            outputs=[status_box, step_box, loss_box, ce_box, aux_box, tps_box,
                     log_box, chart, sample_box],
        )

    return tab
