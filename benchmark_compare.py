"""
benchmark_compare.py — Background benchmark: minimind (pure RAM) vs Chronos (SSD+DRAM)

Runs in background, writes results to benchmark_results.json
Usage: python benchmark_compare.py
"""
import os, sys, time, json, gc, threading

# Ensure chronos package is importable when run as a script
_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import chronos.deps  # auto-bootstrap minimind on sys.path
import torch
import psutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(_SCRIPT_DIR, 'benchmark_results.json')
LOG_FILE     = os.path.join(_SCRIPT_DIR, 'benchmark_log.txt')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def ram_used_gb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024**3)

def measure_tokens_per_sec(model, vocab_size, device, n_tokens=50, seq_len=32):
    model.eval()
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    # Warmup
    with torch.no_grad():
        if hasattr(model, 'forward') and 'labels' in model.forward.__code__.co_varnames:
            try:
                out = model(x)
                if hasattr(out, 'past_key_values'):
                    past_kv = out.past_key_values
                else:
                    past_kv = None
            except:
                past_kv = None
        else:
            past_kv = None

    t0 = time.monotonic()
    with torch.no_grad():
        out = model(x, use_cache=True)
        if hasattr(out, '__iter__'):
            out, _ = out if isinstance(out, tuple) else (out, None)
        past_kv = getattr(out, 'past_key_values', None)
        for _ in range(n_tokens):
            xn = torch.randint(0, vocab_size, (1, 1), device=device)
            try:
                result = model(xn, past_key_values=past_kv, use_cache=True)
                if isinstance(result, tuple):
                    out2, _ = result
                else:
                    out2 = result
                past_kv = getattr(out2, 'past_key_values', None)
            except Exception as e:
                log(f'  decode step error: {e}')
                break
    elapsed = time.monotonic() - t0
    return n_tokens / max(elapsed, 1e-6)

# ── Benchmark 1: minimind pure RAM ────────────────────────────────

def bench_minimind():
    log('=== Benchmark 1: MiniMind (pure RAM) ===')
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    cfg = MiniMindConfig(
        hidden_size=512, num_hidden_layers=8, use_moe=True,
        num_experts=4, num_experts_per_tok=1,
    )
    ram_before = ram_used_gb()
    model = MiniMindForCausalLM(cfg)
    model.eval()
    ram_after = ram_used_gb()
    ram_model_gb = ram_after - ram_before
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log(f'  params={params_m:.1f}M  RAM delta={ram_model_gb:.3f}GB')

    # Train a few steps
    import torch.optim as optim
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    train_losses = []
    for step in range(5):
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        out = model(x, labels=x)
        loss = out.loss + out.aux_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        log(f'  step {step+1}/5 loss={loss.item():.4f}')

    ram_train = ram_used_gb()
    model.eval()
    tps = measure_tokens_per_sec(model, cfg.vocab_size, 'cpu', n_tokens=30)
    ram_infer = ram_used_gb()
    log(f'  TPS={tps:.1f}  RAM_train={ram_train:.3f}GB  RAM_infer={ram_infer:.3f}GB')

    result = {
        'name': 'MiniMind (pure RAM)',
        'params_m': round(params_m, 1),
        'ram_model_gb': round(ram_model_gb, 3),
        'ram_train_gb': round(ram_train, 3),
        'ram_infer_gb': round(ram_infer, 3),
        'tokens_per_sec': round(tps, 1),
        'train_losses': [round(l, 4) for l in train_losses],
    }
    del model, optimizer
    gc.collect()
    return result

# ── Benchmark 2: Chronos SSD+DRAM ────────────────────────────────

def bench_chronos():
    log('=== Benchmark 2: Chronos (SSD+DRAM hybrid) ===')
    from chronos.model.config import ChronosConfig
    from chronos.model.model_chronos import ChronosForCausalLM
    from chronos.io.expert_store import ExpertStore
    from chronos.runtime.cache_manager import CacheManager

    cfg = ChronosConfig(
        hidden_size=512, num_hidden_layers=8, use_moe=True,
        num_experts=4, num_experts_per_tok=1,
        use_hybrid_attention=True,
        kv_latent_dim=64, rope_dim=32, sliding_window_size=256,
        vram_budget_gb=0.5,  # simulate constrained VRAM
        num_shared_experts=1,
    )
    ram_before = ram_used_gb()
    model = ChronosForCausalLM(cfg)
    model.eval()
    ram_after = ram_used_gb()
    ram_model_gb = ram_after - ram_before
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    log(f'  params={params_m:.1f}M  RAM delta={ram_model_gb:.3f}GB')

    # Train a few steps with temporal loss
    import torch.optim as optim
    from chronos.model.temporal_loss import total_loss
    from chronos.model.moe_chronos import ChronosMOEFeedForward
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    train_losses = []
    for step in range(5):
        x = torch.randint(0, cfg.vocab_size, (2, 64))
        out, lp = model(x, labels=x)
        moe_layers = [l.mlp for l in model.model.layers if isinstance(l.mlp, ChronosMOEFeedForward)]
        probs = torch.stack([l.last_router_probs for l in moe_layers], dim=2).mean(dim=2)
        loss = total_loss(out.loss, out.aux_loss, probs, cfg.lambda_balance, cfg.lambda_temporal)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        log(f'  step {step+1}/5 loss={loss.item():.4f}')

    ram_train = ram_used_gb()

    # Setup SSD offload + cache manager
    ssd_dir = '/tmp/chronos_bench_ssd'
    store = ExpertStore(model, cfg, ssd_dir=ssd_dir)
    store.offload_all_to_ssd()
    mgr = CacheManager(model, cfg, ssd_dir=ssd_dir)
    mgr.start()
    mgr.warm_up()

    # Measure TPS with SSD+DRAM mode
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    t0 = time.monotonic()
    with torch.no_grad():
        masks = mgr.availability_masks_all_layers()
        out2, lp2 = model(x, use_cache=True, available_expert_masks=masks)
        past_kv = out2.past_key_values
        for step in range(30):
            xn = torch.randint(0, cfg.vocab_size, (1, 1))
            masks = mgr.availability_masks_all_layers()
            out3, lp3 = model(xn, past_key_values=past_kv, use_cache=True,
                              available_expert_masks=masks)
            past_kv = out3.past_key_values
            from chronos.model.moe_chronos import ChronosMOEFeedForward as CMOE
            cur_ids = [l.mlp.last_router_probs[:,-1,:].argmax(-1).item()
                       for l in model.model.layers if isinstance(l.mlp, CMOE)]
            mgr.step(lp3, cur_ids)
    elapsed = time.monotonic() - t0
    tps = 30 / max(elapsed, 1e-6)
    ram_infer = ram_used_gb()
    cache_stats = mgr.stats()
    mgr.stop()

    log(f'  TPS={tps:.1f}  RAM_train={ram_train:.3f}GB  RAM_infer={ram_infer:.3f}GB')
    log(f'  cache_stats={cache_stats}')

    result = {
        'name': 'Chronos (SSD+DRAM hybrid)',
        'params_m': round(params_m, 1),
        'ram_model_gb': round(ram_model_gb, 3),
        'ram_train_gb': round(ram_train, 3),
        'ram_infer_gb': round(ram_infer, 3),
        'tokens_per_sec': round(tps, 1),
        'train_losses': [round(l, 4) for l in train_losses],
        'cache_stats': cache_stats,
        'kv_cache_type': 'MLA(latent)+SlidingWindow',
    }
    del model, optimizer
    gc.collect()
    return result

# ── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    open(LOG_FILE, 'w').close()
    log('Starting benchmark comparison...')

    results = {}
    try:
        results['minimind'] = bench_minimind()
    except Exception as e:
        log(f'minimind bench failed: {e}')
        import traceback; log(traceback.format_exc())

    try:
        results['chronos'] = bench_chronos()
    except Exception as e:
        log(f'chronos bench failed: {e}')
        import traceback; log(traceback.format_exc())

    # Summary
    log('\n=== COMPARISON SUMMARY ===')
    for k, r in results.items():
        log(f"{r['name']}:")
        log(f"  tokens/s     : {r.get('tokens_per_sec', 'N/A')}")
        log(f"  RAM (train)  : {r.get('ram_train_gb', 'N/A')} GB")
        log(f"  RAM (infer)  : {r.get('ram_infer_gb', 'N/A')} GB")
        if 'kv_cache_type' in r:
            log(f"  KV cache     : {r['kv_cache_type']}")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    log(f'Results saved to {RESULTS_FILE}')
