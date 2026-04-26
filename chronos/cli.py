"""
chronos/cli.py — Project Chronos unified CLI

Commands:
    chronos train     — pretrain / fine-tune a Chronos model
    chronos eval      — Phase 1 lookahead accuracy validation
    chronos benchmark — Phase 3 end-to-end benchmark
    chronos diagnose  — checkpoint topology + offload/predictive-routing report
    chronos export    — export full checkpoints or expert SSD cluster layout
    chronos mac       — Apple Silicon backend diagnostics
"""
import sys
import os

import argparse
import json

if __package__ in {None, ""}:
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_pkg_dir)
    # Running `python chronos/cli.py` puts `chronos/` itself on sys.path.
    # That makes the local `chronos/mlx` package shadow Apple's top-level
    # `mlx` package. Keep only the repo root.
    sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _pkg_dir]
    sys.path.insert(0, _repo_root)


def cmd_train(args):
    import train_chronos
    sys.argv = ['train_chronos'] + args
    train_chronos.main()


def cmd_eval(args):
    from chronos.eval.io_profiler import main
    sys.argv = ['io_profiler'] + args
    main()


def cmd_benchmark(args):
    from chronos.eval.benchmark import main
    sys.argv = ['benchmark'] + args
    main()


def cmd_diagnose(args):
    from diagnose_checkpoint import main
    sys.argv = ['diagnose_checkpoint'] + args
    main()


def cmd_export(args):
    # Backward compatibility: the original `chronos export` command generated
    # an expert cluster layout and required --data_path. New full-checkpoint
    # deployment export does not.
    if "--data_path" in args:
        from chronos.io.cluster_layout import main
        sys.argv = ['cluster_layout'] + args
        main()
    else:
        from chronos.export import main
        main(args)


def cmd_mac(args):
    from chronos.backend.mac_diagnostics import mac_backend_diagnostics

    configure = "--no-configure-threads" not in args
    print(json.dumps(mac_backend_diagnostics(configure_threads=configure), indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        prog='chronos',
        description='Project Chronos — On-device low-latency lookahead dual-layer MoE inference',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('train',     help='Train a Chronos model',           add_help=False)
    sub.add_parser('eval',      help='Phase 1 lookahead accuracy eval',  add_help=False)
    sub.add_parser('benchmark', help='Phase 3 end-to-end benchmark',     add_help=False)
    sub.add_parser('diagnose',  help='Checkpoint/offload diagnostic',    add_help=False)
    sub.add_parser('export',    help='Export checkpoint or expert cluster layout', add_help=False)
    sub.add_parser('mac',       help='Apple Silicon backend diagnostics', add_help=False)

    known, rest = parser.parse_known_args()
    dispatch = {
        'train':     cmd_train,
        'eval':      cmd_eval,
        'benchmark': cmd_benchmark,
        'diagnose':  cmd_diagnose,
        'export':    cmd_export,
        'mac':       cmd_mac,
    }
    dispatch[known.command](rest)


if __name__ == '__main__':
    main()
