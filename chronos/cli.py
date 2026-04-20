"""
chronos/cli.py — Project Chronos unified CLI

Commands:
    chronos train     — pretrain / fine-tune a Chronos model
    chronos eval      — Phase 1 lookahead accuracy validation
    chronos benchmark — Phase 3 end-to-end benchmark
    chronos export    — export expert weights to SSD cluster layout
"""
import sys

import argparse


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


def cmd_export(args):
    from chronos.io.cluster_layout import main
    sys.argv = ['cluster_layout'] + args
    main()


def main():
    parser = argparse.ArgumentParser(
        prog='chronos',
        description='Project Chronos — On-device low-latency lookahead dual-layer MoE inference',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('train',     help='Train a Chronos model',           add_help=False)
    sub.add_parser('eval',      help='Phase 1 lookahead accuracy eval',  add_help=False)
    sub.add_parser('benchmark', help='Phase 3 end-to-end benchmark',     add_help=False)
    sub.add_parser('export',    help='Export expert cluster layout',     add_help=False)

    known, rest = parser.parse_known_args()
    dispatch = {
        'train':     cmd_train,
        'eval':      cmd_eval,
        'benchmark': cmd_benchmark,
        'export':    cmd_export,
    }
    dispatch[known.command](rest)


if __name__ == '__main__':
    main()
