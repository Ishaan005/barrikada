from __future__ import annotations

import argparse

from core.agent import DEFAULT_MODEL_NAME, evaluate, interactive


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/agent.py",
        description="Compatibility wrapper for Barrikade agent workflows.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="chat",
        choices=["chat", "eval"],
        help="chat for interactive mode, eval for dataset evaluation.",
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="datasets/barrikade_test.csv",
        help="CSV path used by eval mode.",
    )
    parser.add_argument(
        "max_samples",
        nargs="?",
        type=int,
        default=None,
        help="Optional row limit for eval mode.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL_NAME,
        help="Local model path for agent responses.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.mode == "eval":
        evaluate(csv_path=args.csv, max_samples=args.max_samples, model_name=args.model)
        return 0

    interactive(model_name=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
