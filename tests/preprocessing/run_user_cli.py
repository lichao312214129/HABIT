"""
Run preprocessing CLI like a local user (no pytest).

Usage:

    python tests/preprocessing/run_user_cli.py
    python tests/preprocessing/run_user_cli.py preprocess-help

cwd is set to the repository root before invoking ``habit``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCENARIOS: dict[str, list[str]] = {
    "preprocess-demo": ["preprocess", "-c", "config/preprocessing/config_preprocessing_demo_elastix.yaml"],
    "preprocess-help": ["preprocess", "--help"],
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _invoke_cli(argv_tail: list[str]) -> None:
    root = _repository_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.argv = ["habit", *argv_tail]
    from habit.cli import cli

    cli()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-run preprocess CLI without pytest.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="preprocess-demo",
        choices=sorted(_SCENARIOS.keys()),
        help="Which demo command to run.",
    )
    args = parser.parse_args()
    _invoke_cli(_SCENARIOS[args.scenario])


if __name__ == "__main__":
    main()
