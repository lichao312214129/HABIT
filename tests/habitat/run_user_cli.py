"""
Run habitat-related CLI commands like a local user (no pytest).

Usage:

    python tests/habitat/run_user_cli.py --help
    python tests/habitat/run_user_cli.py get-habitat-two-step

cwd is set to the repository root before invoking ``habit``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCENARIOS: dict[str, list[str]] = {
    "get-habitat-two-step": [
        "get-habitat",
        "-c",
        "config/habitat/config_habitat_two_step.yaml",
    ],
    "get-habitat-one-step": [
        "get-habitat",
        "-c",
        "config/habitat/config_habitat_one_step.yaml",
    ],
    "get-habitat-direct-pooling": [
        "get-habitat",
        "-c",
        "config/habitat/config_habitat_direct_pooling.yaml",
    ],
    "get-habitat-two-step-debug": [
        "get-habitat",
        "-c",
        "config/habitat/config_habitat_two_step.yaml",
        "--debug",
    ],
    "extract-demo": ["extract", "-c", "config/feature_extraction/config_extract_features_demo.yaml"],
    "get-habitat-help": ["get-habitat", "--help"],
    "extract-help": ["extract", "--help"],
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _invoke_cli(argv_tail: list[str]) -> None:
    root = _repository_root()
    os.chdir(root)
    sys.argv = ["habit", *argv_tail]
    from habit.cli import cli

    cli()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-run habitat CLI without pytest.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="get-habitat-two-step",
        choices=sorted(_SCENARIOS.keys()),
        help="Which demo command to run.",
    )
    args = parser.parse_args()
    _invoke_cli(_SCENARIOS[args.scenario])


if __name__ == "__main__":
    main()
