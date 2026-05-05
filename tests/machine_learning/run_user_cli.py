"""
Run machine-learning CLI commands like a local user (no pytest).

Usage:

    python tests/machine_learning/run_user_cli.py --help
    python tests/machine_learning/run_user_cli.py model-train-clinical
    python tests/machine_learning/run_user_cli.py cv-kfold

cwd is set to the repository root before invoking ``habit``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCENARIOS: dict[str, list[str]] = {
    "model-train-clinical": [
        "model",
        "-c",
        "demo_data/config_machine_learning_clinical.yaml",
        "-m",
        "train",
    ],
    "model-train-radiomics": [
        "model",
        "-c",
        "demo_data/config_machine_learning_radiomics.yaml",
        "-m",
        "train",
    ],
    "model-predict": [
        "model",
        "-c",
        "demo_data/config_machine_learning_predict.yaml",
        "-m",
        "predict",
    ],
    "cv-kfold": ["cv", "-c", "demo_data/config_machine_learning_kfold.yaml"],
    "radiomics-standalone": [
        "radiomics",
        "-c",
        "demo_data/config_machine_learning_radiomics.yaml",
    ],
    "model-help": ["model", "--help"],
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
        description="Smoke-run ML CLI scenarios without pytest.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="model-train-clinical",
        choices=sorted(_SCENARIOS.keys()),
        help="Which demo command to run.",
    )
    args = parser.parse_args()
    _invoke_cli(_SCENARIOS[args.scenario])


if __name__ == "__main__":
    main()
