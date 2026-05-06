"""
Run integration-style CLI flows like a local user (no pytest).

Supports:

- Single-command demos (icc, retest).
- Optional multi-step workflow: preprocess → get-habitat → extract → model → compare (same order as ``integration/manual_end_to_end_workflow.py``).

Usage:

    python tests/integration/run_user_cli.py --help
    python tests/integration/run_user_cli.py retest-demo
    python tests/integration/run_user_cli.py workflow-all

Each ``habit`` invocation rebuilds ``sys.argv`` and calls ``cli()`` once (same pattern as manual debugging).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SINGLE_SCENARIOS: dict[str, list[str]] = {
    "icc-demo": ["icc", "-c", "demo_data/config_icc.yaml"],
    "icc-help": ["icc", "--help"],
    "retest-demo": ["retest", "-c", "demo_data/config_test_retest.yaml"],
    "retest-help": ["retest", "--help"],
}

_WORKFLOW_CHAIN: list[list[str]] = [
    ["preprocess", "-c", "demo_data/config_preprocessing.yaml"],
    ["get-habitat", "-c", "demo_data/config_habitat_two_step.yaml"],
    ["extract", "-c", "demo_data/config_extract_features.yaml"],
    [
        "model",
        "-c",
        "demo_data/config_machine_learning_clinical.yaml",
        "-m",
        "train",
    ],
    ["compare", "-c", "demo_data/config_model_comparison.yaml"],
]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _invoke_cli_once(argv_tail: list[str]) -> None:
    """Match ``if __name__ == '__main__'`` debug blocks: one argv, one ``cli()`` call."""
    root = _repository_root()
    os.chdir(root)
    sys.argv = ["habit", *argv_tail]
    from habit.cli import cli

    cli()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integration CLI smoke runs without pytest.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="retest-demo",
        choices=sorted(_SINGLE_SCENARIOS.keys()) + ["workflow-all"],
        help="Single demo or sequential workflow-all.",
    )
    args = parser.parse_args()

    if args.scenario == "workflow-all":
        for step_idx, tail in enumerate(_WORKFLOW_CHAIN, start=1):
            print(f"\n--- workflow step {step_idx}/{len(_WORKFLOW_CHAIN)}: habit {' '.join(tail)} ---\n")
            _invoke_cli_once(tail)
        return

    _invoke_cli_once(_SINGLE_SCENARIOS[args.scenario])


if __name__ == "__main__":
    main()
