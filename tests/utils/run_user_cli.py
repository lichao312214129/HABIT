"""
Run utility-related CLI commands like a local user (no pytest).

Covers ``dicom-info``, ``dice``, and ``merge-csv`` with minimal temp CSV inputs.

Usage:

    python tests/utils/run_user_cli.py --help
    python tests/utils/run_user_cli.py dicom-help
    python tests/utils/run_user_cli.py merge-csv-demo

cwd is set to the repository root before invoking ``habit``, except ``merge-csv-demo``
which writes temporary CSV paths that remain valid after chdir.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _invoke_cli(argv_tail: list[str]) -> None:
    root = _repository_root()
    os.chdir(root)
    sys.argv = ["habit", *argv_tail]
    from habit.cli import cli

    cli()


def _run_merge_csv_demo() -> None:
    """Create two CSV files and merge them via the CLI (same pattern as unit tests)."""
    root = _repository_root()
    os.chdir(root)

    with tempfile.TemporaryDirectory() as tmpdir:
        tdir = Path(tmpdir)
        file1 = tdir / "a.csv"
        file2 = tdir / "b.csv"
        out_path = tdir / "merged.csv"

        pd.DataFrame({"id": ["x", "y"], "v1": [1, 2]}).to_csv(file1, index=False)
        pd.DataFrame({"id": ["x", "y"], "v2": [3, 4]}).to_csv(file2, index=False)

        sys.argv = [
            "habit",
            "merge-csv",
            str(file1),
            str(file2),
            "-o",
            str(out_path),
            "--index-col",
            "id",
        ]
        from habit.cli import cli

        cli()


_SINGLE_SCENARIOS: dict[str, list[str]] = {
    "dicom-help": ["dicom-info", "--help"],
    "dicom-demo": [
        "dicom-info",
        "--input",
        "demo_data/dicom",
        "--output",
        "dicom_info_from_run_user_cli.csv",
        "--one-file-per-folder",
    ],
    "dice-help": ["dice", "--help"],
    "merge-help": ["merge-csv", "--help"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-run utility CLI without pytest.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="dicom-help",
        choices=sorted(_SINGLE_SCENARIOS.keys()) + ["merge-csv-demo"],
        help="Utility CLI scenario.",
    )
    args = parser.parse_args()

    if args.scenario == "merge-csv-demo":
        _run_merge_csv_demo()
        return

    _invoke_cli(_SINGLE_SCENARIOS[args.scenario])


if __name__ == "__main__":
    main()
