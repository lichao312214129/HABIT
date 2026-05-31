"""
CLI-level tests for ``habit merge-csv`` command.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from habit.cli import cli
from habit.utils.log_utils import stop_queue_listener


class TestMergeCsvCLI:
    """Tests for ``merge-csv`` subcommand."""

    def test_merge_csv_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["merge-csv", "--help"])
        assert result.exit_code == 0
        assert "merge" in result.output.lower() or "csv" in result.output.lower()

    def test_merge_csv_with_test_files(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            df1 = pd.DataFrame({"id": ["A", "B", "C"], "value1": [1, 2, 3]})
            file1 = Path(tmpdir) / "file1.csv"
            df1.to_csv(file1, index=False)

            df2 = pd.DataFrame({"id": ["A", "B", "C"], "value2": [10, 20, 30]})
            file2 = Path(tmpdir) / "file2.csv"
            df2.to_csv(file2, index=False)

            output_file = Path(tmpdir) / "merged.csv"

            try:
                result = runner.invoke(
                    cli,
                    [
                        "merge-csv",
                        str(file1),
                        str(file2),
                        "-o",
                        str(output_file),
                        "--index-col",
                        "id",
                    ],
                )

                assert result.exit_code == 0
                assert output_file.exists()

                merged = pd.read_csv(output_file)
                assert "id" in merged.columns
                assert "value1" in merged.columns
                assert "value2" in merged.columns
                assert len(merged) == 3
            finally:
                # Release merge_csv.log file handle before TemporaryDirectory cleanup (Windows).
                stop_queue_listener()

    def test_merge_csv_missing_files(self) -> None:
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "merged.csv"

            result = runner.invoke(
                cli,
                [
                    "merge-csv",
                    "nonexistent1.csv",
                    "nonexistent2.csv",
                    "-o",
                    str(output_file),
                ],
            )

            assert result.exit_code != 0


if __name__ == "__main__":
    sys.argv = [
        "habit",
        "merge-csv",
        "file1.csv",
        "file2.csv",
        "-o",
        "merged.csv",
        "--index-col",
        "id",
    ]
    cli()
