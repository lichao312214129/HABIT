"""
CLI-level tests for `habit compare` command (model comparison).

Heavy demo YAML run: ``tests/model_comparison/manual_cli_compare.py``.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from habit.cli import cli


class TestCompareCLI:
    # ------------------------------------------------------------------
    # Help / meta
    # ------------------------------------------------------------------

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_compare(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert "compare" in result.output.lower() or "model" in result.output.lower()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0
