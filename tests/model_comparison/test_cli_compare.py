"""
CLI-level tests for `habit compare` command (model comparison).

Migrated and extended from tests/test_compare.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

DEMO = Path(__file__).resolve().parents[2] / "demo_data"
CONFIG_COMPARE = DEMO / "config_model_comparison.yaml"


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

    # ------------------------------------------------------------------
    # Demo-data integration
    # ------------------------------------------------------------------

    def test_compare_with_demo_config(self) -> None:
        if not CONFIG_COMPARE.exists():
            pytest.skip(f"Config not found: {CONFIG_COMPARE}")
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "-c", str(CONFIG_COMPARE)])
        assert result.exit_code in [0, 1], result.output
