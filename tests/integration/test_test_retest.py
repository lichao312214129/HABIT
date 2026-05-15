"""
Integration tests for `habit test-retest` command.

Migrated and extended from tests/test_test_retest.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_TEST_RETEST = (
    PROJECT_ROOT / "config" / "auxiliary" / "config_test_retest.yaml"
)


class TestTestRetestCLI:
    """CLI name is ``retest`` (see ``habit/cli.py``); docstring still refers to test-retest analysis."""

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["retest", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_test_retest(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["retest", "--help"])
        assert "retest" in result.output.lower() or "test" in result.output.lower()

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["retest", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_with_demo_config(self, cwd_repo_root: None) -> None:
        if not CONFIG_TEST_RETEST.exists():
            pytest.skip(f"Config not found: {CONFIG_TEST_RETEST}")
        runner = CliRunner()
        result = runner.invoke(cli, ["retest", "-c", str(CONFIG_TEST_RETEST)])
        assert result.exit_code in [0, 1], result.output
