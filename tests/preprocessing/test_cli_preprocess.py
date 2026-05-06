"""
CLI-level tests for the `habit preprocess` command.

Tests verify argument parsing, help output, and error handling for missing
config. Heavy demo-data runs live in ``manual_cli_preprocess.py`` (not
auto-collected)::

    pytest tests/preprocessing/manual_cli_preprocess.py -v
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from habit.cli import cli


class TestPreprocessCLI:
    """Tests for `habit preprocess` command."""

    # ------------------------------------------------------------------
    # Help / meta
    # ------------------------------------------------------------------

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_preprocess(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert "preprocess" in result.output.lower()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "-c", "nonexistent_file.yaml"])
        assert result.exit_code != 0

    def test_no_config_arg_fails(self) -> None:
        """Invoking without -c should not silently succeed."""
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess"])
        # Should fail because -c is required, or show help
        assert result.exit_code != 0 or "help" in result.output.lower()
