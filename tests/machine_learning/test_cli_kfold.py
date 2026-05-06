"""
CLI-level tests for `habit cv` command (K-Fold cross-validation).

Heavy K-fold demo run: ``tests/machine_learning/manual_cli_kfold.py``.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from habit.cli import cli


class TestKFoldCLI:
    # ------------------------------------------------------------------
    # Help / meta
    # ------------------------------------------------------------------

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["cv", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_cv_or_crossvalidation(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["cv", "--help"])
        assert any(
            kw in result.output.lower()
            for kw in ("cross-validation", "k-fold", "cv", "fold")
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["cv", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0
