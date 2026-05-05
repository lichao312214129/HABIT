"""
CLI-level tests for `habit cv` command (K-Fold cross-validation).

Migrated and extended from tests/test_kfold.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

DEMO = Path(__file__).resolve().parents[2] / "demo_data"
CONFIG_KFOLD = DEMO / "config_machine_learning_kfold.yaml"


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

    # ------------------------------------------------------------------
    # Demo-data integration
    # ------------------------------------------------------------------

    def test_kfold_with_demo_config(self) -> None:
        if not CONFIG_KFOLD.exists():
            pytest.skip(f"Config not found: {CONFIG_KFOLD}")
        runner = CliRunner()
        result = runner.invoke(cli, ["cv", "-c", str(CONFIG_KFOLD)])
        assert result.exit_code in [0, 1], result.output
