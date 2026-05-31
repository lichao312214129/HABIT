"""
CLI-level tests for `habit model` command (holdout train + predict).

Heavy demo train/predict: ``tests/machine_learning/ml_train_clinical.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
CONFIG_CLINICAL = (
    CONFIG_ROOT / "machine_learning" / "config_machine_learning_clinical.yaml"
)


class TestModelCLI:
    # ------------------------------------------------------------------
    # Help / meta
    # ------------------------------------------------------------------

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_model_or_ml(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "--help"])
        assert "model" in result.output.lower() or "machine" in result.output.lower()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_invalid_run_mode_fails(self) -> None:
        """Passing an unknown mode should exit non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["model", "-c", str(CONFIG_CLINICAL), "-m", "invalid_mode"]
        )
        assert result.exit_code != 0
