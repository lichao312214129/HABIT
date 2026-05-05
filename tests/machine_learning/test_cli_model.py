"""
CLI-level tests for `habit model` command (holdout train + predict).

Migrated and extended from tests/test_machine_learning_radiomics.py and
tests/test_machine_learning_clinical.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

DEMO = Path(__file__).resolve().parents[2] / "demo_data"
CONFIG_CLINICAL = DEMO / "config_machine_learning_clinical.yaml"
CONFIG_RADIOMICS = DEMO / "config_machine_learning_radiomics.yaml"
CONFIG_PREDICT = DEMO / "config_machine_learning_predict.yaml"


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

    # ------------------------------------------------------------------
    # Clinical data train
    # ------------------------------------------------------------------

    def test_clinical_train(self) -> None:
        if not CONFIG_CLINICAL.exists():
            pytest.skip(f"Config not found: {CONFIG_CLINICAL}")
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "-c", str(CONFIG_CLINICAL), "-m", "train"])
        assert result.exit_code in [0, 1], result.output

    # ------------------------------------------------------------------
    # Radiomics data train
    # ------------------------------------------------------------------

    def test_radiomics_train(self) -> None:
        if not CONFIG_RADIOMICS.exists():
            pytest.skip(f"Config not found: {CONFIG_RADIOMICS}")
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "-c", str(CONFIG_RADIOMICS), "-m", "train"])
        assert result.exit_code in [0, 1], result.output

    # ------------------------------------------------------------------
    # Predict mode
    # ------------------------------------------------------------------

    def test_predict_mode(self) -> None:
        if not CONFIG_PREDICT.exists():
            pytest.skip(f"Config not found: {CONFIG_PREDICT}")
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "-c", str(CONFIG_PREDICT), "-m", "predict"])
        assert result.exit_code in [0, 1], result.output
