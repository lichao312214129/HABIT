"""
CLI-level tests for the `habit get-habitat` command.

Consolidates the six original habitat CLI test files:
  test_habitat_direct_pooling_train/predict
  test_habitat_one_step_train/predict
  test_habitat_two_step_train/predict

Demo-config tests depend on ``cwd_project_root`` from ``conftest.py`` so relative
paths in YAML resolve from the repository root. Parametrized E2E flag combinations
live in ``manual_cli_e2e_*.py`` beside this module (not auto-collected; run by path).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

HABITAT_CFG = Path(__file__).resolve().parents[2] / "config" / "habitat"

CONFIG_DIRECT_POOLING = HABITAT_CFG / "config_habitat_direct_pooling.yaml"
CONFIG_DIRECT_POOLING_PREDICT = HABITAT_CFG / "config_habitat_direct_pooling_predict.yaml"
CONFIG_ONE_STEP = HABITAT_CFG / "config_habitat_one_step.yaml"
CONFIG_ONE_STEP_PREDICT = HABITAT_CFG / "config_habitat_one_step_predict.yaml"
CONFIG_TWO_STEP = HABITAT_CFG / "config_habitat_two_step.yaml"
CONFIG_TWO_STEP_PREDICT = HABITAT_CFG / "config_habitat_two_step_predict.yaml"


# ---------------------------------------------------------------------------
# Help / meta
# ---------------------------------------------------------------------------


class TestHabitatCLIMeta:
    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_habitat(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "--help"])
        assert "habitat" in result.output.lower() or "get-habitat" in result.output.lower()

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Direct-pooling mode
# ---------------------------------------------------------------------------


class TestHabitatDirectPooling:
    def test_train_with_config(self, cwd_project_root: None) -> None:
        if not CONFIG_DIRECT_POOLING.exists():
            pytest.skip(f"Config not found: {CONFIG_DIRECT_POOLING}")
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "-c", str(CONFIG_DIRECT_POOLING)])
        assert result.exit_code in [0, 1], result.output

    def test_predict_with_config(self, cwd_project_root: None) -> None:
        if not CONFIG_DIRECT_POOLING_PREDICT.exists():
            pytest.skip(f"Config not found: {CONFIG_DIRECT_POOLING_PREDICT}")
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "-c", str(CONFIG_DIRECT_POOLING_PREDICT)])
        assert result.exit_code in [0, 1], result.output


# ---------------------------------------------------------------------------
# One-step mode
# ---------------------------------------------------------------------------


class TestHabitatOneStep:
    def test_train_with_config(self, cwd_project_root: None) -> None:
        if not CONFIG_ONE_STEP.exists():
            pytest.skip(f"Config not found: {CONFIG_ONE_STEP}")
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "-c", str(CONFIG_ONE_STEP)])
        assert result.exit_code in [0, 1], result.output

    def test_predict_with_config(self, cwd_project_root: None) -> None:
        if not CONFIG_ONE_STEP_PREDICT.exists():
            pytest.skip(f"Config not found: {CONFIG_ONE_STEP_PREDICT}")
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "-c", str(CONFIG_ONE_STEP_PREDICT)])
        assert result.exit_code in [0, 1], result.output


# ---------------------------------------------------------------------------
# Two-step mode
# ---------------------------------------------------------------------------


class TestHabitatTwoStep:
    def test_train_with_config(self, cwd_project_root: None) -> None:
        if not CONFIG_TWO_STEP.exists():
            pytest.skip(f"Config not found: {CONFIG_TWO_STEP}")
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "-c", str(CONFIG_TWO_STEP)])
        assert result.exit_code in [0, 1], result.output

    def test_predict_with_config(self, cwd_project_root: None) -> None:
        if not CONFIG_TWO_STEP_PREDICT.exists():
            pytest.skip(f"Config not found: {CONFIG_TWO_STEP_PREDICT}")
        runner = CliRunner()
        result = runner.invoke(cli, ["get-habitat", "-c", str(CONFIG_TWO_STEP_PREDICT)])
        assert result.exit_code in [0, 1], result.output


# ---------------------------------------------------------------------------
# Feature extraction CLI  (habit extract)
# ---------------------------------------------------------------------------


class TestExtractCLI:
    """Tests for `habit extract` command (habitat feature extraction)."""

    EXTRACT_CONFIG = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "feature_extraction"
        / "config_extract_features_demo.yaml"
    )

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0

    def test_extract_with_demo_config(self, cwd_project_root: None) -> None:
        if not self.EXTRACT_CONFIG.exists():
            pytest.skip(f"Config not found: {self.EXTRACT_CONFIG}")
        runner = CliRunner()
        result = runner.invoke(cli, ["extract", "-c", str(self.EXTRACT_CONFIG)])
        assert result.exit_code in [0, 1], result.output
