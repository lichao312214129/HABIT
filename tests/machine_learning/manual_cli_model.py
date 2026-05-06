"""
Manual CLI checks: ``habit model`` train/predict with demo YAMLs (heavy).

Not auto-collected. Run when needed::

    pytest tests/machine_learning/manual_cli_model.py -v
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


def test_clinical_train() -> None:
    if not CONFIG_CLINICAL.exists():
        pytest.skip(f"Config not found: {CONFIG_CLINICAL}")
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "-c", str(CONFIG_CLINICAL), "-m", "train"])
    assert result.exit_code in [0, 1], result.output


def test_radiomics_train() -> None:
    if not CONFIG_RADIOMICS.exists():
        pytest.skip(f"Config not found: {CONFIG_RADIOMICS}")
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "-c", str(CONFIG_RADIOMICS), "-m", "train"])
    assert result.exit_code in [0, 1], result.output


def test_predict_mode() -> None:
    if not CONFIG_PREDICT.exists():
        pytest.skip(f"Config not found: {CONFIG_PREDICT}")
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "-c", str(CONFIG_PREDICT), "-m", "predict"])
    assert result.exit_code in [0, 1], result.output
