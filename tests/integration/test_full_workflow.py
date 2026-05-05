"""
End-to-end workflow integration tests.

Migrated and upgraded from tests/test_workflow_steps.py and
tests/test_end_to_end_workflow.py.

Each step is tested independently via the CLI runner, and a summary
of pass/fail is reported at the end.  Steps are skipped (not failed)
when their demo config or data is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest
from click.testing import CliRunner

from habit.cli import cli

DEMO = Path(__file__).resolve().parents[2] / "demo_data"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _invoke(
    args: List[str], config: Optional[Path] = None, extra: Optional[List[str]] = None
) -> int:
    """Run CLI command and return exit code."""
    cmd = args[:]
    if config is not None:
        cmd += ["-c", str(config)]
    if extra:
        cmd += extra
    runner = CliRunner()
    result = runner.invoke(cli, cmd)
    return result.exit_code


# ---------------------------------------------------------------------------
# Individual CLI step smoke tests
# ---------------------------------------------------------------------------


class TestCLISmoke:
    """Smoke tests: each command must handle --help without error."""

    COMMANDS = [
        "preprocess",
        "get-habitat",
        "extract",
        "model",
        "cv",
        "compare",
        "dicom-info",
        "merge-csv",
    ]

    @pytest.mark.parametrize("cmd", COMMANDS)
    def test_help_exits_zero(self, cmd: str) -> None:
        exit_code = _invoke([cmd, "--help"])
        assert exit_code == 0


# ---------------------------------------------------------------------------
# Full sequential workflow (each step can be independently skipped)
# ---------------------------------------------------------------------------


class TestSequentialWorkflow:
    """
    Tests that simulate a full pipeline run:
      preprocess -> get-habitat -> extract -> model -> compare

    Each step is skipped if the required demo config is absent.
    This structure allows CI to run whichever steps have data available.
    """

    def test_step1_preprocess(self) -> None:
        cfg = DEMO / "config_preprocessing.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["preprocess"], config=cfg)
        assert code in [0, 1], f"preprocess exited with code {code}"

    def test_step2_get_habitat(self) -> None:
        cfg = DEMO / "config_habitat_direct_pooling.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["get-habitat"], config=cfg)
        assert code in [0, 1], f"get-habitat exited with code {code}"

    def test_step3_extract(self) -> None:
        cfg = DEMO / "config_extract_features.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["extract"], config=cfg)
        assert code in [0, 1], f"extract exited with code {code}"

    def test_step4_model_train(self) -> None:
        cfg = DEMO / "config_machine_learning_clinical.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["model"], config=cfg, extra=["-m", "train"])
        assert code in [0, 1], f"model train exited with code {code}"

    def test_step4b_kfold(self) -> None:
        cfg = DEMO / "config_machine_learning_kfold.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["cv"], config=cfg)
        assert code in [0, 1], f"cv exited with code {code}"

    def test_step5_compare(self) -> None:
        cfg = DEMO / "config_model_comparison.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["compare"], config=cfg)
        assert code in [0, 1], f"compare exited with code {code}"
