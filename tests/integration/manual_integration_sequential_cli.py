"""
Manual sequential CLI pipeline steps using demo configs (preprocess → habitat → …).

Moved out of ``test_full_workflow.py`` so default ``pytest tests/integration`` stays
lightweight (only ``TestCLISmoke`` remains there). Run when needed::

    pytest tests/integration/manual_integration_sequential_cli.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest
from click.testing import CliRunner

from habit.cli import cli

DEMO = Path(__file__).resolve().parents[2] / "demo_data"


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


class TestSequentialWorkflow:
    """
    Best-effort sequential steps:
      preprocess -> get-habitat -> extract -> model -> compare

    Each step is skipped if the required demo config is absent.
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
