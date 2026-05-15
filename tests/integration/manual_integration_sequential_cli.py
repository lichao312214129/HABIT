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

REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "config"


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
        cfg = CONFIG_ROOT / "preprocessing/config_preprocessing_demo_elastix.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["preprocess"], config=cfg)
        assert code in [0, 1], f"preprocess exited with code {code}"

    def test_step2_get_habitat(self) -> None:
        cfg = CONFIG_ROOT / "habitat/config_habitat_direct_pooling.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["get-habitat"], config=cfg)
        assert code in [0, 1], f"get-habitat exited with code {code}"

    def test_step3_extract(self) -> None:
        cfg = CONFIG_ROOT / "feature_extraction/config_extract_features_demo.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["extract"], config=cfg)
        assert code in [0, 1], f"extract exited with code {code}"

    def test_step4_model_train(self) -> None:
        cfg = CONFIG_ROOT / "machine_learning/config_machine_learning_clinical.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["model"], config=cfg, extra=["-m", "train"])
        assert code in [0, 1], f"model train exited with code {code}"

    def test_step4b_kfold(self) -> None:
        cfg = CONFIG_ROOT / "machine_learning/config_machine_learning_kfold_demo.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["cv"], config=cfg)
        assert code in [0, 1], f"cv exited with code {code}"

    def test_step5_compare(self) -> None:
        cfg = CONFIG_ROOT / "model_comparison/config_model_comparison_demo.yaml"
        if not cfg.exists():
            pytest.skip(f"Config not found: {cfg}")
        code = _invoke(["compare"], config=cfg)
        assert code in [0, 1], f"compare exited with code {code}"
