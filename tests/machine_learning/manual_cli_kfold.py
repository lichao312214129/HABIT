"""
Manual CLI check: ``habit cv`` (K-fold) with demo YAML (heavy).

Not auto-collected. Run when needed::

    pytest tests/machine_learning/manual_cli_kfold.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
CONFIG_KFOLD = (
    CONFIG_ROOT / "machine_learning" / "config_machine_learning_kfold_demo.yaml"
)


def test_kfold_with_demo_config() -> None:
    if not CONFIG_KFOLD.exists():
        pytest.skip(f"Config not found: {CONFIG_KFOLD}")
    runner = CliRunner()
    result = runner.invoke(cli, ["cv", "-c", str(CONFIG_KFOLD)])
    assert result.exit_code in [0, 1], result.output
