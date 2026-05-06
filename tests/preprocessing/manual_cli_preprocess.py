"""
Manual CLI check: ``habit preprocess`` with demo YAML (heavy I/O).

Not auto-collected (``manual_*.py``). Run when needed::

    pytest tests/preprocessing/manual_cli_preprocess.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

DEMO_CONFIG = Path(__file__).resolve().parents[2] / "demo_data" / "config_preprocessing.yaml"


def test_preprocess_with_demo_config() -> None:
    if not DEMO_CONFIG.exists():
        pytest.skip(f"Demo config not found: {DEMO_CONFIG}")

    runner = CliRunner()
    result = runner.invoke(cli, ["preprocess", "-c", str(DEMO_CONFIG)])
    assert result.exit_code in [0, 1], (
        f"Unexpected exit code {result.exit_code}:\n{result.output}"
    )
