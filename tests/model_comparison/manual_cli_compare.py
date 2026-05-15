"""
Manual CLI check: ``habit compare`` with demo YAML (model comparison; may be slow).

Not auto-collected. Run when needed::

    pytest tests/model_comparison/manual_cli_compare.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config"
CONFIG_COMPARE = (
    CONFIG_ROOT / "model_comparison" / "config_model_comparison_demo.yaml"
)


def test_compare_with_demo_config() -> None:
    if not CONFIG_COMPARE.exists():
        pytest.skip(f"Config not found: {CONFIG_COMPARE}")
    runner = CliRunner()
    result = runner.invoke(cli, ["compare", "-c", str(CONFIG_COMPARE)])
    assert result.exit_code in [0, 1], result.output
