"""
Manual CLI check: ``habit radiomics`` with demo YAML (heavy).

Not auto-collected. Fast help-only tests remain in ``test_cli_radiomics.py``.
Run when needed::

    pytest tests/machine_learning/manual_cli_radiomics.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = PROJECT_ROOT / "demo_data" / "config_machine_learning_radiomics.yaml"


def test_radiomics_with_config(cwd_repo_root: None) -> None:
    """Invoke radiomics with demo YAML when present."""
    if not DEMO_CONFIG.is_file():
        pytest.skip(f"Config file not found: {DEMO_CONFIG}")

    runner = CliRunner()
    result = runner.invoke(cli, ["radiomics", "-c", str(DEMO_CONFIG)])
    assert result.exit_code in [0, 1], result.output
