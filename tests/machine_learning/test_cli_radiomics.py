"""
CLI-level tests for ``habit radiomics`` command (standalone radiomics pipeline entry).

Heavy demo invocation: ``manual_cli_radiomics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "machine_learning"
    / "config_machine_learning_radiomics.yaml"
)


class TestRadiomicsCLI:
    """Smoke tests for ``habit radiomics``."""

    def test_radiomics_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["radiomics", "--help"])
        assert result.exit_code == 0
        assert "radiomics" in result.output.lower()


if __name__ == "__main__":
    sys.argv = ["habit", "radiomics", "-c", str(DEMO_CONFIG)]
    cli()
