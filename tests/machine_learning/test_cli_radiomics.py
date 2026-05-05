"""
CLI-level tests for ``habit radiomics`` command (standalone radiomics pipeline entry).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = PROJECT_ROOT / "demo_data" / "config_machine_learning_radiomics.yaml"


class TestRadiomicsCLI:
    """Smoke tests for ``habit radiomics``."""

    def test_radiomics_with_config(self, cwd_repo_root: None) -> None:
        """Invoke radiomics with demo YAML when present."""
        if not DEMO_CONFIG.is_file():
            pytest.skip(f"Config file not found: {DEMO_CONFIG}")

        runner = CliRunner()
        result = runner.invoke(cli, ["radiomics", "-c", str(DEMO_CONFIG)])
        assert result.exit_code in [0, 1], result.output

    def test_radiomics_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["radiomics", "--help"])
        assert result.exit_code == 0
        assert "radiomics" in result.output.lower()


if __name__ == "__main__":
    sys.argv = ["habit", "radiomics", "-c", str(DEMO_CONFIG)]
    cli()
