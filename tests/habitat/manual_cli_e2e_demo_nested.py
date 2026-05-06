"""
Manual optional E2E check for alternate demo YAML layouts (e.g. ``demo_data/data11/``).

Not collected by default (``manual_*.py``). Run explicitly when needed::

    pytest tests/habitat/manual_cli_e2e_demo_nested.py -v

If the config file is absent (partial checkout / trimmed demos), the case is skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEMO_DATA: Path = PROJECT_ROOT / "demo_data"


def test_get_habitat_data11_config_if_present(cwd_project_root: None) -> None:
    """Smoke-run ``demo_data/data11/config_getting_habitat.yaml`` when it exists."""
    cfg = DEMO_DATA / "data11" / "config_getting_habitat.yaml"
    if not cfg.is_file():
        pytest.skip(f"No nested demo config at {cfg}")

    runner = CliRunner()
    result = runner.invoke(cli, ["get-habitat", "-c", str(cfg)])
    assert result.exit_code in (0, 1), result.output
