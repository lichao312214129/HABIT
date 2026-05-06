"""
Manual end-to-end CLI checks for two-step habitat demo configs (heavy I/O).

Renamed with the ``manual_*.py`` prefix so normal ``pytest tests/habitat`` does
not collect this module. Run explicitly when needed::

    pytest tests/habitat/manual_cli_e2e_two_step.py -v

Covers ``habit get-habitat`` with YAML-only runs and common flag combinations
(``--debug``, explicit ``-m train`` / ``-m predict``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import pytest
from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEMO_DATA: Path = PROJECT_ROOT / "demo_data"


def _config(filename: str) -> Path:
    return DEMO_DATA / filename


def _skip_if_missing(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"Demo config not found: {path}")


@pytest.mark.parametrize(
    "relative_config_path, extra_args",
    (
        pytest.param(
            "config_habitat_two_step.yaml",
            [],
            id="train_yaml_only",
        ),
        pytest.param(
            "config_habitat_two_step.yaml",
            ["--debug"],
            id="train_plus_debug",
        ),
        pytest.param(
            "config_habitat_two_step.yaml",
            ["-m", "train"],
            id="train_plus_cli_mode_train",
        ),
        pytest.param(
            "config_habitat_two_step_predict.yaml",
            [],
            id="predict_yaml_only",
        ),
        pytest.param(
            "config_habitat_two_step_predict.yaml",
            ["-m", "predict"],
            id="predict_plus_cli_mode_predict",
        ),
    ),
)
def test_two_step_user_like_cli_combinations(
    cwd_project_root: None,
    relative_config_path: str,
    extra_args: Sequence[str],
) -> None:
    """
    Parameters
    ----------
    cwd_project_root:
        Ensures demo YAML relative paths resolve from repo root.
    relative_config_path:
        Filename under ``demo_data/``.
    extra_args:
        Tokens after ``-c <path>``.
    """
    cfg = _config(relative_config_path)
    _skip_if_missing(cfg)

    argv: List[str] = ["get-habitat", "-c", str(cfg), *list(extra_args)]
    runner = CliRunner()
    result = runner.invoke(cli, argv)

    assert result.exit_code in (
        0,
        1,
    ), (
        f"Unexpected exit {result.exit_code} for argv={argv!r}\n"
        f"stdout/stderr:\n{result.output}"
    )
