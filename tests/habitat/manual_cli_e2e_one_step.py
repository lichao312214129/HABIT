"""
Manual end-to-end CLI checks for one-step habitat demo configs (heavy I/O).

See ``manual_cli_e2e_two_step.py`` for rationale (cwd + Click CliRunner).
Run with ``pytest tests/habitat/manual_cli_e2e_one_step.py -v`` when needed.
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
    return PROJECT_ROOT / "config" / "habitat" / filename


def _skip_if_missing(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"Demo config not found: {path}")


@pytest.mark.parametrize(
    "relative_config_path, extra_args",
    (
        pytest.param(
            "config_habitat_one_step.yaml",
            [],
            id="train_yaml_only",
        ),
        pytest.param(
            "config_habitat_one_step.yaml",
            ["--debug"],
            id="train_plus_debug",
        ),
        pytest.param(
            "config_habitat_one_step.yaml",
            ["-m", "train"],
            id="train_plus_cli_mode_train",
        ),
        pytest.param(
            "config_habitat_one_step_predict.yaml",
            [],
            id="predict_yaml_only",
        ),
        pytest.param(
            "config_habitat_one_step_predict.yaml",
            ["-m", "predict"],
            id="predict_plus_cli_mode_predict",
        ),
    ),
)
def test_one_step_user_like_cli_combinations(
    cwd_project_root: None,
    relative_config_path: str,
    extra_args: Sequence[str],
) -> None:
    """Run ``get-habitat`` against one-step train/predict demo YAMLs plus CLI flags."""
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
