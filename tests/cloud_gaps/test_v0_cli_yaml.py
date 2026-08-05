# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Gap tests for v0 CLI YAML ``config/habitat/config_habitat_two_step_cli_demo.yaml``."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli

V0_CLI_CONFIG: Path = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "habitat"
    / "config_habitat_two_step_cli_demo.yaml"
)
EXPECTED_OUT_DIR: Path = (
    Path(__file__).resolve().parents[2]
    / "demo_data"
    / "results"
    / "habitat_two_step_cli"
)


@pytest.mark.integration
def test_v0_cli_demo_check_config_exits_zero(
    cwd_repo_root: Path,
    demo_data_root: Path,
) -> None:
    """``habit check-config`` accepts the shipped v0 two-step CLI demo YAML."""
    assert demo_data_root.is_dir()
    result = CliRunner().invoke(
        cli,
        ["check-config", "-c", str(V0_CLI_CONFIG), "-w", "habitat"],
    )
    assert result.exit_code == 0, result.output
    assert "workflow=habitat" in result.output


@pytest.mark.integration
def test_v0_cli_demo_get_habitat_train_completes(
    cwd_repo_root: Path,
    demo_data_root: Path,
) -> None:
    """``habit get-habitat -m train`` runs the v0 CLI demo config end-to-end."""
    assert demo_data_root.is_dir()
    if EXPECTED_OUT_DIR.exists():
        shutil.rmtree(EXPECTED_OUT_DIR)

    result = CliRunner().invoke(
        cli,
        ["get-habitat", "-c", str(V0_CLI_CONFIG), "-m", "train"],
    )
    assert result.exit_code == 0, result.output
    assert (EXPECTED_OUT_DIR / "habitat_model.habitatmodel").is_file()
