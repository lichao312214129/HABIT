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
"""Tests for ``habit migrate-config`` and the dual-schema ``check-config``."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from click.testing import CliRunner

from habit.cli import cli

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
TWO_STEP_CONFIG: Path = (
    PROJECT_ROOT / "config" / "habitat" / "config_habitat_two_step.yaml"
)


def _stage_config(tmp_path: Path, name: str = "config_habitat_demo.yaml") -> Path:
    """Copy the bundled two-step config into a scratch directory."""
    staged = tmp_path / name
    shutil.copy(TWO_STEP_CONFIG, staged)
    return staged


def test_migrate_config_writes_v1_next_to_source(tmp_path: Path) -> None:
    """Default migration writes ``<name>.v1.yaml`` and reports the workflow."""
    staged = _stage_config(tmp_path)
    result = CliRunner().invoke(cli, ["migrate-config", "-c", str(staged)])

    assert result.exit_code == 0, result.output
    destination = tmp_path / "config_habitat_demo.v1.yaml"
    assert destination.is_file()
    assert "workflow=habitat" in result.output

    document = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert document["version"] == "1.0"
    assert document["workflow"] == "habitat"
    assert document["spec"]["habitat_model_fitter"]["name"] == "kmeans"


def test_migrate_config_dry_run_prints_diff_without_writing(tmp_path: Path) -> None:
    """--dry-run shows the unified diff and leaves the disk untouched."""
    staged = _stage_config(tmp_path)
    result = CliRunner().invoke(
        cli, ["migrate-config", "-c", str(staged), "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert "--- " in result.output and "+++ " in result.output
    assert "spec:" in result.output
    assert not (tmp_path / "config_habitat_demo.v1.yaml").exists()


def test_migrate_config_custom_output_path(tmp_path: Path) -> None:
    """-o redirects the v1 document to an explicit destination."""
    staged = _stage_config(tmp_path)
    destination = tmp_path / "migrated" / "out.yaml"
    result = CliRunner().invoke(
        cli, ["migrate-config", "-c", str(staged), "-o", str(destination)]
    )

    assert result.exit_code == 0, result.output
    assert destination.is_file()


def test_migrate_config_refuses_in_place_overwrite(tmp_path: Path) -> None:
    """The output path must differ from the source path."""
    staged = _stage_config(tmp_path)
    result = CliRunner().invoke(
        cli, ["migrate-config", "-c", str(staged), "-o", str(staged)]
    )

    assert result.exit_code == 1
    assert "must differ" in result.output


def test_migrate_config_refuses_v1_input(tmp_path: Path) -> None:
    """Migrating an already-v1 document is a clear no-op error."""
    staged = _stage_config(tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli, ["migrate-config", "-c", str(staged)]).exit_code == 0
    v1_path = tmp_path / "config_habitat_demo.v1.yaml"

    result = runner.invoke(cli, ["migrate-config", "-c", str(v1_path)])
    assert result.exit_code == 1
    assert "already follows the v1 layout" in result.output


def test_migrated_document_passes_check_config(tmp_path: Path) -> None:
    """The migration output validates under the dual-schema check-config."""
    staged = _stage_config(tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli, ["migrate-config", "-c", str(staged)]).exit_code == 0
    v1_path = tmp_path / "config_habitat_demo.v1.yaml"

    # No -w flag: the v1 document carries its own workflow tag.
    result = runner.invoke(cli, ["check-config", "-c", str(v1_path)])
    assert result.exit_code == 0, result.output
    assert "v1 document" in result.output
    assert "workflow=habitat" in result.output

    # An explicit -w flag must agree with the document tag.
    result = runner.invoke(cli, ["check-config", "-c", str(v1_path), "-w", "habitat"])
    assert result.exit_code == 0, result.output


def test_check_config_v1_rejects_schema_violations(tmp_path: Path) -> None:
    """A v1 document with an invalid policy section fails validation."""
    broken = tmp_path / "broken.v1.yaml"
    broken.write_text(
        yaml.safe_dump(
            {
                "version": "1.0",
                "workflow": "model",
                "policy": {"workers": 0},
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli, ["check-config", "-c", str(broken)])

    assert result.exit_code == 1
    assert "v1 document validation failed" in result.output


def test_check_config_still_validates_v0_configs() -> None:
    """The v0 schema path keeps working unchanged alongside v1 support."""
    result = CliRunner().invoke(
        cli, ["check-config", "-c", str(TWO_STEP_CONFIG), "-w", "habitat"]
    )
    assert result.exit_code == 0, result.output
    assert "workflow=habitat" in result.output
