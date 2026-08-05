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
"""
Coverage matrix: CLI smoke tests for every ``habit`` subcommand.

Every subcommand is at minimum invoked with ``--help`` and must exit 0.
Subcommands with cheap real runs get one here (check-config,
migrate-config, merge-csv, dice); the heavier ones are covered end-to-end
by their dedicated matrix modules. ``gui`` launches a web server, so it is
exercised help-only plus an import check of its command module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from habit.cli import cli
from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree

#: All 16 v1.0.0 subcommands.
SUBCOMMANDS = (
    "check-config",
    "compare",
    "cv",
    "dice",
    "dicom-info",
    "extract",
    "get-habitat",
    "gui",
    "icc",
    "merge-csv",
    "migrate-config",
    "model",
    "preprocess",
    "radiomics",
    "retest",
    "sort-dicom",
)


@pytest.mark.unit
@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_subcommand_help_exits_zero(subcommand: str) -> None:
    """``habit <subcommand> --help`` exits 0 and prints a usage line."""
    result = run_cli(CliRunner(), [subcommand, "--help"])
    assert "Usage:" in result.output or "usage:" in result.output.lower()


@pytest.mark.unit
def test_cli_lists_all_sixteen_subcommands() -> None:
    """The top-level group registers exactly the documented subcommands."""
    result = run_cli(CliRunner(), ["--help"])
    for subcommand in SUBCOMMANDS:
        assert subcommand in result.output, f"{subcommand} missing from habit --help"


@pytest.mark.unit
def test_gui_command_module_importable() -> None:
    """The gui command module imports (the server itself is not started)."""
    import importlib

    module = importlib.import_module("habit.commands.cmd_gui")
    assert hasattr(module, "run_next_gui_server")


@pytest.mark.integration
def test_check_config_accepts_valid_v0_config(
    synthetic_tree: SyntheticTree, render_config
) -> None:
    """check-config validates a rendered v0 habitat config."""
    rendered: RenderedConfig = render_config(
        "habitat_two_step_train.yaml", "check_config_v0", synthetic_tree
    )
    run_cli(CliRunner(), ["check-config", "-c", str(rendered.path)])


@pytest.mark.integration
def test_check_config_accepts_valid_v1_config(
    synthetic_tree: SyntheticTree, render_config
) -> None:
    """check-config validates a rendered v1 habitat document."""
    rendered: RenderedConfig = render_config(
        "habitat_two_step_v1.yaml", "check_config_v1", synthetic_tree
    )
    run_cli(CliRunner(), ["check-config", "-c", str(rendered.path)])


@pytest.mark.integration
def test_migrate_config_produces_valid_v1(
    synthetic_tree: SyntheticTree, render_config, results_root: Path
) -> None:
    """migrate-config upgrades a v0 config; check-config accepts the result."""
    rendered: RenderedConfig = render_config(
        "habitat_two_step_train.yaml", "migrate_source", synthetic_tree
    )
    migrated = results_root / "migrated_v1.yaml"
    run_cli(
        CliRunner(),
        [
            "migrate-config",
            "-c",
            str(rendered.path),
            "-o",
            str(migrated),
            "-w",
            "habitat",
        ],
    )
    assert migrated.is_file()
    document = yaml.safe_load(migrated.read_text(encoding="utf-8"))
    assert document.get("version") == "1.0"
    assert "spec" in document and "data" in document
    run_cli(CliRunner(), ["check-config", "-c", str(migrated)])


@pytest.mark.integration
def test_merge_csv_cli(synthetic_tree: SyntheticTree, results_root: Path) -> None:
    """merge-csv joins the paired ICC tables on subject_id."""
    merged = results_root / "merged_icc.csv"
    run_cli(
        CliRunner(),
        [
            "merge-csv",
            str(synthetic_tree.icc_test_csv),
            str(synthetic_tree.icc_retest_csv),
            "-o",
            str(merged),
            "--index-col",
            "subject_id",
        ],
    )
    assert merged.is_file()
    frame = pd.read_csv(merged)
    assert len(frame) == 30
    # 1 id column + 8 + 8 measurement columns.
    assert frame.shape[1] == 17


@pytest.mark.integration
def test_dice_cli_identical_masks(synthetic_tree: SyntheticTree, results_root: Path) -> None:
    """dice of the mask tree against itself is exactly 1.0 per subject."""
    output_csv = results_root / "dice_self.csv"
    run_cli(
        CliRunner(),
        [
            "dice",
            "--input1",
            str(synthetic_tree.root),
            "--input2",
            str(synthetic_tree.root),
            "--output",
            str(output_csv),
        ],
    )
    assert output_csv.is_file()
    frame = pd.read_csv(output_csv)
    dice_cols = [c for c in frame.columns if "dice" in c.lower()]
    assert dice_cols, f"no dice column in {list(frame.columns)}"
    assert (frame[dice_cols[0]].astype(float) == 1.0).all()
