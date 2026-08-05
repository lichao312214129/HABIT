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
"""CLI error-message quality tests for non-programmer users."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from habit.cli import cli
from tests.cloud_behavior.helpers import assert_cli_user_error
from tests.cloud_behavior.synthetic_data import MODALITIES, minimal_v0_habitat_yaml


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return an isolated Click test runner."""
    return CliRunner()


@pytest.mark.unit
def test_cli_rejects_yaml_with_tab_indentation(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    """
    Tab-indented YAML must fail ``habit check-config`` with actionable guidance.

    Doctors editing configs in plain-text editors often insert tabs by mistake;
    the CLI should explain the problem without a Python traceback.
    """
    config_path = tmp_path / "tab_indent.yaml"
    config_path.write_text(
        "run_mode: train\ndata_dir: x\nout_dir: y\nfeature_construction:\n\tvoxel_level:\n\t  method: x()\n",
        encoding="utf-8",
    )

    result = cli_runner.invoke(
        cli,
        ["check-config", "-c", str(config_path), "-w", "habitat"],
    )
    assert_cli_user_error(
        result,
        must_mention=("tab", "yaml", "缩进"),
        forbid_traceback=True,
    )


@pytest.mark.unit
def test_cli_rejects_yaml_missing_data_dir(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    """
    A habitat config without ``data_dir`` must fail schema validation clearly.
    """
    config_path = tmp_path / "missing_data_dir.yaml"
    config_path.write_text(
        "run_mode: train\nout_dir: /tmp/out\n",
        encoding="utf-8",
    )

    result = cli_runner.invoke(
        cli,
        ["check-config", "-c", str(config_path), "-w", "habitat"],
    )
    assert_cli_user_error(
        result,
        must_mention=("data_dir", "Field required"),
        forbid_traceback=True,
    )


@pytest.mark.integration
def test_cli_rejects_nonexistent_data_dir_without_traceback(
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    """
    ``data_dir`` referencing a missing directory must fail without a traceback.
    """
    missing = tmp_path / "does_not_exist"
    config_path = tmp_path / "bad_data_dir.yaml"
    config_path.write_text(
        minimal_v0_habitat_yaml(missing, tmp_path / "out"),
        encoding="utf-8",
    )

    result = cli_runner.invoke(cli, ["get-habitat", "-c", str(config_path)])
    assert_cli_user_error(
        result,
        must_mention=("Data path not found", str(missing)),
        forbid_traceback=True,
    )


@pytest.mark.integration
def test_cli_rejects_missing_modality_without_traceback(
    cli_runner: CliRunner,
    synthetic_tree: Path,
    tmp_path: Path,
) -> None:
    """
    Requesting a modality that is absent from the tree must fail cleanly.

    The synthetic tree only contains delay2/delay3/delay5; asking for delay9
    should surface a readable data-layout error without a traceback.
    """
    config_text = minimal_v0_habitat_yaml(synthetic_tree, tmp_path / "out")
    config_text = config_text.replace(
        f"concat(raw({MODALITIES[0]}), raw({MODALITIES[1]}), raw({MODALITIES[2]}))",
        "concat(raw(delay2), raw(delay9))",
    )
    config_path = tmp_path / "missing_modality.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    result = cli_runner.invoke(cli, ["get-habitat", "-c", str(config_path)])
    assert_cli_user_error(
        result,
        must_mention=(
            "delay9",
            "No complete subjects",
            "Modalities present in the data tree",
            MODALITIES[0],
        ),
        forbid_traceback=True,
    )


@pytest.mark.unit
def test_cli_rejects_unknown_supervoxel_algorithm_in_v0_config(
    cli_runner: CliRunner,
    synthetic_tree: Path,
    tmp_path: Path,
) -> None:
    """
    An unknown v0 supervoxel ``algorithm`` must fail with an explicit allowed list.
    """
    config_text = minimal_v0_habitat_yaml(synthetic_tree, tmp_path / "out")
    config_text = config_text.replace(
        "algorithm: kmeans",
        "algorithm: does_not_exist",
        1,
    )
    config_path = tmp_path / "unknown_algorithm.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    result = cli_runner.invoke(cli, ["get-habitat", "-c", str(config_path)])
    assert_cli_user_error(
        result,
        must_mention=("does_not_exist", "kmeans", "gmm", "slic"),
        forbid_traceback=True,
    )


@pytest.mark.unit
def test_cli_check_config_rejects_unknown_v1_supervoxelizer(
    cli_runner: CliRunner,
    synthetic_tree: Path,
    tmp_path: Path,
) -> None:
    """
    ``check-config`` should reject ``supervoxelizer.name=\"does_not_exist\"``.

    The error must name the unknown component and list registered alternatives.
    """
    config_path = tmp_path / "habitat" / "unknown_component.v1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""version: "1.0"
workflow: habitat
mode: train
spec:
  name: habitat_two_step
  voxel_feature_extractor:
    name: raw
    params:
      modalities: [{MODALITIES[0]}]
      roi: {MODALITIES[0]}
  supervoxelizer:
    name: does_not_exist
    params:
      n_supervoxels: 20
  habitat_model_fitter:
    name: kmeans
    params:
      min_habitats: 2
      max_habitats: 4
  habitat_assigner:
    name: nearest_centroid
    params: {{}}
  random_seed: 42
data:
  source: "{synthetic_tree.as_posix()}"
output:
  out_dir: "{(tmp_path / 'out').as_posix()}"
policy:
  workers: 1
""",
        encoding="utf-8",
    )

    result = cli_runner.invoke(cli, ["check-config", "-c", str(config_path)])
    assert_cli_user_error(
        result,
        must_mention=("does_not_exist", "supervoxelizer", "Available", "kmeans"),
        forbid_traceback=True,
    )
