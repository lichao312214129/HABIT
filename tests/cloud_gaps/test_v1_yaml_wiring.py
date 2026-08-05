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
"""Gap tests for native v1 YAML ``config/habitat/config_habitat_two_step_v1.yaml``."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from habit.recipes.result import StudyResult
from habit.recipes.yaml_runner import run_from_yaml
from habit.spec.legacy import validate_v1_document
from habit.spec.specs import HabitatSpec

V1_CONFIG: Path = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "habitat"
    / "config_habitat_two_step_v1.yaml"
)
EXPECTED_OUT_DIR: Path = (
    Path(__file__).resolve().parents[2]
    / "demo_data"
    / "results"
    / "habitat_two_step_v1"
)


@pytest.mark.unit
def test_v1_yaml_document_parses_and_validates(repo_root: Path) -> None:
    """The shipped v1 document passes structural validation and builds a spec."""
    payload = yaml.safe_load(V1_CONFIG.read_text(encoding="utf-8"))
    validate_v1_document(payload, workflow="habitat")
    spec = HabitatSpec.from_dict(payload["spec"])
    assert spec.name == "habitat_two_step"
    assert list(spec.voxel_feature_extractor.params.get("modalities") or []) == [
        "delay2",
        "delay3",
        "delay5",
    ]


@pytest.mark.integration
def test_v1_yaml_run_from_yaml_writes_outputs(
    cwd_repo_root: Path,
    demo_data_root: Path,
) -> None:
    """``run_from_yaml`` on the v1 config completes and persists artefacts."""
    assert demo_data_root.is_dir()
    if EXPECTED_OUT_DIR.exists():
        shutil.rmtree(EXPECTED_OUT_DIR)

    result = run_from_yaml(V1_CONFIG, workflow="habitat", save=True)

    assert isinstance(result, StudyResult)
    assert result.habitat_model is not None
    assert EXPECTED_OUT_DIR.is_dir()
    assert (EXPECTED_OUT_DIR / "habitat_model.habitatmodel").is_file()
