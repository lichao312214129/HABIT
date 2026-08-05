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
"""Pytest fixtures for cloud behavior tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.cloud_behavior.helpers import parse_yaml, spec_from_v0_payload
from tests.cloud_behavior.synthetic_data import (
    build_synthetic_tree,
    minimal_v0_habitat_yaml,
)
from habit.spec.specs import HabitatSpec


@pytest.fixture
def synthetic_tree(tmp_path: Path) -> Path:
    """
    Provide a byte-stable synthetic cohort directory tree.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Cohort root containing ``images/`` and ``masks/`` subtrees.
    """
    data_root = tmp_path / "synthetic_data"
    return build_synthetic_tree(data_root)


@pytest.fixture
def habitat_spec(synthetic_tree: Path, tmp_path: Path) -> HabitatSpec:
    """
    Build the translated :class:`HabitatSpec` for the synthetic cohort.

    Args:
        synthetic_tree: Synthetic cohort root fixture.
        tmp_path: Pytest temporary directory.

    Returns:
        Habitat specification matching the minimal v0 demo config.
    """
    config_path = tmp_path / "config_habitat_minimal.yaml"
    config_path.write_text(
        minimal_v0_habitat_yaml(synthetic_tree, tmp_path / "habitat_out"),
        encoding="utf-8",
    )
    return spec_from_v0_payload(parse_yaml(config_path))


@pytest.fixture
def v0_config_path(synthetic_tree: Path, tmp_path: Path) -> Path:
    """
    Write the minimal v0 habitat YAML pointing at the synthetic tree.

    Args:
        synthetic_tree: Synthetic cohort root fixture.
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the v0 configuration file.
    """
    config_path = tmp_path / "config_habitat_minimal.yaml"
    config_path.write_text(
        minimal_v0_habitat_yaml(synthetic_tree, tmp_path / "habitat_out"),
        encoding="utf-8",
    )
    return config_path
