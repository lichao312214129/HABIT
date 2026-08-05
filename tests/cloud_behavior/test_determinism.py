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
"""Determinism tests for the two-step habitat recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.cloud_behavior.helpers import (
    assert_habitat_maps_equal,
    assert_manifests_equal_except_volatile,
    assert_parquet_frames_equal,
    run_two_step_on_tree,
)
from habit.spec.specs import HabitatSpec


@pytest.mark.integration
def test_two_step_is_deterministic_on_synthetic_cohort(
    synthetic_tree: Path,
    habitat_spec: HabitatSpec,
    tmp_path: Path,
) -> None:
    """
    Running ``two_step`` twice with the same seed yields identical artefacts.

    Habitat label maps and ``habitats.parquet`` must match exactly across runs.
    ``run_manifest.json`` may differ only in volatile timestamp/run-id fields.
    """
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    result_a = run_two_step_on_tree(synthetic_tree, habitat_spec)
    result_b = run_two_step_on_tree(synthetic_tree, habitat_spec)

    assert_habitat_maps_equal(result_a.habitat_maps, result_b.habitat_maps)

    result_a.save(out_a)
    result_b.save(out_b)

    assert_parquet_frames_equal(out_a / "habitats.parquet", out_b / "habitats.parquet")
    assert_manifests_equal_except_volatile(
        out_a / "run_manifest.json",
        out_b / "run_manifest.json",
    )
