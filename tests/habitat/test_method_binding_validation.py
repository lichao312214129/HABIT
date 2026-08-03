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
"""Tests for functional DSL parameter-binding validation."""

from __future__ import annotations

import logging

import pytest

from habit.core.habitat_analysis.clustering_features.method_binding_validation import (
    validate_feature_method_binding,
)

def test_kinetic_requires_timestamps_in_parentheses() -> None:
    """kinetic(...) must list timestamps inside its parentheses."""
    with pytest.raises(ValueError, match="timestamps"):
        validate_feature_method_binding(
            "kinetic(raw(LAP), raw(PVP))",
            {},
            level_name="test",
        )


def test_minimal_voxel_radiomics_passes() -> None:
    """Omitting params_file and optional knobs is valid."""
    validate_feature_method_binding(
        "concat(voxel_radiomics(T2))",
        {},
        level_name="test",
    )


def test_minimal_voxel_radiomics_uses_ct_defaults() -> None:
    """Omitting all optional params yields CT R3B12 kernel_radius and bundled preset."""
    from habit.core.habitat_analysis.services.feature_service import resolve_voxel_step_params

    resolved = resolve_voxel_step_params({}, {}, method="voxel_radiomics")
    assert resolved["kernel_radius"] == 3
    assert resolved["voxel_batch"] == 1000
    assert resolved["params_file"].endswith("params_voxel_radiomics.yaml")


def test_implicit_param_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Params known to a method but not listed in parentheses emit a warning."""
    # Inject an isolated logger because other integration tests intentionally
    # reconfigure the global ``habit`` logger and stop propagation.
    capture_logger = logging.getLogger("tests.method_binding.implicit")
    caplog.set_level(logging.WARNING, logger=capture_logger.name)
    validate_feature_method_binding(
        "concat(voxel_radiomics(T2, kernel_radius))",
        {"kernel_radius": 3, "voxel_batch": 1000},
        level_name="test",
        logger=capture_logger,
    )
    assert any("voxel_batch" in rec.message for rec in caplog.records)


def test_orphan_param_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown params keys emit an orphan warning."""
    capture_logger = logging.getLogger("tests.method_binding.orphan")
    caplog.set_level(logging.WARNING, logger=capture_logger.name)
    validate_feature_method_binding(
        "concat(voxel_radiomics(T2))",
        {"typo_key": 1},
        level_name="test",
        logger=capture_logger,
    )
    assert any("typo_key" in rec.message for rec in caplog.records)
