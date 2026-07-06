# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

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
    caplog.set_level(logging.WARNING)
    validate_feature_method_binding(
        "concat(voxel_radiomics(T2, kernel_radius))",
        {"kernel_radius": 3, "voxel_batch": 1000},
        level_name="test",
    )
    assert any("voxel_batch" in rec.message for rec in caplog.records)


def test_orphan_param_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown params keys emit an orphan warning."""
    caplog.set_level(logging.WARNING)
    validate_feature_method_binding(
        "concat(voxel_radiomics(T2))",
        {"typo_key": 1},
        level_name="test",
    )
    assert any("typo_key" in rec.message for rec in caplog.records)
