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
"""Tests for machine-aware voxel radiomics batch selection."""

from __future__ import annotations

import pytest

from habit.utils.voxel_batch_utils import (
    DEFAULT_VOXEL_BATCH,
    MIN_AUTO_VOXEL_BATCH,
    recommend_voxel_batch,
    resolve_voxel_batch,
)


def test_explicit_large_batch_is_kept() -> None:
    """A user-set 4000 must not be forced down to the 1000 default."""
    assert resolve_voxel_batch(4000) == 4000
    assert resolve_voxel_batch(8000) == 8000
    assert resolve_voxel_batch(-1) == -1


def test_zero_is_rejected() -> None:
    """Zero is not a legal PyRadiomics batch."""
    with pytest.raises(ValueError, match="positive"):
        resolve_voxel_batch(0)


def test_recommend_is_positive() -> None:
    """Auto pick on this machine must be a positive batch."""
    batch = recommend_voxel_batch(kernel_radius=3, torch_device="auto")
    assert batch >= MIN_AUTO_VOXEL_BATCH


def test_eight_gb_class_recommends_default() -> None:
    """CPU probe (no CUDA in this call) falls back to the 1000 default."""
    assert recommend_voxel_batch(kernel_radius=3, torch_device="cpu") == (
        DEFAULT_VOXEL_BATCH
    )


def test_larger_kernel_does_not_raise_batch() -> None:
    """Radius 5 must not request a larger workspace than radius 3."""
    r3 = recommend_voxel_batch(kernel_radius=3, torch_device="cpu")
    r5 = recommend_voxel_batch(kernel_radius=5, torch_device="cpu")
    assert r5 <= r3


def test_resolve_auto_and_default() -> None:
    """``auto`` probes; omitted default stays 1000 unless env overrides."""
    auto = resolve_voxel_batch("auto", kernel_radius=3, torch_device="cpu")
    assert auto >= MIN_AUTO_VOXEL_BATCH
    assert resolve_voxel_batch(200) == 200
