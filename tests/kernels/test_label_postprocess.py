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
"""Tests for connected-component label cleanup (L0)."""

from __future__ import annotations

import numpy as np
import pytest

from habit.kernels.label_postprocess import remove_small_connected_components


@pytest.mark.unit
def test_remove_small_components_merges_tiny_island_and_keeps_roi() -> None:
    """Tiny islands are reassigned; ROI voxel count stays constant."""
    label_map = np.zeros((5, 5, 5), dtype=np.int32)
    # Large component of label 1.
    label_map[1:4, 1:4, 1:4] = 1
    # One-voxel island of label 2 (below min_component_size).
    label_map[0, 0, 0] = 2
    roi_mask = label_map > 0
    roi_before = int(np.count_nonzero(roi_mask))

    cleaned = remove_small_connected_components(
        label_map,
        roi_mask,
        min_component_size=5,
        connectivity=1,
    )

    assert int(np.count_nonzero(cleaned > 0)) == roi_before
    assert cleaned[0, 0, 0] == 1
    assert set(np.unique(cleaned[roi_mask])) == {1}


@pytest.mark.unit
def test_remove_small_components_settings_override() -> None:
    """Legacy settings mapping overrides explicit keyword defaults."""
    label_map = np.zeros((3, 3, 3), dtype=np.int32)
    label_map[0:2, 0:2, 0:2] = 1
    label_map[2, 2, 2] = 2
    roi_mask = label_map > 0
    cleaned = remove_small_connected_components(
        label_map,
        roi_mask,
        min_component_size=1,
        settings={"min_component_size": 5, "connectivity": 1},
    )
    assert cleaned[2, 2, 2] == 1
