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
"""Golden-value tests for ITH feature extraction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
from habit.kernels.habitat_metrics import habitat_region_stats, ith_score


@pytest.mark.unit
def test_ith_features_golden(
    synthetic_habitat_array: np.ndarray,
    golden_msi_ith_data: Dict[str, Any],
) -> None:
    """ITH score and summary fields must match the frozen golden snapshot."""
    stats = habitat_region_stats(synthetic_habitat_array)
    result: Dict[str, Any] = {
        "ith_score": float(ith_score(synthetic_habitat_array)),
        "num_habitats": len(stats),
        "total_area": int(np.count_nonzero(synthetic_habitat_array)),
    }
    for habitat_id, (regions, largest) in stats.items():
        result[f"habitat_{habitat_id}_regions"] = regions
        result[f"habitat_{habitat_id}_largest_area"] = largest
        result[f"habitat_{habitat_id}_area_ratio"] = largest / regions

    assert "error" not in result
    expected: Dict[str, Any] = golden_msi_ith_data["ith_features"]
    for key, expected_value in expected.items():
        assert result[key] == pytest.approx(expected_value, rel=0, abs=1e-9)
