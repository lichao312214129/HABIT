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
"""Kernel-level match of Prior 2024 ``filtering()`` Spearman drop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from habit.kernels.feature_transforms import select_precise_correlation_columns


def _prior_filtering_drop(
    frame: pd.DataFrame,
    corr_threshold: float = 0.7,
    p_threshold: float = 0.05,
) -> list[str]:
    """Byte-level copy of precise-habitats ``filtering()`` keep/drop."""
    corr_matrix, p_matrix = stats.spearmanr(frame)
    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    combined_mask = mask & (p_matrix < p_threshold)
    to_drop = [
        column
        for column in frame.columns
        if any(
            combined_mask[frame.columns.get_loc(column)]
            & (corr_matrix[:, frame.columns.get_loc(column)] > corr_threshold)
        )
    ]
    return [str(name) for name in frame.columns if name not in to_drop]


@pytest.mark.unit
def test_precise_kernel_matches_prior_filtering_on_mixed_signs() -> None:
    """Keep-last + signed r + p-gate must equal their published snippet."""
    rng = np.random.default_rng(2)
    n = 70
    a = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "a": a,
            "b": a + rng.normal(scale=0.02, size=n),
            "neg": -a,
            "noise": rng.normal(size=n),
        }
    )
    ours = select_precise_correlation_columns(frame, 0.7, 0.05)
    theirs = _prior_filtering_drop(frame, 0.7, 0.05)
    assert ours == theirs
    assert "a" not in ours
    assert "b" in ours
    assert "neg" in ours
