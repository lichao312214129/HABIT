# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for between-habitat feature contrast (panel + compare)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from habit.contracts.table import FeatureTable
from habit.habitat_features.compare import (
    HabitatFeaturePanel,
    compare_habitat_features,
    to_habitat_feature_panel,
)
from habit.exceptions import HABITAPIError

pytestmark = pytest.mark.unit


def _wide_cohort_table(
    n_subjects: int = 12,
    *,
    seed: int = 0,
    missing_last: bool = False,
) -> FeatureTable:
    """
    Synthetic each_habitat-shaped table.

    Habitat 2 is shifted +1.0 on ``glcm_Contrast_of_T2`` so the paired
    contrast H2 vs H1 is detectably positive.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_subjects):
        row = {"subject": f"s{index:03d}"}
        for hid, shift in ((1, 0.0), (2, 1.0), (3, -0.4)):
            present = not (missing_last and hid == 3 and index == n_subjects - 1)
            row[f"has_habitat_{hid}"] = 1.0 if present else 0.0
            contrast = (
                float(rng.normal(shift, 0.15)) if present else float("nan")
            )
            entropy = (
                float(rng.normal(0.2 * hid, 0.10)) if present else float("nan")
            )
            row[f"habitat_{hid}_glcm_Contrast_of_T2"] = contrast
            row[f"habitat_{hid}_glcm_Entropy_of_T2"] = entropy
        rows.append(row)
    frame = pd.DataFrame(rows)
    feature_columns = tuple(
        name for name in frame.columns if name != "subject"
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=feature_columns,
    )


def test_to_panel_melts_wide_each_habitat_columns() -> None:
    """Wide habitat_{id}_{feature} columns become a long panel."""
    table = _wide_cohort_table(n_subjects=4)
    panel = to_habitat_feature_panel(table)
    assert panel.n_subjects == 4
    assert panel.habitat_ids == (1, 2, 3)
    assert "glcm_Contrast_of_T2" in panel.feature_names
    assert "has_habitat_1" not in panel.feature_names
    assert panel.frame[panel.value_column].notna().all()


def test_to_panel_accepts_already_long_frame() -> None:
    """A long DataFrame is wrapped without re-melting."""
    frame = pd.DataFrame(
        {
            "subject": ["a", "a"],
            "habitat": [1, 2],
            "feature": ["f", "f"],
            "value": [1.0, 2.0],
        }
    )
    panel = to_habitat_feature_panel(frame)
    assert panel.n_subjects == 1
    assert panel.habitat_ids == (1, 2)


def test_to_panel_rejects_unrelated_table() -> None:
    """A table without habitat_{id}_* columns raises a clear error."""
    table = FeatureTable(
        frame=pd.DataFrame({"subject": ["a"], "age": [1.0]}),
        id_columns=("subject",),
        feature_columns=("age",),
    )
    with pytest.raises(HABITAPIError, match="habitat_"):
        to_habitat_feature_panel(table)


def test_cohort_compare_detects_shifted_habitat() -> None:
    """Habitat 2 > habitat 1 on Contrast; Cliff's delta is positive."""
    table = _wide_cohort_table(n_subjects=16, seed=1)
    comparison = compare_habitat_features(table, paired=True)
    assert comparison.is_cohort
    assert comparison.n_subjects == 16
    pair = comparison.pairwise
    # Pairs are emitted in sorted habitat order: (1, 2) means H1 vs H2.
    contrast = pair[
        (pair["feature"] == "glcm_Contrast_of_T2")
        & (pair["habitat_a"] == 1)
        & (pair["habitat_b"] == 2)
    ]
    assert len(contrast) == 1
    delta = float(contrast["effect"].iloc[0])
    # Habitat 2 is shifted +1.0, so H1 vs H2 is a negative dominance.
    assert delta < -0.5
    assert float(contrast["q_value"].iloc[0]) < 0.05


def test_missing_habitat_is_not_imputed() -> None:
    """A NaN habitat drops that subject from the pair, not filled with 0."""
    table = _wide_cohort_table(n_subjects=8, missing_last=True)
    comparison = compare_habitat_features(table)
    pair = comparison.pairwise
    row = pair[
        (pair["feature"] == "glcm_Contrast_of_T2")
        & (pair["habitat_a"] == 1)
        & (pair["habitat_b"] == 3)
    ].iloc[0]
    assert int(row["n_paired"]) == 7


def test_single_subject_compare_has_nan_p_values() -> None:
    """One subject still yields differences; inferential columns are NaN."""
    table = _wide_cohort_table(n_subjects=8, seed=2)
    comparison = compare_habitat_features(table, subject_id="s000")
    assert comparison.n_subjects == 1
    assert not comparison.is_cohort
    assert comparison.pairwise["p_value"].isna().all()
    assert comparison.pairwise["q_value"].isna().all()
    assert comparison.pairwise["mean_diff"].notna().all()


def test_for_subject_unknown_id_raises() -> None:
    """Restricting to a missing subject is an API error."""
    panel = to_habitat_feature_panel(_wide_cohort_table(n_subjects=3))
    with pytest.raises(HABITAPIError, match="no subject"):
        panel.for_subject("missing")


def test_compare_needs_two_habitats() -> None:
    """A one-habitat panel cannot be contrasted."""
    frame = pd.DataFrame(
        {
            "subject": ["a", "b"],
            "habitat": [1, 1],
            "feature": ["f", "f"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(HABITAPIError, match="two habitats"):
        compare_habitat_features(frame)


def test_top_features_ranks_by_abs_effect() -> None:
    """top_features follows |Cliff's delta|."""
    table = _wide_cohort_table(n_subjects=12, seed=3)
    comparison = compare_habitat_features(table)
    names = comparison.top_features(k=1, pair=(2, 1))
    assert names[0] == "glcm_Contrast_of_T2"


def test_panel_passthrough() -> None:
    """to_habitat_feature_panel is idempotent on a panel."""
    panel = to_habitat_feature_panel(_wide_cohort_table(n_subjects=3))
    assert to_habitat_feature_panel(panel) is panel
    assert isinstance(panel, HabitatFeaturePanel)
