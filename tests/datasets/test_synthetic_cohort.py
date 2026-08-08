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
"""Unit tests for :func:`habit.datasets.make_synthetic_cohort` realism."""

from __future__ import annotations

import numpy as np
import pytest

from habit.datasets import make_synthetic_cohort


@pytest.mark.unit
def test_make_synthetic_cohort_default_signature() -> None:
    """Default args still build a named multi-modality cohort."""
    cohort = make_synthetic_cohort()
    assert len(cohort) == 4
    assert cohort.name == "synthetic"
    subject = cohort[0]
    assert sorted(subject.images) == ["T1", "T2"]
    assert "tumor" in subject.masks


@pytest.mark.unit
def test_demo_realism_has_soft_background_and_aligned_mask() -> None:
    """
    Demo volumes keep non-zero tissue outside the ROI and align the mask.

    Legacy volumes zeroed the background; demo realism must not.
    """
    cohort = make_synthetic_cohort(
        n_subjects=1,
        shape=(24, 24, 24),
        rng=42,
        realism="demo",
    )
    subject = cohort[0]
    t1 = np.asarray(subject.image("T1").data, dtype=np.float64)
    mask = np.asarray(subject.mask("tumor").data)
    assert mask.dtype == np.int32 or np.issubdtype(mask.dtype, np.integer)
    assert int(mask.sum()) > 0
    assert int((mask == 0).sum()) > 0
    # Soft tissue background: outside-ROI mean clearly above pure zero.
    outside = t1[mask == 0]
    assert float(outside.mean()) > 0.1
    assert float(outside.std()) > 0.01
    # Lesion interior brighter / more structured than a flat field.
    inside = t1[mask > 0]
    assert float(inside.std()) > float(outside.std()) * 0.5


@pytest.mark.unit
def test_demo_modalities_are_correlated_but_not_identical() -> None:
    """T1/T2 share structure inside the ROI but use different contrast."""
    cohort = make_synthetic_cohort(
        n_subjects=1,
        modalities=("T1", "T2"),
        shape=(20, 20, 20),
        rng=7,
        realism="demo",
    )
    subject = cohort[0]
    mask = np.asarray(subject.mask("tumor").data) > 0
    t1 = np.asarray(subject.image("T1").data, dtype=np.float64)[mask]
    t2 = np.asarray(subject.image("T2").data, dtype=np.float64)[mask]
    corr = float(np.corrcoef(t1, t2)[0, 1])
    # Shared layout without near-copy / near-perfect mirror contrast.
    assert -0.90 < corr < 0.90
    assert not np.allclose(t1, t2)
    assert float(np.mean(np.abs(t1 - t2))) > 0.05


@pytest.mark.unit
def test_demo_is_deterministic_for_fixed_rng() -> None:
    """The same master seed reproduces identical demo volumes."""
    first = make_synthetic_cohort(n_subjects=2, shape=(16, 16, 16), rng=99)
    second = make_synthetic_cohort(n_subjects=2, shape=(16, 16, 16), rng=99)
    for left, right in zip(first, second):
        for modality in left.images:
            np.testing.assert_array_equal(
                left.image(modality).data,
                right.image(modality).data,
            )
        np.testing.assert_array_equal(
            left.mask("tumor").data,
            right.mask("tumor").data,
        )


@pytest.mark.unit
def test_legacy_realism_keeps_zero_background() -> None:
    """``realism='legacy'`` preserves the flat zero-background contract."""
    cohort = make_synthetic_cohort(
        n_subjects=1,
        shape=(16, 16, 16),
        rng=0,
        realism="legacy",
    )
    t1 = np.asarray(cohort[0].image("T1").data)
    mask = np.asarray(cohort[0].mask("tumor").data)
    assert np.allclose(t1[mask == 0], 0.0)


@pytest.mark.unit
def test_invalid_realism_raises() -> None:
    """Unsupported realism values are rejected early."""
    with pytest.raises(ValueError, match="realism"):
        make_synthetic_cohort(n_subjects=1, realism="clinical")  # type: ignore[arg-type]
