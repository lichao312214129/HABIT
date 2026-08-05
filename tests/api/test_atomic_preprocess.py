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
"""Atomic image-preprocessing API: subject-level and single-volume."""

from __future__ import annotations

import numpy as np
import pytest

from habit import preprocess_image, preprocess_subject
from habit.datasets import make_synthetic_cohort
from habit.exceptions import HABITAPIError


@pytest.mark.unit
def test_preprocess_subject_resample_is_atomic() -> None:
    """One Subject in, one Subject out — no data_dir / out_dir / YAML."""
    cohort = make_synthetic_cohort(n_subjects=2, modalities=("T1", "T2"), rng=0)
    subject = cohort[0]
    before = subject.image("T1")
    assert before.spacing == (1.0, 1.0, 1.0)

    processed = preprocess_subject(
        subject,
        {
            "resample": {
                "target_spacing": [2.0, 2.0, 2.0],
                "img_mode": "bilinear",
            },
        },
    )

    after = processed.image("T1")
    assert processed.subject_id == subject.subject_id
    assert tuple(round(float(v), 6) for v in after.spacing) == (2.0, 2.0, 2.0)
    # Input subject is not mutated (lazy refs still resolve to original spacing).
    assert subject.image("T1").spacing == before.spacing
    # Mask survives resampling when broadcast_mask is on.
    assert "ROI" in processed.masks or len(processed.masks) == 1


@pytest.mark.unit
def test_preprocess_image_single_volume() -> None:
    """Single ImageVolume path wraps Subject internally."""
    cohort = make_synthetic_cohort(n_subjects=1, modalities=("T1",), rng=1)
    subject = cohort[0]
    volume = subject.image("T1")
    mask = subject.mask()

    out = preprocess_image(
        volume,
        {
            "resample": {
                "target_spacing": [2.0, 2.0, 2.0],
                "img_mode": "nearest",
            },
        },
        mask=mask,
        modality="T1",
    )
    assert tuple(round(float(v), 6) for v in out.spacing) == (2.0, 2.0, 2.0)
    assert out.data.ndim == 3


@pytest.mark.unit
def test_preprocess_subject_rejects_unknown_step() -> None:
    """Unknown preprocessor names fail loudly."""
    cohort = make_synthetic_cohort(n_subjects=1, modalities=("T1",), rng=2)
    with pytest.raises(HABITAPIError, match="Unknown image preprocessor"):
        preprocess_subject(
            cohort[0],
            {"not_a_real_step": {"images": ["T1"]}},
        )


@pytest.mark.unit
def test_preprocess_subject_rejects_empty_steps() -> None:
    """Empty step mapping is rejected at the public boundary."""
    cohort = make_synthetic_cohort(n_subjects=1, modalities=("T1",), rng=3)
    with pytest.raises(HABITAPIError, match="non-empty"):
        preprocess_subject(cohort[0], {})
