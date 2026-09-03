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

from habit.api.preprocessing import preprocess_image, preprocess_subject
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
def test_preprocess_subject_zscore_keeps_negative_float() -> None:
    """Integer input volumes must not truncate negative z-scores to zero."""
    cohort = make_synthetic_cohort(n_subjects=1, modalities=("T1",), rng=4)
    subject = cohort[0]
    raw = np.asarray(subject.image("T1").data)
    # Force an integer storage dtype like clinical NRRD/NIfTI volumes.
    from habit.contracts.image import ArrayImageRef
    from habit.contracts.subject import Subject

    int16_vol = raw.astype(np.int16)
    geo = subject.image("T1").geometry
    int_subject = Subject(
        subject_id=subject.subject_id,
        images={
            "T1": ArrayImageRef(array=int16_vol, geometry=geo),
        },
        masks=subject.masks,
        metadata=dict(subject.metadata),
    )

    processed = preprocess_subject(
        int_subject,
        {"zscore_normalization": {"only_inmask": False}},
    )
    out = np.asarray(processed.image("T1").data)
    assert np.issubdtype(out.dtype, np.floating)
    assert float(out.min()) < 0.0
    assert abs(float(out.mean())) < 1e-3
    assert abs(float(out.std()) - 1.0) < 1e-3


@pytest.mark.unit
def test_n4_uses_a_mask_only_when_explicitly_named() -> None:
    """N4 must not silently change its fitted bias field by inferring an ROI."""
    from habit.contracts.image import ArrayImageRef
    from habit.contracts.subject import Subject
    from habit.image_preprocessing.methods import N4Correction

    image_array = np.linspace(1.0, 200.0, num=27**3, dtype=np.float32).reshape(
        (27, 27, 27)
    )
    mask_array = np.zeros_like(image_array, dtype=np.uint8)
    mask_array[8:19, 8:19, 8:19] = 1
    geometry = subject_geometry = make_synthetic_cohort(
        n_subjects=1, modalities=("T1",), rng=5
    )[0].image("T1").geometry
    geometry = type(subject_geometry)(
        shape=image_array.shape,
        spacing=subject_geometry.spacing,
        origin=subject_geometry.origin,
        direction=subject_geometry.direction,
    )
    subject = Subject(
        subject_id="n4-mask-control",
        images={"T1": ArrayImageRef(array=image_array, geometry=geometry)},
        masks={"ROI": ArrayImageRef(array=mask_array, geometry=geometry)},
    )

    no_mask = N4Correction(
        num_fitting_levels=1,
        num_iterations=[2],
        shrink_factor=2,
    )(subject, images=["T1"], mask_roi="ROI")
    explicit_mask = N4Correction(
        num_fitting_levels=1,
        num_iterations=[2],
        shrink_factor=2,
        mask_name="ROI",
    )(subject, images=["T1"])

    assert not np.array_equal(
        no_mask.image("T1").data,
        explicit_mask.image("T1").data,
    )


@pytest.mark.unit
def test_preprocess_subject_uses_v1_registry() -> None:
    """Atomic preprocess_subject is served by the v1 preprocessor domain."""
    from habit.image_preprocessing import PreprocessorRegistry

    assert "zscore_normalization" in PreprocessorRegistry.available()
    assert "resample" in PreprocessorRegistry.available()


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
