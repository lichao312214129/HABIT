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
"""Contract tests for Subject, Cohort and the in-memory operator ladder."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from habit.api.exceptions import HABITAPIError, ProcessingError
from habit.contracts import (
    ArrayImageRef,
    Cohort,
    Geometry,
    ImageVolume,
    MaskVolume,
    Subject,
)


def _make_subject(subject_id: str = "P001", shape=(3, 4, 5)) -> Subject:
    """Build a fully in-memory subject with one modality and one ROI."""
    geometry = Geometry.from_array(shape, spacing=(1.0, 1.0, 2.0))
    image = ArrayImageRef(array=np.ones(shape, dtype=np.float32), geometry=geometry)
    mask = ArrayImageRef(
        array=(np.arange(int(np.prod(shape))).reshape(shape) % 2).astype(np.uint8),
        geometry=geometry,
    )
    return Subject(
        subject_id=subject_id,
        images={"T1": image},
        masks={"tumor": mask},
        metadata={"center": "A"},
    )


@pytest.mark.unit
def test_subject_materialises_modalities_and_masks() -> None:
    """Subject.image / Subject.mask return geometry-bound volumes."""
    subject = _make_subject()

    image = subject.image("T1")
    mask = subject.mask()

    assert isinstance(image, ImageVolume)
    assert isinstance(mask, MaskVolume)
    assert image.geometry.is_compatible_with(mask.geometry)
    assert image.modality == "T1"
    assert mask.roi_name == "tumor"


@pytest.mark.unit
def test_subject_mask_requires_explicit_roi_when_ambiguous() -> None:
    """With several masks, mask() refuses to silently pick one."""
    subject = _make_subject()
    subject.masks["node"] = subject.masks["tumor"]

    with pytest.raises(ValueError):
        subject.mask()
    assert subject.mask("node").roi_name == "node"


@pytest.mark.unit
def test_subject_missing_keys_raise_key_error() -> None:
    """Absent modalities and ROIs surface as KeyError with context."""
    subject = _make_subject()
    with pytest.raises(KeyError):
        subject.image("T2")
    with pytest.raises(KeyError):
        subject.mask("node")


@pytest.mark.unit
def test_subject_accepts_eager_volumes_directly() -> None:
    """Materialised volumes can fill the image slots (one family of types)."""
    geometry_shape = (2, 2, 2)
    volume = ImageVolume.from_array(np.ones(geometry_shape, dtype=np.float32))
    mask = MaskVolume.from_array(np.ones(geometry_shape, dtype=np.uint8))
    subject = Subject(subject_id="P002", images={"T1": volume}, masks={"tumor": mask})

    assert subject.image("T1") is volume
    assert subject.mask("tumor") is mask


@pytest.mark.unit
def test_cohort_rejects_duplicate_subject_ids() -> None:
    """subject_id uniqueness is enforced at construction."""
    with pytest.raises(HABITAPIError):
        Cohort([_make_subject("dup"), _make_subject("dup")])


@pytest.mark.unit
def test_cohort_sequence_protocol_and_filter() -> None:
    """Cohort supports len / index / slice / iteration / filter in order."""
    cohort = Cohort([_make_subject(f"P{i:03d}") for i in range(4)], name="training")

    assert len(cohort) == 4
    assert cohort[0].subject_id == "P000"
    assert cohort.subject_ids == ("P000", "P001", "P002", "P003")
    sliced = cohort[1:3]
    assert isinstance(sliced, Cohort)
    assert sliced.subject_ids == ("P001", "P002")
    assert sliced.name == "training"
    filtered = cohort.filter(lambda s: s.subject_id.endswith("2"))
    assert filtered.subject_ids == ("P002",)


@pytest.mark.unit
def test_cohort_map_single_subject_atomic_ladder() -> None:
    """op(subject), cohort.map(op) and checkpointed map form one ladder."""
    cohort = Cohort([_make_subject(f"P{i}") for i in range(3)])

    def op(subject: Subject) -> str:
        return subject.subject_id.lower()

    # The atomic call needs no cohort, backend, or configuration.
    assert op(cohort[0]) == "p0"
    # The whole cohort runs serially by default, in cohort order.
    assert cohort.map(op) == ["p0", "p1", "p2"]


@pytest.mark.unit
def test_cohort_map_reports_failures_with_subject_context() -> None:
    """A failing subject surfaces its id and error, not a bare traceback."""

    cohort = Cohort([_make_subject("ok"), _make_subject("bad")])

    def op(subject: Subject) -> str:
        if subject.subject_id == "bad":
            raise RuntimeError("boom")
        return "fine"

    with pytest.raises(ProcessingError, match="bad"):
        cohort.map(op)


@pytest.mark.unit
def test_cohort_summarize_is_non_identifiable() -> None:
    """The cohort fingerprint carries a digest, never the raw ids."""
    cohort = Cohort([_make_subject("alice"), _make_subject("bob")], name="training")
    fingerprint = cohort.summarize()

    assert fingerprint.n_subjects == 2
    assert fingerprint.modalities == ("T1",)
    assert fingerprint.name == "training"
    assert "alice" not in fingerprint.subject_id_digest
    same = Cohort([_make_subject("alice"), _make_subject("bob")]).summarize()
    different = Cohort([_make_subject("alice"), _make_subject("carol")]).summarize()
    assert fingerprint.subject_id_digest == same.subject_id_digest
    assert fingerprint.subject_id_digest != different.subject_id_digest


@pytest.mark.unit
def test_subject_and_cohort_pickle_for_process_boundaries() -> None:
    """Lazy subjects must cross process boundaries (multiprocessing)."""
    cohort = Cohort([_make_subject("P1"), _make_subject("P2")])
    restored = pickle.loads(pickle.dumps(cohort))
    assert restored.subject_ids == ("P1", "P2")
    assert np.allclose(restored[0].image("T1").data, 1.0)
