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
"""Contract tests for ``habit.compat.monai`` (Subject <-> MONAI dict)."""

from __future__ import annotations

import numpy as np
import pytest

from habit.api.exceptions import DataFormatError, HABITAPIError
from habit.compat.monai import (
    AsDictTransform,
    AsMonaiDict,
    FromMonaiDict,
    _affine_from_geometry,
    _geometry_from_affine,
    from_monai_dict,
    to_monai_dict,
)
from habit.contracts import Geometry, Subject
from tests.domain.conftest import make_subject

torch = pytest.importorskip("torch")
monai = pytest.importorskip("monai")


@pytest.mark.unit
def test_roundtrip_preserves_arrays_geometry_and_metadata() -> None:
    """Subject -> dict -> Subject is lossless, channel axis included."""
    subject = make_subject("P1", modalities=("T1", "T2"))
    sample = to_monai_dict(subject, channel_first=True)

    assert sample["subject_id"] == "P1"
    assert sample["T1"].shape == (1, 6, 6, 6)
    assert set(sample) >= {
        "T1", "T1_meta_dict", "T2", "T2_meta_dict",
        "tumor", "tumor_meta_dict", "subject_id", "metadata",
    }
    meta = sample["T1_meta_dict"]
    assert set(meta) == {"spacing", "origin", "direction", "affine"}

    restored = from_monai_dict(
        sample, mask_keys=("tumor",), squeeze_channel=True
    )
    assert restored.subject_id == "P1"
    assert set(restored.images) == {"T1", "T2"}
    assert set(restored.masks) == {"tumor"}
    np.testing.assert_array_equal(
        np.asarray(restored.image("T1").data), np.asarray(subject.image("T1").data)
    )
    np.testing.assert_array_equal(
        np.asarray(restored.mask("tumor").data), np.asarray(subject.mask("tumor").data)
    )
    assert restored.image("T1").geometry == subject.image("T1").geometry


@pytest.mark.unit
def test_affine_geometry_conversion_handles_anisotropic_grids() -> None:
    """spacing/origin/direction survive the affine round trip exactly."""
    geometry = Geometry(
        shape=(4, 5, 6),
        spacing=(2.0, 1.5, 0.75),
        origin=(10.0, -5.0, 3.0),
        direction=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    )
    affine = _affine_from_geometry(geometry)
    assert affine.shape == (4, 4)
    np.testing.assert_allclose(affine[:3, 3], geometry.origin)
    recovered = _geometry_from_affine(affine, geometry.shape)
    np.testing.assert_allclose(recovered.spacing, geometry.spacing)
    np.testing.assert_allclose(recovered.origin, geometry.origin)
    np.testing.assert_allclose(recovered.direction, geometry.direction)


@pytest.mark.unit
def test_from_monai_dict_rebuilds_geometry_from_meta_or_affine() -> None:
    """Explicit meta entries win; affine alone decomposes; nothing is a grid."""
    array = np.zeros((2, 3, 4), dtype=np.float64)
    base = {"subject_id": "P1", "T1": array}

    # Identity fallback: no companion at all.
    plain = from_monai_dict(base)
    assert plain.image("T1").geometry.spacing == (1.0, 1.0, 1.0)

    # Affine-only companion.
    geometry = Geometry(
        shape=(2, 3, 4),
        spacing=(2.0, 2.0, 4.0),
        origin=(1.0, 2.0, 3.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    with_affine = from_monai_dict(
        {**base, "T1_meta_dict": {"affine": _affine_from_geometry(geometry)}}
    )
    np.testing.assert_allclose(with_affine.image("T1").geometry.spacing, (2.0, 2.0, 4.0))
    np.testing.assert_allclose(with_affine.image("T1").geometry.origin, (1.0, 2.0, 3.0))


@pytest.mark.unit
def test_from_monai_dict_routes_scalars_to_metadata() -> None:
    """Non-array entries (clinical fields) survive in Subject.metadata."""
    sample = {
        "subject_id": "P1",
        "T1": np.zeros((2, 2, 2)),
        "age": 71,
        "metadata": {"centre": "A"},
    }
    subject = from_monai_dict(sample)
    assert subject.metadata["age"] == 71
    assert subject.metadata["centre"] == "A"


@pytest.mark.unit
def test_from_monai_dict_error_paths() -> None:
    """Missing ids, bad channel axes and non-integer masks are refused."""
    with pytest.raises(DataFormatError, match="without an id"):
        from_monai_dict({"T1": np.zeros((2, 2, 2))})
    with pytest.raises(DataFormatError, match="singleton channel"):
        from_monai_dict(
            {"subject_id": "P1", "T1": np.zeros((2, 2, 2))},
            squeeze_channel=True,
        )
    with pytest.raises(DataFormatError, match="not integer-valued"):
        from_monai_dict(
            {"subject_id": "P1", "label": np.full((2, 2, 2), 0.5)},
            mask_keys=("label",),
        )
    # Float masks holding exact integers are accepted and become int32.
    subject = from_monai_dict(
        {"subject_id": "P1", "label": np.ones((2, 2, 2), dtype=np.float64)},
        mask_keys=("label",),
    )
    assert np.asarray(subject.mask("label").data).dtype == np.int32


@pytest.mark.unit
def test_transform_wrappers_are_plain_callables() -> None:
    """AsMonaiDict/FromMonaiDict compose like MONAI transforms."""
    subject = make_subject("P1")
    with pytest.raises(HABITAPIError, match="habit Subject"):
        AsMonaiDict()({"not": "a subject"})

    sample = AsMonaiDict(channel_first=True)(subject)
    restored = FromMonaiDict(mask_keys=("tumor",), squeeze_channel=True)(sample)
    assert restored.subject_id == "P1"


@pytest.mark.unit
def test_as_dict_transform_runs_habit_operators_inside_dict_pipelines() -> None:
    """A HABIT operator drops into a dict pipeline, writing results back."""

    def count_mask_voxels(subject: Subject) -> int:
        return int(np.asarray(subject.mask("tumor").data).sum())

    sample = to_monai_dict(make_subject("P1"))
    transform = AsDictTransform(
        count_mask_voxels, result_key="tumor_voxels", mask_keys=("tumor",)
    )
    updated = transform(sample)
    assert updated["tumor_voxels"] == 4 * 4 * 4  # inner block of the 6^3 grid
    assert "T1" in updated  # the original sample is preserved, not replaced

    # Without result_key the operator's result replaces the sample outright.
    raw = AsDictTransform(count_mask_voxels, mask_keys=("tumor",))(sample)
    assert raw == 4 * 4 * 4

    with pytest.raises(HABITAPIError, match="callable"):
        AsDictTransform(42)


@pytest.mark.unit
def test_torch_and_metatensor_inputs_convert_via_duck_typing() -> None:
    """torch tensors and MetaTensors feed from_monai_dict unchanged in value."""
    array = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    sample = {
        "subject_id": "P1",
        "T1": torch.from_numpy(array),
        "label": monai.data.MetaTensor(np.ones((2, 2, 2), dtype=np.int32)),
    }
    subject = from_monai_dict(sample, mask_keys=("label",))
    np.testing.assert_array_equal(np.asarray(subject.image("T1").data), array)
    assert np.asarray(subject.mask("label").data).sum() == 8


@pytest.mark.unit
def test_monai_compose_accepts_the_wrappers() -> None:
    """The wrappers run inside monai.transforms.Compose as documented."""
    compose = monai.transforms.Compose(
        [
            AsMonaiDict(channel_first=True),
            FromMonaiDict(mask_keys=("tumor",), squeeze_channel=True),
        ]
    )
    restored = compose(make_subject("P1"))
    assert isinstance(restored, Subject)
    np.testing.assert_array_equal(
        np.asarray(restored.image("T1").data),
        np.asarray(make_subject("P1").image("T1").data),
    )
