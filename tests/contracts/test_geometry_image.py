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
"""Contract tests for Geometry, ImageRef and the materialised volumes."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from habit.contracts import ArrayImageRef, Geometry, ImageRef, ImageVolume, MaskVolume


@pytest.mark.unit
def test_geometry_compatibility_requires_same_grid() -> None:
    """Two geometries are compatible only when the grids coincide."""
    base = Geometry.from_array((4, 5, 6), spacing=(1.0, 1.0, 2.0))
    same = Geometry.from_array((4, 5, 6), spacing=(1.0, 1.0, 2.0))
    different_shape = Geometry.from_array((4, 5, 7), spacing=(1.0, 1.0, 2.0))
    different_spacing = Geometry.from_array((4, 5, 6), spacing=(1.0, 1.0, 3.0))

    assert base.is_compatible_with(same)
    assert not base.is_compatible_with(different_shape)
    assert not base.is_compatible_with(different_spacing)


@pytest.mark.unit
def test_geometry_frame_of_reference_mismatch_detected() -> None:
    """Two different shared-space identifiers are never silently compatible."""
    a = Geometry.from_array((2, 2, 2), frame_of_reference="frame-a")
    b = Geometry.from_array((2, 2, 2), frame_of_reference="frame-b")
    c = Geometry.from_array((2, 2, 2))

    assert not a.is_compatible_with(b)
    # An unlabeled grid cannot prove mismatch, so it stays compatible.
    assert a.is_compatible_with(c)


@pytest.mark.unit
def test_image_volume_satisfies_image_ref_structurally() -> None:
    """The materialised volume is itself an ImageRef (one family of types)."""
    volume = ImageVolume.from_array(np.ones((3, 4, 5), dtype=np.float32), spacing=(2.0, 2.0, 2.0))

    assert isinstance(volume, ImageRef)
    assert volume.load() is volume.data
    assert volume.geometry.shape == (3, 4, 5)
    assert volume.geometry.spacing == (2.0, 2.0, 2.0)


@pytest.mark.unit
def test_mask_volume_satisfies_image_ref_and_exposes_roi() -> None:
    """MaskVolume loads itself and reports its ROI name from the modality slot."""
    mask = MaskVolume.from_array(
        (np.arange(8).reshape(2, 2, 2) > 3).astype(np.uint8),
        modality="tumor",
    )

    assert isinstance(mask, ImageRef)
    assert mask.load() is mask.data
    assert mask.roi_name == "tumor"
    assert mask.labels == (1,)


@pytest.mark.unit
def test_array_image_ref_roundtrip_and_pickle() -> None:
    """ArrayImageRef materialises with geometry and survives pickling."""
    array = np.random.rand(2, 3, 4).astype(np.float32)
    ref = ArrayImageRef(array=array, geometry=Geometry.from_array(array.shape, spacing=(1.0, 1.0, 4.0)))

    restored = pickle.loads(pickle.dumps(ref))
    assert np.allclose(restored.load(), array)
    volume = restored.load_volume(modality="T2")
    assert volume.modality == "T2"
    assert volume.geometry.spacing == (1.0, 1.0, 4.0)


@pytest.mark.unit
def test_contracts_volume_is_public_volume_subclass() -> None:
    """Contracts volumes reuse the stable public classes (single family)."""
    from habit.api.image import ImageVolume as PublicImageVolume
    from habit.api.image import MaskVolume as PublicMaskVolume

    assert issubclass(ImageVolume, PublicImageVolume)
    assert issubclass(MaskVolume, PublicMaskVolume)
