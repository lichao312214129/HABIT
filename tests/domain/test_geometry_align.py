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
"""Tests for default mask→image geometry alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from habit.contracts import ArrayImageRef, Geometry, Subject
from habit.image_preprocessing.geometry import (
    GEOMETRY_ALIGN_METADATA_KEY,
    align_mask_to_reference,
    align_subject_masks,
    anatomy_aware_field_geometry,
    geometry_mismatch_fields,
)
from habit.pipeline import SubjectPipeline
from habit.voxel_features import RawVoxelFeatures
from habit.exceptions import GeometryError

from .conftest import make_model, make_subject


def _mismatched_mask_subject(subject_id: str = "P1") -> Subject:
    """
    Build a subject whose ROI shares Size/Spacing but not Origin/Direction.

    This mirrors the demo_data failure mode that previously raised
    GeometryError during raw voxel extraction.
    """
    subject = make_subject(subject_id, modalities=("T1", "T2"))
    image_geom = subject.image("T1").geometry
    bad_geometry = Geometry(
        shape=tuple(image_geom.shape),
        spacing=tuple(image_geom.spacing),
        origin=tuple(float(v) + 2.5 for v in image_geom.origin),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    )
    assert geometry_mismatch_fields(image_geom, bad_geometry) == (
        "origin",
        "direction",
    )
    mask_array = np.asarray(subject.mask("tumor").data).copy()
    return Subject(
        subject_id=subject.subject_id,
        images=subject.images,
        masks={
            "tumor": ArrayImageRef(array=mask_array, geometry=bad_geometry),
        },
        metadata=subject.metadata,
    )


@pytest.mark.unit
def test_align_mask_adopts_geometry_when_shape_matches() -> None:
    """Same-shaped masks keep voxels and adopt the image header metadata."""
    subject = _mismatched_mask_subject()
    image = subject.image("T1")
    mask = subject.mask("tumor")
    source_nonzero = int(np.count_nonzero(mask.data))
    aligned = align_mask_to_reference(
        mask,
        image,
        subject_id=subject.subject_id,
        roi_name="tumor",
        reference_label="T1",
    )
    assert aligned.geometry.is_compatible_with(image.geometry)
    meta = (aligned.metadata or {})[GEOMETRY_ALIGN_METADATA_KEY]
    assert meta["action"] == "adopt_geometry"
    assert int(np.count_nonzero(aligned.data)) == source_nonzero
    np.testing.assert_array_equal(aligned.data, mask.data)


@pytest.mark.unit
def test_align_mask_raises_when_shape_differs() -> None:
    """Different array shapes always raise: auto-resampling across physical
    extents would silently drop or invent mask voxels."""
    subject = make_subject("P1", shape=(6, 6, 6))
    image = subject.image("T1")
    # Build a smaller mask geometry so shapes disagree.
    small = Geometry.from_array((4, 4, 4), spacing=image.geometry.spacing)
    mask_array = np.zeros((4, 4, 4), dtype=np.int32)
    mask_array[1:3, 1:3, 1:3] = 1
    from habit.contracts.image import MaskVolume

    mask = MaskVolume.from_geometry(mask_array, small, roi_name="tumor")
    with pytest.raises(GeometryError, match="different physical extents"):
        align_mask_to_reference(
            mask,
            image,
            subject_id="P1",
            roi_name="tumor",
            reference_label="T1",
        )


@pytest.mark.unit
def test_align_mask_strict_raises() -> None:
    """Strict policy keeps the historical GeometryError behaviour."""
    subject = _mismatched_mask_subject()
    with pytest.raises(GeometryError, match="compatible voxel grid"):
        align_mask_to_reference(
            subject.mask("tumor"),
            subject.image("T1"),
            on_geometry_mismatch="strict",
            subject_id=subject.subject_id,
            roi_name="tumor",
            reference_label="T1",
        )


@pytest.mark.unit
def test_raw_extractor_default_resamples_mismatched_mask() -> None:
    """Direct voxel-extractor API succeeds under the default resample policy."""
    subject = _mismatched_mask_subject()
    field = RawVoxelFeatures(modalities=["T1"])(subject)
    image_geom = subject.image("T1").geometry
    source_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    # Index grid follows the image; direction stays the original mask header
    # so HabitatMap display can prefer labelled anatomy (demo LAP +z sign).
    assert tuple(field.geometry.shape) == tuple(image_geom.shape)
    assert tuple(field.geometry.direction) == source_direction
    assert not field.geometry.is_compatible_with(image_geom)
    assert field.values.shape[0] == field.voxel_index.shape[0]
    assert field.values.shape[0] > 0
    assert GEOMETRY_ALIGN_METADATA_KEY in (field.provenance.notes or {})


@pytest.mark.unit
def test_raw_extractor_strict_roi_voxels_still_raises() -> None:
    """Callers can still opt into strict checks at the helper boundary."""
    from habit.voxel_features._base import roi_voxels

    subject = _mismatched_mask_subject()
    with pytest.raises(GeometryError):
        roi_voxels(subject, "tumor", on_geometry_mismatch="strict")


@pytest.mark.unit
def test_subject_pipeline_strict_raises_on_mismatch() -> None:
    """HabitatSpec-equivalent strict policy fails fast in SubjectPipeline."""
    subject = _mismatched_mask_subject()
    pipeline = SubjectPipeline(
        voxel_feature_extractor=RawVoxelFeatures(modalities=["T1"]),
        supervoxelizer=None,
        habitat_assigner=None,
        on_geometry_mismatch="strict",
    )
    with pytest.raises(GeometryError):
        pipeline.units(subject)


@pytest.mark.unit
def test_align_subject_masks_rewrites_mapping() -> None:
    """Subject-level alignment returns a new subject with MaskVolume ROIs."""
    subject = _mismatched_mask_subject()
    aligned = align_subject_masks(subject)
    assert aligned is not subject
    assert aligned.mask("tumor").geometry.is_compatible_with(
        aligned.image("T1").geometry
    )
    assert GEOMETRY_ALIGN_METADATA_KEY in (aligned.metadata or {})


@pytest.mark.unit
def test_anatomy_aware_geometry_restores_mask_direction_after_adopt() -> None:
    """Same-shape adopt keeps image index grid but restores mask direction."""
    subject = _mismatched_mask_subject()
    image = subject.image("T1")
    source_direction = tuple(subject.mask("tumor").geometry.direction)
    aligned = align_mask_to_reference(
        subject.mask("tumor"),
        image,
        subject_id=subject.subject_id,
        roi_name="tumor",
        reference_label="T1",
    )
    assert aligned.geometry.is_compatible_with(image.geometry)
    restored = anatomy_aware_field_geometry(aligned)
    assert tuple(restored.shape) == tuple(image.geometry.shape)
    assert tuple(restored.direction) == source_direction
    assert not restored.is_compatible_with(image.geometry)


@pytest.mark.unit
def test_habitat_map_keeps_mask_direction_for_display(tmp_path: Path) -> None:
    """
    One-step NRRD output keeps its mask SI sign without changing labels.

    The input image's direction is identity while the source mask has
    ``z=-1``. The pipeline deliberately resolves that header-only conflict in
    favour of the labelled anatomy, so this regression covers both the
    in-memory map and the persisted user-facing artefact.

    Args:
        tmp_path: Isolated directory for the written NRRD file.
    """
    import SimpleITK as sitk

    from habit.adapters import DirectoryResultWriter
    from habit.habitat_model.assignment.nearest_centroid import NearestCentroidAssigner
    from habit.pipeline import voxel_units
    from habit.viz.orientation import resolve_display_geometry

    subject = _mismatched_mask_subject()
    source_direction = tuple(subject.mask("tumor").geometry.direction)
    field = RawVoxelFeatures(modalities=["T1"])(subject)
    units = voxel_units(field)
    assert tuple(units.geometry.direction) == source_direction
    habitat_map = NearestCentroidAssigner(
        make_model(n_habitats=1, feature_names=field.feature_names)
    )(units)
    assert tuple(habitat_map.geometry.direction) == source_direction
    original_labels = habitat_map.label_array.copy()
    destination = DirectoryResultWriter(tmp_path).write_habitat_map(habitat_map)
    assert destination is not None
    written = sitk.ReadImage(destination)
    assert tuple(written.GetDirection()) == source_direction
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(written),
        original_labels,
    )
    resolved, _spacing = resolve_display_geometry(
        subject.image("T1"), habitat_map
    )
    assert resolved == source_direction
