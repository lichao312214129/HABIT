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
"""Shared synthetic fixtures for the L3 domain tests.

Everything here is built from in-memory NumPy arrays: the cloud constraint
forbids processing real imaging data, and small synthetic volumes exercise
the full protocol chain deterministically.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from habit.contracts import (
    ArrayImageRef,
    CohortFingerprint,
    Geometry,
    HabitatMap,
    HabitatModel,
    Provenance,
    Subject,
    Supervoxelization,
    VoxelFeatureField,
)


def provenance() -> Provenance:
    """Return a root provenance for synthetic fixtures."""
    return Provenance.source("domain_test")


def make_subject(
    subject_id: str,
    *,
    shape: Tuple[int, int, int] = (6, 6, 6),
    seed: int = 0,
    modalities: Sequence[str] = ("T1",),
) -> Subject:
    """
    Build a synthetic subject with two spatially separated intensity blobs.

    The first octant holds low intensities and the last octant holds high
    intensities, so voxel features form two obvious clusters.
    """
    rng = np.random.RandomState(seed)
    geometry = Geometry.from_array(shape)
    low = tuple(s // 4 for s in shape)
    high = tuple(3 * s // 4 for s in shape)
    images = {}
    for offset, modality in enumerate(modalities):
        array = np.zeros(shape, dtype=np.float64)
        array[: low[0] + 2, : low[1] + 2, : low[2] + 2] = 1.0
        array[high[0] - 2 :, high[1] - 2 :, high[2] - 2 :] = 10.0
        array += rng.normal(scale=0.01, size=shape) + offset
        images[modality] = ArrayImageRef(array=array, geometry=geometry)
    mask = np.zeros(shape, dtype=np.int32)
    mask[1:-1, 1:-1, 1:-1] = 1
    return Subject(
        subject_id=subject_id,
        images=images,
        masks={"tumor": ArrayImageRef(array=mask, geometry=geometry)},
    )


def make_field(
    subject_id: str = "P1",
    *,
    n_voxels: int = 8,
    two_blobs: bool = True,
) -> VoxelFeatureField:
    """Build a synthetic voxel feature field on a small cubic grid."""
    side = int(np.ceil(n_voxels ** (1 / 3))) + 1
    geometry = Geometry.from_array((side, side, side))
    index = np.array(
        [(z, y, x) for z in range(side) for y in range(side) for x in range(side)][
            :n_voxels
        ]
    )
    if two_blobs:
        half = n_voxels // 2
        values = np.vstack(
            [
                np.full((half, 2), [0.0, 0.0]),
                np.full((n_voxels - half, 2), [10.0, 10.0]),
            ]
        )
    else:
        values = np.arange(n_voxels * 2, dtype=np.float64).reshape(n_voxels, 2)
    return VoxelFeatureField(
        subject_id=subject_id,
        feature_names=("f1", "f2"),
        values=values,
        voxel_index=index,
        geometry=geometry,
        provenance=provenance(),
    )


def make_supervoxelization(
    subject_id: str,
    features: pd.DataFrame,
    *,
    shape: Tuple[int, int, int] = (4, 4, 4),
) -> Supervoxelization:
    """Build a supervoxelization whose labels match the feature index."""
    labels = np.zeros(shape, dtype=np.int32)
    flat = labels.ravel()
    unit_ids = features.index.to_numpy()
    flat[: len(unit_ids)] = unit_ids.astype(np.int32)
    return Supervoxelization(
        subject_id=subject_id,
        label_array=labels,
        features=features,
        geometry=Geometry.from_array(shape),
        provenance=provenance(),
    )


def two_cluster_units(
    subject_ids: Sequence[str] = ("P1", "P2", "P3"),
    *,
    supervoxels_per_subject: int = 4,
) -> Tuple[Supervoxelization, ...]:
    """Build units whose pooled features form two well-separated blobs."""
    units = []
    for offset, subject_id in enumerate(subject_ids):
        rows = np.vstack(
            [
                np.full((supervoxels_per_subject // 2, 2), [0.0, 0.0]) + offset * 0.1,
                np.full((supervoxels_per_subject // 2, 2), [10.0, 10.0]) + offset * 0.1,
            ]
        )
        features = pd.DataFrame(rows, columns=["f1", "f2"])
        features.index = pd.Index(
            np.arange(1, len(features) + 1), name="supervoxel"
        )
        units.append(make_supervoxelization(subject_id, features))
    return tuple(units)


def make_model(
    *,
    n_habitats: int = 2,
    feature_names: Tuple[str, ...] = ("f1", "f2"),
) -> HabitatModel:
    """Build a habitat model with well-separated centroids."""
    centroids = np.vstack(
        [
            np.zeros((1, len(feature_names))),
            np.full((n_habitats - 1, len(feature_names)), 10.0),
        ]
    )
    return HabitatModel(
        model_id="test-model",
        n_habitats=n_habitats,
        feature_names=feature_names,
        centroids=centroids,
        preprocessing_state={},
        spec_payload={"habitat_model_fitter": {"name": "kmeans", "params": {}}},
        cohort_fingerprint=CohortFingerprint(
            n_subjects=3,
            modalities=("T1",),
            subject_id_digest="0" * 64,
        ),
        provenance=provenance(),
    )


def make_habitat_map(subject_id: str = "P1") -> HabitatMap:
    """Build a habitat map with two adjacent habitat blocks (2x2x2 each)."""
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1
    labels[2:4, 0:2, 0:2] = 2
    return HabitatMap(
        subject_id=subject_id,
        label_array=labels,
        geometry=Geometry.from_array((4, 4, 4)),
        model_id="test-model",
        habitat_ids=(1, 2),
        provenance=provenance(),
    )


def make_feature_table(
    subject_ids: Sequence[str] = tuple(f"S{i:02d}" for i in range(40)),
    *,
    seed: int = 0,
    n_noise: int = 3,
    non_negative: bool = False,
    constant_column: bool = False,
    outcome: bool = True,
) -> "FeatureTable":
    """
    Build a synthetic binary-outcome feature table for the table-ML tests.

    The ``signal`` column separates the two outcome classes; the remaining
    ``noise{i}`` columns are pure noise, so supervised selectors and
    classifiers have a deterministic correct answer to find.
    """
    from habit.contracts import BinaryOutcome, FeatureTable

    rng = np.random.RandomState(seed)
    n = len(subject_ids)
    y = (np.arange(n) % 2) if outcome else None
    data: dict = {"subject": list(subject_ids)}
    signal = rng.normal(loc=0.0, scale=0.5, size=n) + (
        0.0 if y is None else np.where(y == 1, 2.0, 0.0)
    )
    data["signal"] = signal
    for i in range(n_noise):
        data[f"noise{i}"] = rng.normal(size=n)
    if constant_column:
        data["constant"] = np.full(n, 3.14)
    if non_negative:
        for key in [k for k in data if k != "subject"]:
            data[key] = np.abs(data[key])
    feature_columns = tuple(k for k in data if k != "subject")
    if outcome:
        data["y"] = y
    return FeatureTable(
        frame=pd.DataFrame(data),
        id_columns=("subject",),
        feature_columns=feature_columns,
        outcome=BinaryOutcome("y") if outcome else None,
        provenance=provenance(),
    )


@pytest.fixture
def subject() -> Subject:
    """Single-modality synthetic subject."""
    return make_subject("P1")
