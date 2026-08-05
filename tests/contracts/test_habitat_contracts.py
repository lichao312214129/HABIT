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
"""Contract tests for the habitat vocabulary and HabitatModel persistence."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from habit.api.exceptions import CompatibilityError, HABITAPIError
from habit.contracts import (
    CohortFingerprint,
    Geometry,
    HabitatMap,
    HabitatModel,
    Provenance,
    Supervoxelization,
    VoxelFeatureField,
)


def _provenance() -> Provenance:
    """Return a root provenance for contract fixtures."""
    return Provenance.source("contract_test")


def _field() -> VoxelFeatureField:
    """Build a minimal valid voxel feature field."""
    return VoxelFeatureField(
        subject_id="P1",
        feature_names=("raw_T1", "raw_T2"),
        values=np.ones((4, 2), dtype=np.float32),
        voxel_index=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        geometry=Geometry.from_array((2, 2, 2)),
        provenance=_provenance(),
    )


@pytest.mark.unit
def test_voxel_feature_field_invariants_enforced() -> None:
    """Row/column mismatches are rejected at construction."""
    with pytest.raises(HABITAPIError):
        VoxelFeatureField(
            subject_id="P1",
            feature_names=("a",),
            values=np.ones((4, 2)),
            voxel_index=np.zeros((4, 3)),
            geometry=Geometry.from_array((2, 2, 2)),
            provenance=_provenance(),
        )
    with pytest.raises(HABITAPIError):
        VoxelFeatureField(
            subject_id="P1",
            feature_names=("a",),
            values=np.ones((3, 1)),
            voxel_index=np.zeros((4, 3)),
            geometry=Geometry.from_array((2, 2, 2)),
            provenance=_provenance(),
        )


@pytest.mark.unit
def test_voxel_feature_field_to_frame() -> None:
    """to_frame exposes grid coordinates plus feature columns."""
    frame = _field().to_frame()
    assert list(frame.columns) == ["z", "y", "x", "raw_T1", "raw_T2"]
    assert len(frame) == 4


@pytest.mark.unit
def test_supervoxelization_and_habitat_map_hold_labels() -> None:
    """The partition and the label map keep their grids and provenance."""
    provenance = _provenance()
    unit = Supervoxelization(
        subject_id="P1",
        label_array=np.ones((2, 2, 2), dtype=np.int32),
        features=pd.DataFrame({"mean": [1.0]}),
        geometry=Geometry.from_array((2, 2, 2)),
        provenance=provenance,
    )
    habitat_map = HabitatMap(
        subject_id="P1",
        label_array=np.ones((2, 2, 2), dtype=np.int32),
        geometry=unit.geometry,
        model_id="model-1",
        habitat_ids=(1, 2),
        provenance=provenance,
    )
    assert unit.label_array.shape == (2, 2, 2)
    assert habitat_map.model_id == "model-1"
    assert habitat_map.habitat_ids == (1, 2)


def _model() -> HabitatModel:
    """Build a minimal valid habitat model."""
    return HabitatModel(
        model_id="model-1",
        n_habitats=2,
        feature_names=("f1", "f2"),
        centroids=np.array([[1.0, 2.0], [3.0, 4.0]]),
        preprocessing_state={"bin_edges": np.array([0.5, 1.5]), "scale": 2.0},
        spec_payload={"habitat_model_fitter": {"name": "kmeans", "params": {"n_clusters": 2}}},
        cohort_fingerprint=CohortFingerprint(
            n_subjects=2, modalities=("T1",), subject_id_digest="digest"
        ),
        provenance=_provenance(),
    )


@pytest.mark.unit
def test_habitat_model_validates_centroid_shape() -> None:
    """Centroid dimensions must match n_habitats and feature_names."""
    with pytest.raises(HABITAPIError):
        HabitatModel(
            model_id="bad",
            n_habitats=3,
            feature_names=("f1", "f2"),
            centroids=np.zeros((2, 2)),
            preprocessing_state={},
            spec_payload={},
            cohort_fingerprint=CohortFingerprint(
                n_subjects=1, modalities=(), subject_id_digest="d"
            ),
            provenance=_provenance(),
        )


@pytest.mark.unit
def test_habitat_model_summary_mentions_key_facts() -> None:
    """The model card names the id, habitats, features, cohort and software."""
    text = _model().summary()
    assert "model-1" in text
    assert "habitats           : 2" in text
    assert "f1, f2" in text
    assert "n=2" in text
    assert "habit version" in text


@pytest.mark.unit
def test_habitat_model_save_load_roundtrip(tmp_path: Path) -> None:
    """save/load preserves every field through the versioned format."""
    model = _model()
    written = model.save(tmp_path / "model.habitatmodel")

    assert written.name == "model.habitatmodel"
    loaded = HabitatModel.load(written)
    assert loaded.model_id == model.model_id
    assert loaded.n_habitats == model.n_habitats
    assert loaded.feature_names == model.feature_names
    assert np.allclose(loaded.centroids, model.centroids)
    assert loaded.spec_payload == model.spec_payload
    assert loaded.preprocessing_state["scale"] == 2.0
    assert np.allclose(loaded.preprocessing_state["bin_edges"], [0.5, 1.5])
    assert loaded.cohort_fingerprint.n_subjects == 2
    assert loaded.provenance.produced_by == "contract_test"


@pytest.mark.unit
def test_habitat_model_file_is_self_describing(tmp_path: Path) -> None:
    """The file carries format name, format version and producing version."""
    written = _model().save(tmp_path / "model.habitatmodel")
    with zipfile.ZipFile(written) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["format"] == "habit.habitatmodel"
    assert manifest["format_version"] == 1
    assert "habit_version" in manifest


@pytest.mark.unit
def test_habitat_model_load_rejects_foreign_and_future_files(tmp_path: Path) -> None:
    """Non-format files and future format versions fail with clear guidance."""
    foreign = tmp_path / "legacy.pkl"
    foreign.write_bytes(b"not a zip")
    with pytest.raises(CompatibilityError, match="habitat_pipeline"):
        HabitatModel.load(foreign)

    written = _model().save(tmp_path / "model.habitatmodel")
    doctored = tmp_path / "future.habitatmodel"
    with zipfile.ZipFile(written) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        centroids = archive.read("arrays/centroids.npy")
    manifest["format_version"] = 99
    with zipfile.ZipFile(doctored, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays/centroids.npy", centroids)
    with pytest.raises(CompatibilityError, match="format version 99"):
        HabitatModel.load(doctored)


@pytest.mark.unit
def test_provenance_derive_chains_inputs() -> None:
    """derive() links the new record to its input and stamps the environment."""
    root = Provenance.source("raw_image")
    derived = root.derive(produced_by="supervoxelizer.slic", spec_fingerprint="abc", random_seed=42)

    assert derived.inputs == (root,)
    assert derived.produced_by == "supervoxelizer.slic"
    assert derived.random_seed == 42
    assert derived.software["habit"]
    assert derived.created_at


@pytest.mark.unit
def test_provenance_derive_inherits_random_seed_by_default() -> None:
    """
    Omitting random_seed keeps the parent's seed; None clears it explicitly.

    Deterministic mid-pipeline steps must not erase the seed that defined the
    scientific result, which is what made HabitatModel.summary() report None
    after cohort-level preprocessing was attached.
    """
    seeded = Provenance.source("fitter").derive(
        produced_by="habitat_model_fitter.kmeans",
        spec_fingerprint="fit",
        random_seed=42,
    )
    inherited = seeded.derive(
        produced_by="habitat_model_fitter.kmeans+cohort_preprocessing",
        spec_fingerprint="chain",
    )
    cleared = seeded.derive(
        produced_by="deterministic_clear",
        spec_fingerprint="clear",
        random_seed=None,
    )
    replaced = seeded.derive(
        produced_by="reseeded",
        spec_fingerprint="new",
        random_seed=7,
    )

    assert inherited.random_seed == 42
    assert cleared.random_seed is None
    assert replaced.random_seed == 7


@pytest.mark.unit
def test_with_cohort_preprocessing_preserves_random_seed() -> None:
    """Attaching a cohort preprocessing chain must not wipe the fitter seed."""
    model = HabitatModel(
        model_id="kmeans-abc",
        n_habitats=2,
        feature_names=("f1", "f2"),
        centroids=np.array([[1.0, 2.0], [3.0, 4.0]]),
        preprocessing_state={"inertia": 1.0},
        spec_payload={"habitat_model_fitter": {"name": "kmeans", "params": {}}},
        cohort_fingerprint=CohortFingerprint(
            n_subjects=2, modalities=("T1",), subject_id_digest="digest"
        ),
        provenance=Provenance(
            produced_by="habitat_model_fitter.kmeans",
            spec_fingerprint="fit",
            random_seed=42,
        ),
    )

    rebound = model.with_cohort_preprocessing(
        state={"methods": []},
        spec_payload={"name": "cohort_feature_preprocessor", "params": {}},
    )

    assert rebound.provenance.random_seed == 42
    assert "cohort_preprocessing" in rebound.provenance.produced_by
    assert "random seed        : 42" in rebound.summary()
    assert "cohort_feature_preprocessor" in rebound.preprocessing_state
