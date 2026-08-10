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
"""Gap tests for ``.habitatmodel`` save/load robustness on synthetic data."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import habit
from habit.contracts.habitat import HabitatModel
from habit.exceptions import CompatibilityError
from habit.recipes.study import Study
from habit.spec.specs import HabitatSpec, Spec
from tests.cloud_gaps.synth_data import MODALITIES, ROI

RANDOM_SEED: int = 42


def _tiny_two_step_spec() -> HabitatSpec:
    """
    Build a lightweight two-step spec for gap robustness tests.

    Returns:
        Habitat specification with ``n_supervoxels=20`` and ``max_habitats=4``.
    """
    return HabitatSpec(
        name="habitat_two_step",
        voxel_feature_extractor=Spec(
            name="raw",
            params={"modalities": list(MODALITIES)},
        ),
        supervoxelizer=Spec(
            name="kmeans",
            params={"n_supervoxels": 20, "max_iter": 100, "n_init": 3},
        ),
        habitat_model_fitter=Spec(
            name="kmeans",
            params={
                "min_habitats": 2,
                "max_habitats": 4,
                "validation": "elbow",
                "max_iter": 100,
                "n_init": 3,
            },
        ),
        habitat_assigner=Spec(name="nearest_centroid"),
        habitat_features=(Spec(name="volume"),),
        random_seed=RANDOM_SEED,
    )


@pytest.fixture(scope="module")
def trained_model(demo_data_root: Path) -> HabitatModel:
    """
    Fit a tiny two-step model on the synthetic demo cohort.

    Returns:
        Fitted :class:`~habit.contracts.habitat.HabitatModel`.
    """
    spec = _tiny_two_step_spec()
    cohort = habit.cohort_from_directory(
        demo_data_root,
        modalities=MODALITIES,
        roi=ROI,
        name="gap_test",
    )
    result = Study(spec=spec, design="two_step").fit_predict(cohort)
    assert result.habitat_model is not None
    return result.habitat_model


@pytest.mark.integration
def test_habitatmodel_save_load_roundtrip(
    trained_model: HabitatModel,
    tmp_path: Path,
) -> None:
    """Round-trip preserves centroids, feature names and model_id."""
    archive = trained_model.save(tmp_path / "model.habitatmodel")
    loaded = HabitatModel.load(archive)
    assert loaded.model_id == trained_model.model_id
    assert loaded.feature_names == trained_model.feature_names
    np.testing.assert_allclose(loaded.centroids, trained_model.centroids)


@pytest.mark.unit
def test_habitatmodel_truncated_file_raises_clear_error(
    trained_model: HabitatModel,
    tmp_path: Path,
) -> None:
    """Truncated bytes must not deserialize into a plausible HabitatModel."""
    valid = trained_model.save(tmp_path / "valid.habitatmodel")
    truncated = tmp_path / "truncated.habitatmodel"
    truncated.write_bytes(valid.read_bytes()[:128])
    with pytest.raises(CompatibilityError, match="habit.habitatmodel|not a"):
        HabitatModel.load(truncated)


@pytest.mark.unit
def test_habitatmodel_unsupported_format_version_raises(
    trained_model: HabitatModel,
    tmp_path: Path,
) -> None:
    """A future format version must fail with an explicit incompatibility error."""
    source = trained_model.save(tmp_path / "source.habitatmodel")
    tampered = tmp_path / "future.habitatmodel"
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        centroids = archive.read("arrays/centroids.npy")
    manifest["format_version"] = 99
    with zipfile.ZipFile(tampered, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays/centroids.npy", centroids)
    with pytest.raises(CompatibilityError, match="format version 99"):
        HabitatModel.load(tampered)


@pytest.mark.unit
def test_habitatmodel_wrong_format_name_raises(
    trained_model: HabitatModel,
    tmp_path: Path,
) -> None:
    """A tampered format identifier must not load as a valid HabitatModel."""
    source = trained_model.save(tmp_path / "source.habitatmodel")
    tampered = tmp_path / "wrong_format.habitatmodel"
    with zipfile.ZipFile(source) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        centroids = archive.read("arrays/centroids.npy")
    manifest["format"] = "foreign.format"
    with zipfile.ZipFile(tampered, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays/centroids.npy", centroids)
    with pytest.raises(CompatibilityError, match="foreign.format"):
        HabitatModel.load(tampered)
