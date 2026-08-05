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
"""
The persistence contract of ``StudyResult`` and ``DirectoryResultWriter``.

Two separable promises are checked here, because in v1.0 they became two
different objects:

* ``StudyResult`` decides *what* is persisted and nothing else -- it must hand
  every artefact it holds to any writer, including writers that never touch a
  filesystem. That is what allows HABIT to run inside a service with no output
  directory.
* ``DirectoryResultWriter`` owns the v0.1 directory layout. Its filenames are
  what users' downstream scripts glob for, so they are pinned literally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from habit.contracts import (
    CohortFingerprint,
    FeatureTable,
    Geometry,
    HabitatMap,
    HabitatModel,
    Provenance,
    RunManifest,
)
from habit.recipes import StudyResult


def _provenance() -> Provenance:
    """Return a minimal provenance record."""
    return Provenance.source("study")


def _model(model_id: str = "m") -> HabitatModel:
    """Return a one-habitat model with a single feature."""
    return HabitatModel(
        model_id=model_id,
        n_habitats=1,
        feature_names=("f1",),
        centroids=np.zeros((1, 1)),
        preprocessing_state={},
        spec_payload={},
        cohort_fingerprint=CohortFingerprint(
            n_subjects=1, modalities=("T1",), subject_id_digest="d"
        ),
        provenance=_provenance(),
    )


def _habitat_map(subject_id: str = "a") -> HabitatMap:
    """Return a 2x2x2 label map with one habitat."""
    return HabitatMap(
        subject_id=subject_id,
        label_array=np.ones((2, 2, 2), dtype=np.int32),
        geometry=Geometry.from_array((2, 2, 2), spacing=(1.5, 1.5, 3.0)),
        model_id="m",
        habitat_ids=(1,),
        provenance=_provenance(),
    )


def _features() -> FeatureTable:
    """Return a one-row habitat feature table."""
    return FeatureTable(
        frame=pd.DataFrame({"subject": ["a"], "f1": [1.0]}),
        id_columns=("subject",),
        feature_columns=("f1",),
        provenance=_provenance(),
    )


def _study(**overrides: Any) -> StudyResult:
    """Return a study result with every artefact populated."""
    fields: Dict[str, Any] = {
        "habitat_model": _model(),
        "pipeline": object(),
        "features": _features(),
        "habitat_maps": (_habitat_map(),),
        "manifest": RunManifest(
            spec_payload={"design": "two_step"},
            provenance=_provenance(),
            subject_outcomes={"a": "success"},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
        ),
    }
    fields.update(overrides)
    return StudyResult(**fields)


class _RecordingWriter:
    """A ``ResultWriter`` that records calls instead of writing anything."""

    def __init__(self) -> None:
        self.habitat_maps: List[str] = []
        self.tables: List[str] = []
        self.models: List[str] = []
        self.manifests: List[str] = []

    def write_habitat_map(self, habitat_map: HabitatMap) -> Optional[str]:
        """Record a habitat map hand-off."""
        self.habitat_maps.append(habitat_map.subject_id)
        return None

    def write_feature_table(self, table: FeatureTable, name: str) -> Optional[str]:
        """Record a feature table hand-off."""
        self.tables.append(name)
        return None

    def write_habitat_model(self, model: HabitatModel) -> Optional[str]:
        """Record a model hand-off."""
        self.models.append(model.model_id)
        return None

    def write_manifest(self, manifest: RunManifest) -> Optional[str]:
        """Record a manifest hand-off."""
        self.manifests.append(str(manifest.spec_payload.get("design")))
        return None


@pytest.mark.unit
def test_write_hands_every_artefact_to_the_writer() -> None:
    """Every artefact the study holds reaches the writer, and nothing else."""
    writer = _RecordingWriter()
    _study().write(writer)

    assert writer.habitat_maps == ["a"]
    assert writer.models == ["m"]
    assert writer.manifests == ["two_step"]
    assert writer.tables == ["habitat_features"]


@pytest.mark.unit
def test_write_needs_no_filesystem() -> None:
    """
    A study can be consumed without any directory existing.

    This is the point of the writer protocol: an embedder with an object store
    or an in-process consumer must not be forced through a temp directory.
    """
    writer = _RecordingWriter()
    _study(habitat_model=None).write(writer)

    assert writer.models == [], "a study without a model must not invent one"
    assert writer.habitat_maps and writer.manifests


@pytest.mark.unit
def test_save_writes_the_v01_directory_layout(tmp_path: Path) -> None:
    """
    ``save`` reproduces the filenames v0.1 users' scripts depend on.

    Args:
        tmp_path: Destination root.
    """
    out = _study().save(tmp_path / "study")

    assert (out / "habitat_model.habitatmodel").is_file()
    assert (out / "habitat_features.csv").is_file()
    assert (out / "run_manifest.json").is_file()
    assert (out / "a_habitats.nrrd").is_file()
    assert HabitatModel.load(out / "habitat_model.habitatmodel").model_id == "m"


@pytest.mark.unit
def test_written_label_map_keeps_its_geometry(tmp_path: Path) -> None:
    """
    A habitat map is written on the grid it was computed on.

    A label map that lost its spacing still opens in a viewer and still looks
    like a habitat map -- it is simply in the wrong place, which is why this
    is asserted rather than assumed.

    Args:
        tmp_path: Destination root.
    """
    import SimpleITK as sitk

    out = _study().save(tmp_path / "study")
    image = sitk.ReadImage(str(out / "a_habitats.nrrd"))
    geometry = _habitat_map().geometry

    assert tuple(image.GetSpacing()) == tuple(geometry.spacing)
    assert tuple(image.GetOrigin()) == tuple(geometry.origin)
    assert tuple(image.GetDirection()) == tuple(geometry.direction)
    assert np.array_equal(
        sitk.GetArrayFromImage(image), _habitat_map().label_array
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "map_format, suffix",
    [
        ("nrrd", ".nrrd"),
        ("nii.gz", ".nii.gz"),
        (".nii", ".nii"),
        ("mha", ".mha"),
    ],
)
def test_save_writes_label_maps_in_requested_format(
    tmp_path: Path, map_format: str, suffix: str
) -> None:
    """
    ``map_format`` selects the label-map container without changing the stem.

    Args:
        tmp_path: Destination root.
        map_format: Format string accepted by ``StudyResult.save``.
        suffix: Expected file extension including the leading dot.
    """
    import SimpleITK as sitk

    out = _study().save(tmp_path / "study", map_format=map_format)
    path = out / f"a_habitats{suffix}"
    assert path.is_file()
    image = sitk.ReadImage(str(path))
    assert np.array_equal(
        sitk.GetArrayFromImage(image), _habitat_map().label_array
    )


@pytest.mark.unit
def test_unsupported_map_format_raises() -> None:
    """Unknown ``map_format`` values fail loudly at writer construction."""
    from habit.adapters import DirectoryResultWriter
    from habit.exceptions import HABITAPIError

    with pytest.raises(HABITAPIError, match="Unsupported map_format"):
        DirectoryResultWriter("unused", map_format="dicom")


@pytest.mark.unit
def test_writer_creates_nothing_until_it_writes(tmp_path: Path) -> None:
    """
    Constructing a writer has no side effect.

    Building a writer and then deciding not to use it must not litter the
    filesystem with empty directories.

    Args:
        tmp_path: Destination root.
    """
    from habit.adapters import DirectoryResultWriter

    destination = tmp_path / "unused"
    DirectoryResultWriter(destination)

    assert not destination.exists()


@pytest.mark.unit
def test_subject_models_stay_in_memory() -> None:
    """
    Per-subject definitions are held, not written.

    The one-step design produces one model per subject; the writer protocol
    persists a single study model, so these are deliberately kept in memory
    until a per-subject naming convention is agreed.
    """
    writer = _RecordingWriter()
    result = _study(habitat_model=None, subject_models={"a": _model("a")})
    result.write(writer)

    assert result.subject_models["a"].model_id == "a"
    assert writer.models == []
