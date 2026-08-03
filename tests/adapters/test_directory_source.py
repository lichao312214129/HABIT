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
"""Contract tests for DirectoryDataSource on a synthetic convention tree."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from habit.adapters import DirectoryDataSource
from habit.api.exceptions import DataFormatError
from habit.contracts import ImageVolume, MaskVolume, cohort_from_directory

sitk = pytest.importorskip("SimpleITK")


def _write_image(path: Path, array: np.ndarray, spacing=(1.0, 1.0, 2.0)) -> None:
    """Write one NIfTI file with explicit spacing into the tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


@pytest.fixture()
def convention_tree(tmp_path: Path) -> Path:
    """Build a small <root>/images|masks/<subject>/<modality>/<file> tree."""
    root = tmp_path / "study"
    for subject_id in ("subj_b", "subj_a"):
        for modality in ("T1", "T2"):
            _write_image(
                root / "images" / subject_id / modality / f"{modality}.nii.gz",
                np.full((3, 4, 5), 7.0, dtype=np.float32),
            )
        _write_image(
            root / "masks" / subject_id / "tumor" / "tumor.nii.gz",
            np.ones((3, 4, 5), dtype=np.uint8),
        )
    # A subject missing T2 is present in the tree but must be skipped.
    _write_image(
        root / "images" / "subj_incomplete" / "T1" / "T1.nii.gz",
        np.zeros((3, 4, 5), dtype=np.float32),
    )
    _write_image(
        root / "masks" / "subj_incomplete" / "tumor" / "tumor.nii.gz",
        np.ones((3, 4, 5), dtype=np.uint8),
    )
    return root


@pytest.mark.unit
def test_load_returns_sorted_lazy_cohort(convention_tree: Path) -> None:
    """load() yields sorted ids, skips incomplete subjects, stays lazy."""
    cohort = DirectoryDataSource(
        convention_tree, modalities=("T1", "T2"), roi="tumor", name="demo"
    ).load()

    assert cohort.subject_ids == ("subj_a", "subj_b")
    assert cohort.name == "demo"
    # The lazy reference must be small enough to cross a process boundary.
    assert len(pickle.dumps(cohort[0])) < 4096


@pytest.mark.unit
def test_materialisation_carries_full_geometry(convention_tree: Path) -> None:
    """image()/mask() return geometry-bound volumes read from disk."""
    cohort = DirectoryDataSource(
        convention_tree, modalities=("T1", "T2"), roi="tumor"
    ).load()
    subject = cohort[0]

    image = subject.image("T1")
    mask = subject.mask("tumor")

    assert isinstance(image, ImageVolume)
    assert isinstance(mask, MaskVolume)
    assert image.data.shape == (3, 4, 5)
    assert image.spacing == (1.0, 1.0, 2.0)
    assert np.allclose(image.data, 7.0)
    assert mask.roi_name == "tumor"
    assert image.geometry.is_compatible_with(mask.geometry)


@pytest.mark.unit
def test_cohort_from_directory_matches_data_source(convention_tree: Path) -> None:
    """The class/free-function shortcuts delegate to the adapter."""
    direct = DirectoryDataSource(
        convention_tree, modalities=("T1", "T2"), roi="tumor"
    ).load()
    via_class = cohort_from_directory(convention_tree, modalities=("T1", "T2"), roi="tumor")

    assert via_class.subject_ids == direct.subject_ids
    assert via_class.summarize().subject_id_digest == direct.summarize().subject_id_digest


@pytest.mark.unit
def test_missing_convention_folder_is_a_data_format_error(tmp_path: Path) -> None:
    """A missing images folder raises DataFormatError, not a bare OSError."""
    with pytest.raises(DataFormatError):
        DirectoryDataSource(tmp_path / "nowhere", modalities=("T1",), roi="tumor").load()


@pytest.mark.unit
def test_no_complete_subjects_is_a_data_format_error(tmp_path: Path) -> None:
    """An empty scan raises instead of returning a vacuous cohort."""
    root = tmp_path / "study"
    _write_image(
        root / "images" / "subj" / "T1" / "T1.nii.gz",
        np.zeros((2, 2, 2), dtype=np.float32),
    )
    with pytest.raises(DataFormatError, match="No complete subjects"):
        DirectoryDataSource(root, modalities=("T1",), roi="tumor").load()
