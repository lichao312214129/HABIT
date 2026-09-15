# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License as distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Every documented Data In loader must yield a Cohort that habitat recipes can fit."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pytest

from habit.adapters import FileImageRef
from habit.contracts import (
    ArrayImageRef,
    Cohort,
    Geometry,
    ImageVolume,
    MaskVolume,
    Subject,
    cohort_from_directory,
)
from habit.datasets import make_synthetic_cohort
from habit.image import read_image, read_mask
from habit.recipes import one_step_habitat, two_step_habitat

sitk = pytest.importorskip("SimpleITK")

MODALITY = "T1"
ROI = "tumor"
SHAPE: Tuple[int, int, int] = (16, 16, 12)
N_HABITATS = 3
N_SUPERVOXELS = 8
Loader = Callable[[Path], Cohort]


def _source_cohort() -> Cohort:
    """Build a small clusterable in-memory cohort used as the disk source."""
    return make_synthetic_cohort(
        n_subjects=2,
        modalities=(MODALITY,),
        shape=SHAPE,
        n_subregions=3,
        rng=0,
    )


def _write_habit_tree(root: Path, cohort: Cohort) -> Path:
    """Write ``images/<id>/<mod>/`` and ``masks/<id>/<roi>/`` as ``.nii.gz``."""
    for subject in cohort:
        image_path = root / "images" / subject.subject_id / MODALITY / f"{MODALITY}.nii.gz"
        mask_path = root / "masks" / subject.subject_id / ROI / f"{ROI}.nii.gz"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(subject.image(MODALITY).to_sitk(), str(image_path))
        sitk.WriteImage(subject.mask(ROI).to_sitk(), str(mask_path))
    return root


def _pair_paths(root: Path, subject_id: str) -> Tuple[Path, Path]:
    """Return the image and mask NIfTI paths for one subject in ``root``."""
    image_path = root / "images" / subject_id / MODALITY / f"{MODALITY}.nii.gz"
    mask_path = root / "masks" / subject_id / ROI / f"{ROI}.nii.gz"
    return image_path, mask_path


def _subject_ids(root: Path) -> Tuple[str, ...]:
    """Return sorted subject ids present under ``root/images``."""
    return tuple(sorted(path.name for path in (root / "images").iterdir() if path.is_dir()))


def load_directory(root: Path) -> Cohort:
    """Data In route: HABIT directory tree."""
    return cohort_from_directory(root, modalities=(MODALITY,), roi=ROI, name="directory")


def load_simpleitk(root: Path) -> Cohort:
    """Data In route: SimpleITK.ReadImage + ImageVolume.from_sitk."""
    subjects = []
    for subject_id in _subject_ids(root):
        image_path, mask_path = _pair_paths(root, subject_id)
        volume = ImageVolume.from_sitk(sitk.ReadImage(str(image_path)), modality=MODALITY)
        mask = MaskVolume.from_sitk(sitk.ReadImage(str(mask_path)), modality=ROI)
        subjects.append(
            Subject(
                subject_id=subject_id,
                images={MODALITY: volume},
                masks={ROI: mask},
            )
        )
    return Cohort(subjects, name="simpleitk")


def load_numpy(root: Path) -> Cohort:
    """Data In route: NumPy arrays + ArrayImageRef + Geometry."""
    subjects = []
    for subject_id in _subject_ids(root):
        image_path, mask_path = _pair_paths(root, subject_id)
        sitk_image = sitk.ReadImage(str(image_path))
        sitk_mask = sitk.ReadImage(str(mask_path))
        array = sitk.GetArrayFromImage(sitk_image)
        mask = np.asarray(sitk.GetArrayFromImage(sitk_mask), dtype=np.int32)
        geometry = Geometry.from_array(
            array.shape,
            spacing=tuple(sitk_image.GetSpacing()),
            origin=tuple(sitk_image.GetOrigin()),
            direction=tuple(sitk_image.GetDirection()),
        )
        subjects.append(
            Subject(
                subject_id=subject_id,
                images={MODALITY: ArrayImageRef(array=array, geometry=geometry)},
                masks={ROI: ArrayImageRef(array=mask, geometry=geometry)},
            )
        )
    return Cohort(subjects, name="numpy")


def load_read_image(root: Path) -> Cohort:
    """Data In route: habit.image.read_image / read_mask wrapped as Subject."""
    subjects = []
    for subject_id in _subject_ids(root):
        image_path, mask_path = _pair_paths(root, subject_id)
        subjects.append(
            Subject(
                subject_id=subject_id,
                images={MODALITY: read_image(image_path, modality=MODALITY)},
                masks={ROI: read_mask(mask_path)},
            )
        )
    return Cohort(subjects, name="read_image")


def load_file_image_ref(root: Path) -> Cohort:
    """Data In route: lazy FileImageRef on a Subject."""
    subjects = []
    for subject_id in _subject_ids(root):
        image_path, mask_path = _pair_paths(root, subject_id)
        subjects.append(
            Subject(
                subject_id=subject_id,
                images={
                    MODALITY: FileImageRef(
                        image_path, is_mask=False, role_name=MODALITY
                    )
                },
                masks={ROI: FileImageRef(mask_path, is_mask=True, role_name=ROI)},
            )
        )
    return Cohort(subjects, name="file_image_ref")


LOADERS: Dict[str, Loader] = {
    "directory": load_directory,
    "simpleitk": load_simpleitk,
    "numpy": load_numpy,
    "read_image": load_read_image,
    "file_image_ref": load_file_image_ref,
}


def _assert_fit_ok(result: object, n_subjects: int) -> None:
    """Require a completed study with one labelled map per subject."""
    habitat_maps = getattr(result, "habitat_maps")
    assert len(habitat_maps) == n_subjects
    for habitat_map in habitat_maps:
        labels = np.unique(np.asarray(habitat_map.label_array))
        foreground = labels[labels > 0]
        assert habitat_map.label_array.shape == SHAPE
        assert foreground.size >= 1


@pytest.fixture()
def nifti_tree(tmp_path: Path) -> Path:
    """Write the synthetic cohort once as a HABIT NIfTI tree."""
    return _write_habit_tree(tmp_path / "study", _source_cohort())


@pytest.mark.unit
@pytest.mark.parametrize("loader_name", list(LOADERS))
def test_data_in_loader_fits_one_step_and_two_step(
    nifti_tree: Path, loader_name: str
) -> None:
    """Each documented Data In loader must survive one-step and two-step fit."""
    cohort = LOADERS[loader_name](nifti_tree)
    assert list(cohort.subject_ids) == ["subj001", "subj002"]

    one_step = one_step_habitat(
        modalities=(MODALITY,),
        n_habitats=N_HABITATS,
        random_seed=0,
        roi=ROI,
    ).fit_predict(cohort)
    _assert_fit_ok(one_step, n_subjects=2)
    assert one_step.subject_models

    two_step = two_step_habitat(
        modalities=(MODALITY,),
        n_supervoxels=N_SUPERVOXELS,
        n_habitats=N_HABITATS,
        random_seed=0,
        roi=ROI,
    ).fit_predict(cohort)
    _assert_fit_ok(two_step, n_subjects=2)
    assert two_step.habitat_model is not None


@pytest.mark.unit
def test_data_in_loaders_agree_on_one_step_maps(nifti_tree: Path) -> None:
    """Same voxels + same seed must yield the same one-step habitat labels."""
    maps: Dict[str, Dict[str, np.ndarray]] = {}
    for loader_name, loader in LOADERS.items():
        result = one_step_habitat(
            modalities=(MODALITY,),
            n_habitats=N_HABITATS,
            random_seed=0,
            roi=ROI,
        ).fit_predict(loader(nifti_tree))
        maps[loader_name] = {
            habitat_map.subject_id: np.asarray(habitat_map.label_array)
            for habitat_map in result.habitat_maps
        }
    baseline = maps["directory"]
    for loader_name, got in maps.items():
        for subject_id, expected in baseline.items():
            np.testing.assert_array_equal(
                got[subject_id],
                expected,
                err_msg=f"{loader_name} vs directory, {subject_id}",
            )
