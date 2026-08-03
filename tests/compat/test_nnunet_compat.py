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
"""Contract tests for ``habit.compat.nnunet.NnUNetDataSource``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from habit.adapters import FileImageRef
from habit.api.exceptions import DataFormatError
from habit.compat.nnunet import NnUNetDataSource

sitk = pytest.importorskip("SimpleITK")


def _write_image(path: Path, array: np.ndarray) -> None:
    """Write one small NIfTI file into the dataset tree."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 2.0))
    sitk.WriteImage(image, str(path))


def _labels_array() -> np.ndarray:
    """Multi-label segmentation: background 0, tumor 1, edema 2."""
    labels = np.zeros((3, 4, 5), dtype=np.uint8)
    labels[1, 1, 1] = 1
    labels[1, 1, 2] = 1
    labels[2, 2, 2] = 2
    return labels


def _write_case(root: Path, case: str, *, channels: int = 2, label: bool = True) -> None:
    """Write the channel files (and optionally the label file) of one case."""
    for index in range(channels):
        _write_image(
            root / "imagesTr" / f"{case}_{index:04d}.nii.gz",
            np.full((3, 4, 5), float(index + 1), dtype=np.float32),
        )
    if label:
        _write_image(root / "labelsTr" / f"{case}.nii.gz", _labels_array())


def _write_dataset_json(root: Path, **overrides) -> None:
    """Write an nnU-Net v2 dataset.json, overridable per test."""
    payload = {
        "channel_names": {"0": "T1", "1": "T2"},
        "labels": {"background": 0, "tumor": 1, "edema": 2},
        "training": [
            {
                "image": f"./imagesTr/{case}_0000.nii.gz",
                "label": f"./labelsTr/{case}.nii.gz",
            }
            for case in ("case_b", "case_a")
        ],
        **overrides,
    }
    (root / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def dataset_tree(tmp_path: Path) -> Path:
    """A complete two-case nnU-Net v2 dataset."""
    root = tmp_path / "Dataset001_Test"
    _write_case(root, "case_b")
    _write_case(root, "case_a")
    _write_dataset_json(root)
    return root


@pytest.mark.unit
def test_load_builds_sorted_cohort_with_named_channels(dataset_tree: Path) -> None:
    """Cases come sorted; channel names become modality roles."""
    cohort = NnUNetDataSource(dataset_tree, roi_label=1).load()

    assert [subject.subject_id for subject in cohort] == ["case_a", "case_b"]
    assert cohort.name == "Dataset001_Test"
    subject = cohort[0]
    assert set(subject.images) == {"T1", "T2"}
    # The label name is resolved through dataset.json's labels mapping.
    assert set(subject.masks) == {"tumor"}
    # References stay lazy until a voxel is actually requested.
    assert isinstance(subject.images["T1"], FileImageRef)


@pytest.mark.unit
def test_roi_label_binarisation_modes(dataset_tree: Path) -> None:
    """Integer, named and union roi_label definitions binarise correctly."""
    by_int = NnUNetDataSource(dataset_tree, roi_label=1).load()[0]
    mask = np.asarray(by_int.mask("tumor").data)
    assert mask.sum() == 2  # exactly the two voxels labelled 1

    by_name = NnUNetDataSource(dataset_tree, roi_label="edema").load()[0]
    assert set(by_name.masks) == {"edema"}
    assert np.asarray(by_name.mask("edema").data).sum() == 1

    union = NnUNetDataSource(dataset_tree, roi_label=[1, 2]).load()[0]
    assert set(union.masks) == {"roi_1_2"}
    assert np.asarray(union.mask("roi_1_2").data).sum() == 3

    renamed = NnUNetDataSource(dataset_tree, roi_label=1, roi_name="GTV").load()[0]
    assert set(renamed.masks) == {"GTV"}
    # Binarised masks carry the header geometry like any HABIT mask.
    assert renamed.mask("GTV").spacing == (1.0, 1.0, 2.0)


@pytest.mark.unit
def test_v1_modality_key_and_missing_dataset_json(tmp_path: Path) -> None:
    """The v1 ``modality`` mapping is honoured; absent json falls back."""
    v1 = tmp_path / "Dataset002_Legacy"
    _write_case(v1, "case_x", channels=1)
    _write_dataset_json(
        v1,
        channel_names=None,  # deleted below
        labels={"background": 0, "lesion": 1},
        training=None,
    )
    payload = json.loads((v1 / "dataset.json").read_text(encoding="utf-8"))
    del payload["channel_names"], payload["training"]
    payload["modality"] = {"0": {"name": "CT"}}
    (v1 / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")

    cohort = NnUNetDataSource(v1, roi_label="lesion").load()
    assert set(cohort[0].images) == {"CT"}

    bare = tmp_path / "Dataset003_Bare"
    _write_case(bare, "case_y", channels=2)
    cohort = NnUNetDataSource(bare, roi_label=1).load()
    assert set(cohort[0].images) == {"channel_0000", "channel_0001"}
    assert set(cohort[0].masks) == {"roi_1"}


@pytest.mark.unit
def test_incomplete_cases_are_skipped_with_a_warning(
    dataset_tree: Path, capsys: pytest.CaptureFixture
) -> None:
    """A case without its label file never blocks the cohort."""
    (dataset_tree / "labelsTr" / "case_b.nii.gz").unlink()
    cohort = NnUNetDataSource(dataset_tree, roi_label=1).load()
    assert [subject.subject_id for subject in cohort] == ["case_a"]
    assert "case_b" in capsys.readouterr().out


@pytest.mark.unit
def test_error_paths_are_data_format_errors(tmp_path: Path) -> None:
    """Missing folders, unknown labels and empty scans fail explicitly."""
    with pytest.raises(DataFormatError, match="imagesTr"):
        NnUNetDataSource(tmp_path / "missing").load()

    root = tmp_path / "Dataset004_Errors"
    _write_case(root, "case_a")
    _write_dataset_json(root)
    with pytest.raises(DataFormatError, match="not declared in dataset.json"):
        NnUNetDataSource(root, roi_label="necrosis").load()

    (root / "labelsTr" / "case_a.nii.gz").unlink()
    with pytest.raises(DataFormatError, match="No complete nnU-Net cases"):
        NnUNetDataSource(root, roi_label=1).load()

    bad = tmp_path / "Dataset005_BadJson"
    _write_case(bad, "case_a")
    (bad / "dataset.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(DataFormatError, match="Invalid dataset.json"):
        NnUNetDataSource(bad).load()
