"""Tests for ``get_image_and_mask_paths_dicom_subject_roots`` (one-folder-per-subject layout)."""

from __future__ import annotations

import pytest

from habit.utils.dicom_preprocess_path_utils import (
    get_dicom_subject_root_paths,
    get_image_and_mask_paths_dicom_subject_roots,
)


def test_get_dicom_subject_root_paths_matches_single_key_layout(tmp_path) -> None:
    (tmp_path / "subj_a").mkdir()
    (tmp_path / "subj_b").mkdir()
    flat = get_dicom_subject_root_paths(str(tmp_path))
    assert flat == {
        "subj_a": str(tmp_path / "subj_a"),
        "subj_b": str(tmp_path / "subj_b"),
    }


def test_discovers_subject_folders(tmp_path) -> None:
    (tmp_path / "subj_a").mkdir()
    (tmp_path / "subj_b").mkdir()
    images_paths, mask_paths = get_image_and_mask_paths_dicom_subject_roots(
        str(tmp_path),
        modality_keys=["dicom"],
        auto_select_first_file=False,
    )
    assert set(images_paths.keys()) == {"subj_a", "subj_b"}
    assert images_paths["subj_a"]["dicom"] == str(tmp_path / "subj_a")
    assert mask_paths == {}


def test_empty_modality_keys_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="modality_keys"):
        get_image_and_mask_paths_dicom_subject_roots(str(tmp_path), modality_keys=[])


def test_no_subject_dirs_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="No subject subdirectories"):
        get_image_and_mask_paths_dicom_subject_roots(str(tmp_path), modality_keys=["dicom"])
