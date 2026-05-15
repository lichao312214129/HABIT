"""
Path discovery for DICOM preprocessing (sort / conversion) without importing ``habit.core``.

Keeping this module free of ``habit.core`` imports avoids circular import issues:
``habit.utils.io_utils`` loads ``habit.core`` via the ``habit.core.common.configs`` package.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple


def get_dicom_subject_root_paths(data_dir: str) -> Dict[str, str]:
    """
    Map each immediate subject folder under ``data_dir`` to its DICOM root path.

    This is the usual layout: ``data_dir/<subject_id>/`` contains that person's
    DICOM files (possibly nested). The returned path is always the subject
    folder itself; dcm2niix is responsible for searching inside it.

    Args:
        data_dir: Dataset root that directly contains one subdirectory per subject.

    Returns:
        Mapping ``subject_id -> absolute path`` to each subject directory.

    Raises:
        ValueError: If no subject subdirectories exist.
        NotADirectoryError: If ``data_dir`` is not a directory.
    """
    images_paths, _ = get_image_and_mask_paths_dicom_subject_roots(
        data_dir=data_dir,
        modality_keys=["dicom"],
        auto_select_first_file=False,
    )
    return {subject_id: paths["dicom"] for subject_id, paths in images_paths.items()}


def get_image_and_mask_paths_dicom_subject_roots(
    data_dir: str,
    modality_keys: List[str],
    auto_select_first_file: bool = False,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Build (images_paths, mask_paths) when ``data_dir`` contains one folder per subject.

    Each immediate subdirectory of ``data_dir`` becomes one subject id. The full path
    to that directory is passed through for every entry in ``modality_keys``. This
    matches standalone ``habit sort-dicom`` / dcm2niix use-cases where the interior layout does not
    follow ``<dataset>/images/<subj>/<modality>/`` and is left to dcm2niix to scan.

    Args:
        data_dir: Dataset root that directly contains subject folders (e.g. subj01).
        modality_keys: Keys placed in ``subject_data`` for each subject; must align
            with the first preprocessing step's ``images`` list. Prefer a single key
            (e.g. ``["dicom"]``): duplicate keys point at the same directory and would
            run dcm2niix multiple times on identical input.
        auto_select_first_file: If True, when the subject path is a directory, replace
            it with the first non-hidden file path inside (same semantics as the manifest
            loader). Keep False for DICOM directory inputs.

    Returns:
        ``images_paths`` mapping subject -> {modality_key -> path}, and ``mask_paths``
        as an empty dict (masks are not inferred from this layout).

    Raises:
        ValueError: If ``modality_keys`` is empty or no subject directories exist.
        NotADirectoryError: If ``data_dir`` is not a directory.
    """
    if not modality_keys:
        raise ValueError("modality_keys must not be empty for subject-root (flat) layout")
    root_abs = os.path.abspath(data_dir)
    if not os.path.isdir(root_abs):
        raise NotADirectoryError(f"data_dir is not a directory: {root_abs}")

    subject_names = sorted(
        name
        for name in os.listdir(root_abs)
        if not name.startswith(".") and os.path.isdir(os.path.join(root_abs, name))
    )
    if not subject_names:
        raise ValueError(
            f"No subject subdirectories found under data_dir (subject-root layout): {root_abs}"
        )

    images_paths: Dict[str, Dict[str, str]] = {}
    for subj in subject_names:
        subj_root = os.path.join(root_abs, subj)
        images_paths[subj] = {key: subj_root for key in modality_keys}

    if auto_select_first_file:
        for _subj, img_dict in images_paths.items():
            for _img_type, img_path in list(img_dict.items()):
                if os.path.isdir(img_path):
                    files = [f for f in os.listdir(img_path) if not f.startswith(".")]
                    if files:
                        img_dict[_img_type] = os.path.join(img_path, files[0])

    mask_paths: Dict[str, Dict[str, str]] = {}
    return images_paths, mask_paths
