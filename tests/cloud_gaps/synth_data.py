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
"""Deterministic synthetic demo dataset writer for cloud gap tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import SimpleITK as sitk

#: Master seed for every synthetic volume in this helper.
MASTER_SEED: int = 42

#: Volume shape ``(z, y, x)`` written to every NRRD file.
VOLUME_SHAPE: Tuple[int, int, int] = (48, 48, 12)

#: DCE modality folder names expected by the shipped demo configs.
MODALITIES: Tuple[str, ...] = ("delay2", "delay3", "delay5")

#: ROI key used by demo configs (mask lives under masks/<subject>/delay2/).
ROI: str = "delay2"

#: Subject identifiers written under images/ and masks/.
SUBJECT_IDS: Tuple[str, ...] = ("subj001", "subj002")

#: Default root matching the demo configs' data_dir / data.source paths.
DEFAULT_DATA_ROOT: Path = (
    Path(__file__).resolve().parents[2]
    / "demo_data"
    / "preprocessed"
    / "processed_images"
)


def _ellipsoid_mask(
    shape: Tuple[int, int, int],
    *,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    """
    Build a binary ellipsoid tumour mask inside a cubic grid.

    Args:
        shape: Volume shape ``(z, y, x)``.
        center: Ellipsoid centre in voxel coordinates ``(z, y, x)``.
        radii: Semi-axes along ``(z, y, x)``.

    Returns:
        ``uint8`` mask with ones inside the ellipsoid and zeros elsewhere.
    """
    grid_z, grid_y, grid_x = np.mgrid[
        0 : shape[0], 0 : shape[1], 0 : shape[2]
    ]
    cz, cy, cx = center
    rz, ry, rx = radii
    normalised = (
        ((grid_z - cz) / max(rz, 1.0)) ** 2
        + ((grid_y - cy) / max(ry, 1.0)) ** 2
        + ((grid_x - cx) / max(rx, 1.0)) ** 2
    )
    return (normalised <= 1.0).astype(np.uint8)


def _two_subregion_labels(mask: np.ndarray) -> np.ndarray:
    """
    Split the ROI into two compact subregions along the z axis.

    Args:
        mask: Binary ROI mask.

    Returns:
        Label array with background ``0`` and subregion ids ``1`` and ``2``.
    """
    labels = np.zeros_like(mask, dtype=np.int32)
    roi_coords = np.argwhere(mask > 0)
    if roi_coords.size == 0:
        return labels
    z_values = roi_coords[:, 0]
    z_mid = float(z_values.min() + z_values.max()) / 2.0
    z_index = np.arange(mask.shape[0])[:, None, None]
    labels[(mask > 0) & (z_index <= z_mid)] = 1
    labels[(mask > 0) & (z_index > z_mid)] = 2
    return labels


def _subject_volumes(
    subject_index: int,
    *,
    shape: Tuple[int, int, int] = VOLUME_SHAPE,
    modalities: Sequence[str] = MODALITIES,
    seed: int = MASTER_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build one subject's modality images and shared ROI mask.

    Two subregions inside the ellipsoid receive distinct mean intensities per
    modality so downstream clustering sees separable habitat signal.

    Args:
        subject_index: Zero-based subject index (varies per-subject noise).
        shape: Volume shape ``(z, y, x)``.
        modalities: Modality keys to populate.
        seed: Master seed combined with ``subject_index`` for determinism.

    Returns:
        Tuple of ``(images_by_modality, mask)`` where images are ``float32``
        arrays keyed by modality name.
    """
    rng = np.random.default_rng(seed + subject_index)
    centre = tuple(float(v) / 2.0 for v in shape)
    radii = tuple(max(2.0, float(v) * 0.30) for v in shape)
    mask = _ellipsoid_mask(shape, center=centre, radii=radii)
    subregion_labels = _two_subregion_labels(mask)

    # Distinct baseline intensities per subregion and modality.
    region_profiles = {
        1: {"delay2": 80.0, "delay3": 120.0, "delay5": 160.0},
        2: {"delay2": 140.0, "delay3": 90.0, "delay5": 110.0},
    }
    subject_offset = float(rng.normal(scale=2.0))
    images: dict[str, np.ndarray] = {}
    for modality in modalities:
        array = np.zeros(shape, dtype=np.float32)
        for region_id in (1, 2):
            region_mask = subregion_labels == region_id
            base = region_profiles[region_id][modality] + subject_offset
            array[region_mask] = base
        array += rng.normal(scale=1.0, size=shape).astype(np.float32)
        array[mask == 0] = 0.0
        images[str(modality)] = array
    return images, mask


def _write_nrrd(path: Path, array: np.ndarray) -> None:
    """
    Write one array as NRRD, creating parent directories as needed.

    Args:
        path: Destination ``.nrrd`` path.
        array: Volume array in ``(z, y, x)`` order for SimpleITK.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))


def write_synthetic_demo_dataset(
    root: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Write the two-subject demo layout expected by shipped example/config files.

    Layout::

        <root>/images/<subject>/<modality>/image.nrrd
        <root>/masks/<subject>/<roi>/mask.nrrd

    Args:
        root: Cohort root directory; defaults to ``DEFAULT_DATA_ROOT``.
        overwrite: When ``False`` and the root already exists, leave it intact.

    Returns:
        Absolute path to the cohort root that was written or already present.
    """
    data_root = (DEFAULT_DATA_ROOT if root is None else Path(root)).resolve()
    # Guard against clobbering a pre-existing dataset: the real demo data
    # lives at this path on developer machines with different filenames
    # (e.g. delay2.nii.gz), so key on "subject dir already has any data
    # files", not on the specific synthetic filenames.
    subject_dir = data_root / "images" / SUBJECT_IDS[0]
    if not overwrite and subject_dir.is_dir() and any(subject_dir.rglob("*.*")):
        return data_root

    for index, subject_id in enumerate(SUBJECT_IDS):
        images, mask = _subject_volumes(index)
        for modality, array in images.items():
            image_path = data_root / "images" / subject_id / modality / "image.nrrd"
            _write_nrrd(image_path, array)
        mask_path = data_root / "masks" / subject_id / ROI / "mask.nrrd"
        _write_nrrd(mask_path, mask)
    return data_root
