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
"""Deterministic SimpleITK NRRD cohort tree for cloud behavior tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import SimpleITK as sitk

#: Fixed master seed so regeneration is byte-stable across machines.
SYNTHETIC_SEED: int = 42

#: Subject identifiers written under ``images/`` and ``masks/``.
SUBJECT_IDS: Tuple[str, ...] = ("subj001", "subj002")

#: Modalities matching the minimal two-step habitat demo config.
MODALITIES: Tuple[str, ...] = ("delay2", "delay3", "delay5")

#: ROI key shared by images and masks (mask lives under ``masks/<subject>/delay2/``).
ROI_NAME: str = "delay2"

#: Volume shape in SimpleITK array order ``(z, y, x)`` for a 48x48x12 grid.
VOLUME_SHAPE: Tuple[int, int, int] = (12, 48, 48)

#: Voxel spacing ``(z, y, x)`` corresponding to physical spacing 1x1x2.
VOLUME_SPACING: Tuple[float, float, float] = (2.0, 1.0, 1.0)


def _ellipsoid_mask(
    shape: Tuple[int, int, int],
    *,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
) -> np.ndarray:
    """
    Build a binary ellipsoid mask inside a 3-D grid.

    Args:
        shape: Grid shape ``(z, y, x)``.
        center: Ellipsoid centre in voxel coordinates ``(z, y, x)``.
        radii: Semi-axes along ``(z, y, x)``.

    Returns:
        ``float32`` mask with ones inside the ellipsoid and zeros elsewhere.
    """
    grid_z, grid_y, grid_x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    cz, cy, cx = center
    rz, ry, rx = radii
    normalised = (
        ((grid_z - cz) / max(rz, 1.0)) ** 2
        + ((grid_y - cy) / max(ry, 1.0)) ** 2
        + ((grid_x - cx) / max(rx, 1.0)) ** 2
    )
    return (normalised <= 1.0).astype(np.float32)


def _compact_subregion_mask(
    shape: Tuple[int, int, int],
    *,
    center: Tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """
    Build a compact spherical subregion mask.

    Args:
        shape: Grid shape ``(z, y, x)``.
        center: Sphere centre ``(z, y, x)``.
        radius: Sphere radius in voxels.

    Returns:
        ``float32`` mask with ones inside the sphere.
    """
    grid_z, grid_y, grid_x = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    cz, cy, cx = center
    distance = np.sqrt(
        (grid_z - cz) ** 2 + (grid_y - cy) ** 2 + (grid_x - cx) ** 2
    )
    return (distance <= radius).astype(np.float32)


def _write_nrrd(path: Path, array: np.ndarray, *, spacing: Tuple[float, float, float]) -> None:
    """
    Persist one volume as NRRD with explicit spacing.

    Args:
        path: Destination ``.nrrd`` path.
        array: Volume array in SimpleITK order ``(z, y, x)``.
        spacing: Physical spacing ``(z, y, x)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


def build_synthetic_tree(root: Path, *, seed: int = SYNTHETIC_SEED) -> Path:
    """
    Write the conventional HABIT directory tree with two subjects.

    Layout::

        <root>/images/<subject>/<modality>/image.nrrd
        <root>/masks/<subject>/delay2/mask.nrrd

    Each subject receives an ellipsoid tumour ROI containing two compact
    subregions with distinct baseline intensities per modality. Gaussian
    noise is added with a fixed seed so repeated calls are byte-identical.

    Args:
        root: Cohort root directory to create.
        seed: Master random seed controlling per-voxel noise.

    Returns:
        ``root`` for chaining.
    """
    rng = np.random.default_rng(seed)
    shape = VOLUME_SHAPE
    centre = tuple(float(value) / 2.0 for value in shape)
    tumour = _ellipsoid_mask(shape, center=centre, radii=(4.5, 14.0, 14.0))

    # Two compact subregions planted inside the ellipsoid with different means.
    subregion_a = _compact_subregion_mask(shape, center=(centre[0] - 1.5, centre[1] - 4.0, centre[2] - 4.0), radius=3.5)
    subregion_b = _compact_subregion_mask(shape, center=(centre[0] + 1.5, centre[1] + 4.0, centre[2] + 4.0), radius=3.5)
    subregion_a *= tumour
    subregion_b *= tumour

    region_profiles = {
        "a": (0.85, 1.55, 2.25),
        "b": (1.45, 0.95, 2.05),
    }

    for subject_index, subject_id in enumerate(SUBJECT_IDS):
        subject_offset = float(rng.normal(scale=0.04))
        for modality_index, modality in enumerate(MODALITIES):
            volume = np.zeros(shape, dtype=np.float32)
            for tag, region_mask, profile in (
                ("a", subregion_a, region_profiles["a"]),
                ("b", subregion_b, region_profiles["b"]),
            ):
                base = profile[modality_index] + subject_offset + subject_index * 0.08
                volume[region_mask > 0] = base
            volume += rng.normal(scale=0.015, size=shape).astype(np.float32)
            volume[tumour == 0] = 0.0
            image_path = root / "images" / subject_id / modality / "image.nrrd"
            _write_nrrd(image_path, volume, spacing=VOLUME_SPACING)

        mask_path = root / "masks" / subject_id / ROI_NAME / "mask.nrrd"
        _write_nrrd(mask_path, tumour.astype(np.uint8), spacing=VOLUME_SPACING)

    return root


def minimal_v0_habitat_yaml(
    data_dir: Path,
    out_dir: Path,
    *,
    extra_lines: str = "",
) -> str:
    """
    Render a small v0 two-step habitat config adapted to the synthetic tree.

    Parameters mirror ``config/habitat/config_habitat_two_step_minimal.yaml``
    but use smaller supervoxel/habitat counts for fast CI runs.

    Args:
        data_dir: Cohort root passed to ``data_dir``.
        out_dir: Output directory passed to ``out_dir``.
        extra_lines: Optional YAML fragment appended before ``feature_construction``.

    Returns:
        Raw YAML text.
    """
    return f"""run_mode: train
data_dir: "{data_dir.as_posix()}"
out_dir: "{out_dir.as_posix()}"
processes: 1
plot_curves: false
save_images: true
save_results_csv: true
habitats_results_format: parquet
random_state: {SYNTHETIC_SEED}
{extra_lines}feature_construction:
  voxel_level:
    method: concat(raw({MODALITIES[0]}), raw({MODALITIES[1]}), raw({MODALITIES[2]}))
    params: {{}}
  supervoxel_level:
    supervoxel_file_keyword: '*_supervoxel.nrrd'
    method: mean_voxel_features()
    params: {{}}
  preprocessing_for_subject_level:
    methods:
      - method: winsorize
        winsor_limits: [0.05, 0.05]
        global_normalize: false
      - method: minmax
        global_normalize: false
  preprocessing_for_group_level:
    methods:
      - method: binning
        n_bins: 10
        bin_strategy: uniform
        global_normalize: false
habitat_segmentation:
  clustering_mode: two_step
  supervoxel:
    algorithm: kmeans
    n_clusters: 20
    max_iter: 100
    n_init: 5
  habitat:
    algorithm: kmeans
    max_clusters: 4
    habitat_cluster_selection_method:
      - elbow
    max_iter: 100
    n_init: 5
"""


def modality_list() -> Sequence[str]:
    """Return the modality names used by the synthetic cohort."""
    return MODALITIES
