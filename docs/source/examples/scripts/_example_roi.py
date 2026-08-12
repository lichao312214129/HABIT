#!/usr/bin/env python
"""
Tiny I/O helpers for Examples gallery scripts (not shown in Sphinx pages).

Crops clinical demo volumes to a padded ROI / habitat bbox so sklearn-short
demos stay interactive. Synthetic cohorts are returned unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from habit import cohort_from_directory, make_synthetic_cohort
from habit.contracts import ArrayImageRef, Geometry, Subject
from habit.contracts.subject import Cohort

REPO_ROOT = Path(__file__).resolve().parents[4]
DEMO_PREPROCESSED = REPO_ROOT / "demo_data" / "preprocessed"
EXAMPLES_IMG_DIR = (
    Path(__file__).resolve().parents[2] / "_static" / "images" / "examples"
)


def crop_pair(
    volume: np.ndarray,
    mask_or_labels: np.ndarray,
    *,
    pad: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop ``volume`` and ``mask_or_labels`` to a padded foreground bbox.

    Args:
        volume: Anatomy or matching companion array ``(z, y, x)``.
        mask_or_labels: ROI mask or habitat labels (foreground ``> 0``).
        pad: Voxel padding on each side (clipped to bounds).

    Returns:
        Cropped ``(volume, mask_or_labels)`` sharing one shape.
    """
    foreground = mask_or_labels > 0
    if not np.any(foreground):
        raise RuntimeError("No foreground voxels to crop.")
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        start = max(0, int(lo) - pad)
        stop = min(int(mask_or_labels.shape[axis]), int(hi) + pad + 1)
        slices.append(slice(start, stop))
    indexer = tuple(slices)
    return volume[indexer].copy(), mask_or_labels[indexer].copy()


def one_subject_cohort(
    *,
    demo_modalities: Sequence[str] = ("LAP",),
    demo_roi: str = "LAP",
    synthetic_modalities: Sequence[str] = ("T1",),
    synthetic_shape: Tuple[int, int, int] = (40, 40, 40),
    rng: int = 0,
) -> Tuple[Cohort, Tuple[str, ...], bool]:
    """
    Load one subject from ``demo_data/preprocessed`` or a synthetic fallback.

    Args:
        demo_modalities: Modality keys when demo_data is present.
        demo_roi: Mask key for the demo layout.
        synthetic_modalities: Modality keys for the synthetic fallback.
        synthetic_shape: Volume shape for the synthetic subject.
        rng: Synthetic RNG seed.

    Returns:
        ``(cohort, modalities, from_demo)`` where ``cohort`` has length 1.
    """
    if DEMO_PREPROCESSED.is_dir():
        modalities = tuple(demo_modalities)
        cohort = cohort_from_directory(
            DEMO_PREPROCESSED,
            modalities=modalities,
            roi=demo_roi,
        )[:1]
        return cohort, modalities, True
    modalities = tuple(synthetic_modalities)
    cohort = make_synthetic_cohort(
        n_subjects=1,
        modalities=modalities,
        shape=synthetic_shape,
        rng=rng,
    )
    return cohort, modalities, False


def cropped_subject_from(
    subject: Subject,
    modality: str,
    *,
    pad: int = 5,
) -> Tuple[Subject, np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Rebuild a single-modality ``Subject`` cropped to the ROI bbox.

    Args:
        subject: Source subject (demo or synthetic).
        modality: Image / mask key to keep.
        pad: Crop padding in voxels.

    Returns:
        ``(cropped_subject, image, mask, spacing_xyz)``.
    """
    volume = subject.image(modality)
    image = np.asarray(volume.data, dtype=np.float32)
    mask = np.asarray(subject.mask(modality).data, dtype=np.uint8)
    spacing_xyz = tuple(float(v) for v in volume.spacing)
    image_c, mask_c = crop_pair(image, mask, pad=pad)
    geometry = Geometry.from_array(image_c.shape, spacing=spacing_xyz)
    cropped = Subject(
        subject_id=subject.subject_id,
        images={modality: ArrayImageRef(array=image_c, geometry=geometry)},
        masks={
            modality: ArrayImageRef(
                array=(mask_c > 0).astype(np.int32),
                geometry=geometry,
            )
        },
    )
    return cropped, image_c, mask_c, spacing_xyz


def examples_image_dir() -> Path:
    """Return ``docs/source/_static/images/examples`` (created if needed)."""
    EXAMPLES_IMG_DIR.mkdir(parents=True, exist_ok=True)
    return EXAMPLES_IMG_DIR
