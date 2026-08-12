#!/usr/bin/env python
"""
Voxel-level texture maps on real demo_data anatomy + ROI masks.

Shows the preferred public paths:

* Kernel: ``habit.local_entropy_map`` (fast built-in neighbourhood entropy)
* Domain: ``VoxelFeatureExtractorRegistry.create("local_entropy", ...)``
* Viz: ``habit.viz.plot_voxel_texture_slice`` / ``dense_voxel_feature_map``

Writes English-labelled PNGs under ``docs/source/_static/images/examples/``
for the Sphinx Examples gallery. Requires local ``demo_data/`` (not shipped
in git) plus matplotlib (``pip install 'habit[viz]'``).

Heavy ``voxel_radiomics`` maps are also accepted by the plotter once densified;
this demo uses local entropy so the gallery regenerates in seconds.

Accompanies ``docs/source/examples/voxel_texture.rst``.

Run from the repository root::

    python docs/source/examples/scripts/voxel_texture_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import habit.domain  # registers built-in voxel feature extractors
from habit import local_entropy_map
from habit.contracts import ArrayImageRef, Geometry, Subject
from habit.domain import VoxelFeatureExtractorRegistry
from habit.viz import (
    dense_voxel_feature_map,
    plot_voxel_texture_slice,
    use_style,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = (
    Path(__file__).resolve().parents[2] / "_static" / "images" / "examples"
)

IMAGE_CANDIDATES: Tuple[Path, ...] = (
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "images"
    / "subj001"
    / "LAP"
    / "WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd",
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "images"
    / "subj001"
    / "PVP"
    / "WATER__WATER__WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd",
)
MASK_CANDIDATES: Tuple[Path, ...] = (
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "masks"
    / "subj001"
    / "LAP"
    / "WATER__BH_Ax_LAVA_Flex_10min_Series0017_mask.nrrd",
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "masks"
    / "subj001"
    / "PVP"
    / "WATER__BH_Ax_LAVA_Flex_10min_Series0017_mask.nrrd",
)

BBOX_PAD: int = 5
KERNEL_SIZE: int = 5
BINS: int = 32


def _first_existing(paths: Sequence[Path]) -> Path | None:
    """Return the first path that exists, or ``None``."""
    for path in paths:
        if path.is_file():
            return path
    return None


def _demo_data_missing_message() -> str:
    """Build a clear error when local demo_data volumes are absent."""
    tried_img = "\n".join(f"  - {path}" for path in IMAGE_CANDIDATES)
    tried_mask = "\n".join(f"  - {path}" for path in MASK_CANDIDATES)
    return (
        "demo_data anatomy / mask not found. This example requires local "
        "demo_data (not committed to git).\n"
        "Looked for images:\n"
        f"{tried_img}\n"
        "Looked for masks:\n"
        f"{tried_mask}\n"
        "Committed gallery PNGs under docs/source/_static/images/examples/ "
        "were produced from demo_data on the maintainer machine."
    )


def load_demo_volumes() -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], Path, Path]:
    """
    Load subj001 LAP anatomy and matching ROI mask from demo_data.

    Returns:
        Tuple of ``(image, mask, spacing_xyz, image_path, mask_path)``.

    Raises:
        FileNotFoundError: When neither image nor mask candidates exist.
        RuntimeError: When shapes differ.
    """
    image_path = _first_existing(IMAGE_CANDIDATES)
    mask_path = _first_existing(MASK_CANDIDATES)
    if image_path is None or mask_path is None:
        raise FileNotFoundError(_demo_data_missing_message())

    image_sitk = sitk.ReadImage(str(image_path))
    mask_sitk = sitk.ReadImage(str(mask_path))
    image = np.asarray(sitk.GetArrayFromImage(image_sitk), dtype=np.float32)
    mask = np.asarray(sitk.GetArrayFromImage(mask_sitk), dtype=np.uint8)
    if image.shape != mask.shape:
        raise RuntimeError(
            "Anatomy and mask shapes differ; cannot overlay. "
            f"image={image.shape} mask={mask.shape} "
            f"({image_path} vs {mask_path})"
        )
    spacing_xyz = tuple(float(v) for v in image_sitk.GetSpacing())
    if len(spacing_xyz) != 3:
        raise RuntimeError(f"Expected 3D spacing, got {spacing_xyz!r}")
    return image, mask, spacing_xyz, image_path, mask_path


def crop_to_roi_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    pad: int = BBOX_PAD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop both volumes to the padded bounding box of the ROI.

    Args:
        image: Full anatomy volume ``(z, y, x)``.
        mask: Matching ROI mask.
        pad: Voxel padding on each side (clipped to array bounds).

    Returns:
        Cropped ``(image, mask)`` arrays sharing the same shape.
    """
    foreground = mask > 0
    if not np.any(foreground):
        raise RuntimeError("ROI mask has no positive voxels.")
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        start = max(0, int(lo) - pad)
        stop = min(int(mask.shape[axis]), int(hi) + pad + 1)
        slices.append(slice(start, stop))
    indexer = tuple(slices)
    return image[indexer].copy(), mask[indexer].copy()


def kernel_path(image: np.ndarray) -> np.ndarray:
    """
    Compute a dense local-entropy map (arrays in, same-shaped map out).

    Args:
        image: Intensity volume.

    Returns:
        Float64 entropy map matching ``image.shape``.
    """
    return local_entropy_map(image, kernel_size=KERNEL_SIZE, bins=BINS)


def domain_path(
    image: np.ndarray,
    mask: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, str]:
    """
    Compute local entropy via the domain registry and densify one column.

    Args:
        image: Cropped anatomy volume.
        mask: Cropped ROI mask.
        spacing_xyz: SimpleITK spacing ``(x, y, z)``.

    Returns:
        Tuple of ``(dense_feature_map, feature_name)``.
    """
    geometry = Geometry.from_array(image.shape, spacing=spacing_xyz)
    subject = Subject(
        subject_id="subj001",
        images={
            "LAP": ArrayImageRef(array=image, geometry=geometry),
        },
        masks={
            "tumor": ArrayImageRef(
                array=(mask > 0).astype(np.int32),
                geometry=geometry,
            )
        },
    )
    extractor = VoxelFeatureExtractorRegistry.create(
        "local_entropy",
        modality="LAP",
        roi="tumor",
        kernel_size=KERNEL_SIZE,
        bins=BINS,
    )
    field = extractor(subject)
    feature_name = str(field.feature_names[0])
    dense = dense_voxel_feature_map(field, feature_name)
    return dense, feature_name


def save_publication_figures(
    image: np.ndarray,
    mask: np.ndarray,
    entropy: np.ndarray,
    *,
    spacing_xyz: Tuple[float, float, float],
    feature_label: str,
) -> Dict[str, Path]:
    """
    Render side-by-side and overlay panels for the docs gallery.

    Args:
        image: Cropped anatomy.
        mask: Cropped ROI.
        entropy: Dense texture map (same shape).
        spacing_xyz: SimpleITK spacing ``(x, y, z)``.
        feature_label: English colourbar / panel label.

    Returns:
        Mapping from asset stem to written PNG path.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    with use_style("radiology"):
        fig_side = plot_voxel_texture_slice(
            entropy,
            anatomy=image,
            roi_mask=mask,
            axis=0,
            mode="side_by_side",
            spacing=spacing_xyz,
            feature_label=feature_label,
            title="Demo subj001 LAP local entropy",
        )
    side_path = OUT_DIR / "voxel_texture_side_by_side.png"
    fig_side.savefig(side_path, dpi=200, bbox_inches="tight")
    plt.close(fig_side)
    written["side_by_side"] = side_path

    with use_style("nature"):
        fig_overlay = plot_voxel_texture_slice(
            entropy,
            anatomy=image,
            roi_mask=mask,
            axis=0,
            mode="overlay",
            alpha=0.55,
            cmap="magma",
            spacing=spacing_xyz,
            feature_label=feature_label,
            title="Local entropy on anatomy",
        )
    overlay_path = OUT_DIR / "voxel_texture_overlay.png"
    fig_overlay.savefig(overlay_path, dpi=200, bbox_inches="tight")
    plt.close(fig_overlay)
    written["overlay"] = overlay_path

    with use_style("radiology"):
        fig_trip = plot_voxel_texture_slice(
            entropy,
            anatomy=image,
            roi_mask=mask,
            mode="side_by_side",
            spacing=spacing_xyz,
            feature_label=feature_label,
            title="Local entropy (orthogonal)",
        )
    trip_path = OUT_DIR / "voxel_texture_orthogonal.png"
    fig_trip.savefig(trip_path, dpi=160, bbox_inches="tight")
    plt.close(fig_trip)
    written["orthogonal"] = trip_path

    return written


def main() -> None:
    """Load demo_data, compute local entropy two ways, write gallery PNGs."""
    image, mask, spacing_xyz, image_path, mask_path = load_demo_volumes()
    image_c, mask_c = crop_to_roi_bbox(image, mask)
    print(f"Loaded anatomy: {image_path}")
    print(f"Loaded mask:    {mask_path}")
    print(f"Cropped shape:  {image_c.shape}  spacing_xyz={spacing_xyz}")

    entropy_kernel = kernel_path(image_c)
    # Restrict display to ROI so background neighbourhood damping stays hidden.
    entropy_display = np.where(mask_c > 0, entropy_kernel, np.nan)

    field_dense, feature_name = domain_path(image_c, mask_c, spacing_xyz)
    # Kernel vs domain should agree on ROI voxels (same local_entropy_map).
    inside = mask_c > 0
    max_abs = float(np.nanmax(np.abs(entropy_kernel[inside] - field_dense[inside])))
    print(f"Domain feature column: {feature_name}")
    print(f"Kernel vs domain max |diff| on ROI: {max_abs:.3e}")

    written = save_publication_figures(
        image_c,
        mask_c,
        entropy_display,
        spacing_xyz=spacing_xyz,
        feature_label="Local entropy (bits)",
    )
    for stem, path in written.items():
        print(f"Wrote {stem}: {path} ({path.stat().st_size} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
