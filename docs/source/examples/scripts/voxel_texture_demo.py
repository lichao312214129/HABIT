#!/usr/bin/env python
"""
Voxel texture map: per-voxel GLCM Contrast, then plot.

Accompanies ``docs/source/how_to/voxel_texture.rst``.
Run from the repository root::

    python docs/source/examples/scripts/voxel_texture_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.viz import plot_voxel_texture_slice
from habit.voxel_features import extract_voxel_texture

# Change DATA / MODALITY / ROI to your preprocessed layout
DATA = fetch_demo()  # or "demo_data/preprocessed"
MODALITY = "LAP"
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=(MODALITY,), roi=ROI)[0]
image_vol = subject.image(MODALITY)
mask_vol = subject.mask(ROI)
field = extract_voxel_texture(
    image_vol,
    mask_vol,
    kernel_radius=1,
    bin_width=25.0,
    feature_classes={"glcm": ["Contrast"]},
)
Path("out").mkdir(exist_ok=True)
fig = plot_voxel_texture_slice(
    field, feature=0, anatomy=image_vol, roi_mask=mask_vol,
)
fig.savefig("out/voxel_texture_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote out/voxel_texture_overlay.png")
fig_side = plot_voxel_texture_slice(
    field,
    feature=0,
    anatomy=image_vol,
    roi_mask=mask_vol,
    mode="side_by_side",
)
fig_side.savefig(
    "out/voxel_texture_side_by_side.png", dpi=150, bbox_inches="tight"
)
plt.close(fig_side)
print("Wrote out/voxel_texture_side_by_side.png")
# END example


def _copy_gallery(src: Path, dest: Path) -> None:
    """Copy a PNG into the Sphinx gallery; skip if the dest file is locked."""
    import shutil

    try:
        shutil.copyfile(src, dest)
    except OSError as exc:
        # Windows Errno 22 when another process (preview / AV) holds the PNG.
        print(f"Gallery copy skipped ({dest.name}): {exc}")


if __name__ == "__main__":
    # Docs gallery assets (maintainers); copy from out/ only.
    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    _copy_gallery(
        Path("out/voxel_texture_overlay.png"),
        gallery / "voxel_texture_overlay.png",
    )
    _copy_gallery(
        Path("out/voxel_texture_overlay.png"),
        gallery / "voxel_texture_glcm_overlay.png",
    )
    _copy_gallery(
        Path("out/voxel_texture_side_by_side.png"),
        gallery / "voxel_texture_side_by_side.png",
    )
    print("Wrote out/ and gallery PNGs")
