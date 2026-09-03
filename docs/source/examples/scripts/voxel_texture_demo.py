#!/usr/bin/env python
"""
Voxel texture maps: local entropy + GLCM Contrast, then plot.

Accompanies ``docs/source/examples/voxel_texture.rst``.
Run from the repository root::

    python docs/source/examples/scripts/voxel_texture_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import matplotlib.pyplot as plt

import habit.voxel_features  # registers built-in voxel extractors
from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import local_entropy_map
from habit.voxel_features import VoxelFeatureExtractorRegistry
from habit.viz import plot_voxel_texture_slice

# Change DATA / MODALITY / ROI to your preprocessed layout
DATA = fetch_demo()
MODALITY = "LAP"
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=(MODALITY,), roi=ROI)[:1]
subject = cohort[0]
# Pass volume objects (not .data) so direction/spacing stay attached.
# Image vs mask direction may disagree; plotters warn and use the mask.
image_vol = subject.image(MODALITY)
mask_vol = subject.mask(ROI)
image = image_vol.data
mask = mask_vol.data

entropy = local_entropy_map(image, kernel_size=5, bins=32)
glcm = VoxelFeatureExtractorRegistry.create(
    "voxel_radiomics",
    modality=MODALITY,
    kernel_radius=1,
    params={
        "imageType": {"Original": {}},
        "featureClass": {"glcm": ["Contrast", "Correlation", "JointEntropy"]},
        "setting": {"binWidth": 25},
    },
)(subject)

kw = dict(anatomy=image_vol, roi_mask=mask_vol)
Path("out").mkdir(exist_ok=True)
fig = plot_voxel_texture_slice(entropy, **kw)
fig.savefig("out/voxel_texture_entropy.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote out/voxel_texture_entropy.png")
fig = plot_voxel_texture_slice(entropy, mode="side_by_side", **kw)
fig.savefig("out/voxel_texture_entropy_side_by_side.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote out/voxel_texture_entropy_side_by_side.png")
fig = plot_voxel_texture_slice(glcm, feature=0, **kw)
fig.savefig("out/voxel_texture_glcm_contrast.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Wrote out/voxel_texture_glcm_contrast.png")
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
    # Docs gallery assets (maintainers); copy from out/ then orthogonal panel.
    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    _copy_gallery(
        Path("out/voxel_texture_entropy.png"),
        gallery / "voxel_texture_overlay.png",
    )
    _copy_gallery(
        Path("out/voxel_texture_entropy_side_by_side.png"),
        gallery / "voxel_texture_side_by_side.png",
    )
    _copy_gallery(
        Path("out/voxel_texture_glcm_contrast.png"),
        gallery / "voxel_texture_glcm_overlay.png",
    )
    fig = plot_voxel_texture_slice(
        entropy,
        anatomy=image_vol,
        roi_mask=mask_vol,
        feature_label="Local entropy",
        title="Local entropy (orthogonal)",
    )
    try:
        fig.savefig(
            gallery / "voxel_texture_orthogonal.png",
            dpi=140,
            bbox_inches="tight",
        )
    except OSError as exc:
        print(f"Gallery save skipped (voxel_texture_orthogonal.png): {exc}")
    plt.close(fig)
    print("Wrote out/ and gallery PNGs")
