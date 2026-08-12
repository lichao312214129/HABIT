#!/usr/bin/env python
"""
Voxel texture maps: local entropy + a few GLCM features, then plot (sklearn-short).

Accompanies ``docs/source/examples/voxel_texture.rst``.
Run from the repository root::

    python docs/source/examples/scripts/voxel_texture_demo.py
"""

from __future__ import annotations

# BEGIN example
import matplotlib.pyplot as plt
import numpy as np

import habit.domain  # registers built-in voxel extractors
from habit import local_entropy_map
from habit.domain import VoxelFeatureExtractorRegistry
from habit.viz import dense_voxel_feature_map, plot_voxel_texture_slice, use_style

from _example_roi import cropped_subject_from, examples_image_dir, one_subject_cohort

cohort, modalities, _ = one_subject_cohort()
modality = modalities[0]
subject, image, mask, spacing = cropped_subject_from(cohort[0], modality)

# Fast local entropy + a small GLCM set (PyRadiomics / voxel_radiomics)
entropy = np.where(mask > 0, local_entropy_map(image, kernel_size=5, bins=32), np.nan)
glcm = VoxelFeatureExtractorRegistry.create(
    "voxel_radiomics",
    modality=modality,
    kernel_radius=1,
    params={
        "imageType": {"Original": {}},
        "featureClass": {"glcm": ["Contrast", "Correlation", "JointEntropy"]},
        "setting": {"binWidth": 25},
    },
)(subject)
contrast = dense_voxel_feature_map(glcm, next(n for n in glcm.feature_names if "Contrast" in n))
print("GLCM columns:", list(glcm.feature_names))

# side_by_side: anatomy + ROI contour | texture (no alpha blend on anatomy)
out = examples_image_dir()
with use_style("radiology"):
    fig = plot_voxel_texture_slice(
        entropy, anatomy=image, roi_mask=mask, axis=0, mode="side_by_side",
        spacing=spacing, feature_label="Local entropy (bits)",
        title="Voxel texture: local entropy",
    )
fig.savefig(out / "voxel_texture_side_by_side.png", dpi=150, bbox_inches="tight")
plt.close(fig)

with use_style("radiology"):
    fig = plot_voxel_texture_slice(
        contrast, anatomy=image, roi_mask=mask, axis=0, mode="side_by_side",
        spacing=spacing, feature_label="GLCM Contrast",
        title="Voxel texture: GLCM Contrast",
    )
fig.savefig(out / "voxel_texture_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
# END example

if __name__ == "__main__":
    with use_style("radiology"):
        fig = plot_voxel_texture_slice(
            entropy, anatomy=image, roi_mask=mask, mode="side_by_side",
            spacing=spacing, feature_label="Local entropy (bits)",
            title="Local entropy (orthogonal)",
        )
    fig.savefig(out / "voxel_texture_orthogonal.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Wrote gallery PNGs under", out)
