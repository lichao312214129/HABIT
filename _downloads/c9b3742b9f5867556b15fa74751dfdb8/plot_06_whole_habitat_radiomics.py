"""
Whole-habitat radiomics
=======================

Quantify the **shape and spatial distribution of the partition map itself**
using :class:`~habit.habitat_features.WholeHabitatRadiomicsFeatures`.
The habitat label image plays both intensity and mask roles.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# One-step habitats, then whole-map PyRadiomics on the label field.
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import MaskVolume, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_features import WholeHabitatRadiomicsFeatures
from habit.kernels import local_entropy_map
from habit.recipes import one_step_habitat
from habit.viz import plot_habitat_overlay, plot_voxel_texture_slice

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)
subject = cohort[0]
habitat_map = result.habitat_maps[0]

params: Dict[str, Any] = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Entropy"],
        "shape": ["Sphericity", "SurfaceArea"],
    },
    "setting": {"binWidth": 1, "voxelArrayShift": 0},
}
table = WholeHabitatRadiomicsFeatures(params=params)(subject, habitat_map)
row = table.frame.iloc[0]
print("Whole-habitat radiomics:")
print(row.to_string())
row

# %%
# Same objects as the table: habitat IDs are the intensity image.
# First figure: the partition. Second: local entropy of those discrete
# labels (not of the MRI), cropped to the same ROI.
Path("out").mkdir(exist_ok=True)
image_vol = subject.image(ROI)
fig_hab = plot_habitat_overlay(
    image_vol,
    habitat_map,
    axis=0,
    crop_to="labels",
    title="habitats",
)
fig_hab.savefig("out/whole_habitat_radiomics_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

label_intensity: np.ndarray = np.asarray(habitat_map.label_array, dtype=np.float64)
n_label_bins: int = max(int(label_intensity.max()) + 1, 2)
entropy = local_entropy_map(label_intensity, kernel_size=5, bins=n_label_bins)
habitat_roi = MaskVolume.from_geometry(
    (habitat_map.label_array > 0).astype(np.uint8),
    habitat_map.geometry,
    roi_name=ROI,
)
fig = plot_voxel_texture_slice(
    entropy,
    anatomy=image_vol,
    roi_mask=habitat_roi,
    axis=0,
    crop_to="roi",
    title="local entropy of habitat labels",
    feature_label="label entropy",
)
fig.savefig("out/whole_habitat_radiomics_texture.png", dpi=150, bbox_inches="tight")
plt.show()
