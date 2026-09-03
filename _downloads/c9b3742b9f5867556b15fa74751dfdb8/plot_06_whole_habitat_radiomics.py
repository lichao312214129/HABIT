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
from typing import Any, Dict

import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_features import WholeHabitatRadiomicsFeatures
from habit.recipes import one_step_habitat

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
