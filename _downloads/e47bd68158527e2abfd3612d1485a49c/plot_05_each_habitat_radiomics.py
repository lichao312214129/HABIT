"""
Per-habitat radiomics
=====================

Extract first-order and GLCM features **within each habitat subregion**
using :class:`~habit.habitat_features.EachHabitatRadiomicsFeatures`.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# One-step habitats, then per-habitat PyRadiomics on the intensity image.
from typing import Any, Dict

import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_features import EachHabitatRadiomicsFeatures
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

# Narrow PyRadiomics params keep the gallery table readable.
params: Dict[str, Any] = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Energy"],
        "glcm": ["Autocorrelation", "Id"],
    },
    "setting": {"binWidth": 25, "voxelArrayShift": 0},
}
table = EachHabitatRadiomicsFeatures(params=params)(subject, habitat_map)
row = table.frame.iloc[0]
display_cols = [
    col
    for col in table.feature_columns
    if "firstorder" in col or "glcm" in col
][:8]
print("Per-habitat radiomics (sample columns):")
print(row[display_cols].to_string())
row[display_cols]
