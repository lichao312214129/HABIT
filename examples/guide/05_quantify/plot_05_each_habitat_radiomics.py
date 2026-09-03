"""
Per-habitat radiomics
=====================

Extract first-order and GLCM features **within each habitat subregion**
using :class:`~habit.habitat_features.EachHabitatRadiomicsFeatures`.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# One-step habitats, then per-habitat PyRadiomics on the intensity image.
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

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

# %%
# Grouped bar chart: Mean, Energy, and GLCM Id across H1 / H2 / H3.
Path("out").mkdir(exist_ok=True)
habitat_ids: List[int] = list(habitat_map.habitat_ids)
# Column pattern: habitat_{id}_original_{class}_{name}_of_{modality}
feature_specs = [
    ("Mean", "firstorder_Mean"),
    ("Energy", "firstorder_Energy"),
    ("GLCM Id", "glcm_Id"),
]
values_by_feature: Dict[str, List[float]] = {}
for label, suffix in feature_specs:
    series: List[float] = []
    for hid in habitat_ids:
        col = next(
            (
                c
                for c in table.feature_columns
                if c.startswith(f"habitat_{hid}_") and suffix in c
            ),
            None,
        )
        series.append(float(row[col]) if col is not None else float("nan"))
    values_by_feature[label] = series

fig, ax = plt.subplots(figsize=(7, 3.5))
x = np.arange(len(habitat_ids))
width = 0.25
for offset, (label, _) in enumerate(feature_specs):
    ax.bar(x + (offset - 1) * width, values_by_feature[label], width, label=label)
ax.set_xticks(x)
ax.set_xticklabels([f"H{hid}" for hid in habitat_ids])
ax.set_ylabel("Feature value")
ax.set_title("Per-habitat radiomics (Mean, Energy, GLCM Id)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("out/each_habitat_radiomics_bar.png", dpi=150, bbox_inches="tight")
plt.show()
