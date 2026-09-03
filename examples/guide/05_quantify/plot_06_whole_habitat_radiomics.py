"""
Whole-habitat radiomics
=======================

Quantify the **shape and spatial distribution of the partition map itself**
using :class:`~habit.habitat_features.WholeHabitatRadiomicsFeatures`.
The habitat label image plays both intensity and mask roles.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# One-step habitats, then whole-map PyRadiomics on the label field.
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
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

# %%
# Horizontal bar chart of shape and first-order features on the label map.
Path("out").mkdir(exist_ok=True)
plot_items: List[Tuple[str, str]] = [
    ("Sphericity", "original_shape_Sphericity"),
    ("SurfaceArea", "original_shape_SurfaceArea"),
    ("Mean", "original_firstorder_Mean"),
    ("Entropy", "original_firstorder_Entropy"),
]
labels = [name for name, _ in plot_items]
values = [float(row[col]) for _, col in plot_items]

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(labels, values, color="#4C72B0")
ax.set_xlabel("Feature value")
ax.set_title("Whole-habitat radiomics (shape + first-order)")
fig.tight_layout()
fig.savefig("out/whole_habitat_radiomics_bar.png", dpi=150, bbox_inches="tight")
plt.show()
