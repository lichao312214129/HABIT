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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

# %%
# One panel per feature so scales stay honest (Mean / Energy / GLCM Id
# must not share a single y-axis — Energy dominates and hides the rest).
Path("out").mkdir(exist_ok=True)
habitat_ids: List[int] = list(habitat_map.habitat_ids)
feature_specs: List[Tuple[str, str]] = [
    ("Mean", "firstorder_Mean"),
    ("Energy", "firstorder_Energy"),
    ("GLCM Id", "glcm_Id"),
]


def _column_for(habitat_id: int, suffix: str) -> Optional[str]:
    """Return the first feature column matching habitat id + suffix."""
    prefix = f"habitat_{habitat_id}_"
    for col in table.feature_columns:
        if col.startswith(prefix) and suffix in col:
            return col
    return None


panel_rows: List[Dict[str, Any]] = []
for hid in habitat_ids:
    for label, suffix in feature_specs:
        col = _column_for(hid, suffix)
        panel_rows.append(
            {
                "habitat": f"H{hid}",
                "feature": label,
                "value": float(row[col]) if col is not None else float("nan"),
            }
        )
panel = pd.DataFrame(panel_rows)
print(panel.to_string(index=False))
panel

fig, axes = plt.subplots(1, len(feature_specs), figsize=(9.5, 3.2), sharey=False)
if len(feature_specs) == 1:
    axes = [axes]
x = np.arange(len(habitat_ids))
xticklabels = [f"H{hid}" for hid in habitat_ids]
for ax, (label, suffix) in zip(axes, feature_specs):
    values: List[float] = []
    for hid in habitat_ids:
        col = _column_for(hid, suffix)
        values.append(float(row[col]) if col is not None else float("nan"))
    ax.bar(x, values, color="#0072B2", width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(label)
    ax.set_title(label)
fig.suptitle("Per-habitat radiomics (one scale per feature)", y=1.02)
fig.tight_layout()
fig.savefig("out/each_habitat_radiomics_bar.png", dpi=150, bbox_inches="tight")
plt.show()
