"""
Intratumoral heterogeneity (ITH)
================================

Atomic ITH metrics from a habitat label map:
:func:`~habit.kernels.ith_score` and
:func:`~habit.kernels.habitat_ith_dispersion`.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# One-step habitats, then ITH scalar plus per-habitat dispersion.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import habitat_ith_dispersion, ith_score
from habit.recipes import one_step_habitat
from habit.viz import plot_ith_summary

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)
labels = result.habitat_maps[0].label_array

ith = float(ith_score(labels))
dispersion = habitat_ith_dispersion(labels)
ith_table = pd.DataFrame(
    [{"habitat": "ITH", "score": ith}]
    + [{"habitat": f"H{hid}", "score": float(val)} for hid, val in sorted(dispersion.items())]
)
print("ITH scores:")
print(ith_table.to_string(index=False))
ith_table

Path("out").mkdir(exist_ok=True)
fig_ith = plot_ith_summary(ith, dispersion=dispersion)
fig_ith.savefig("out/ith_summary.png", dpi=150, bbox_inches="tight")
plt.show()
