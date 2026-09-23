"""
Multiregional spatial interaction (MSI)
=======================================

Atomic MSI from a habitat label map (Wu et al., *Radiology* 2018):
:func:`~habit.kernels.spatial_interaction_matrix` and
:func:`~habit.kernels.msi_features_from_matrix`.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Build a habitat map, then compute the MSI matrix and scalar summaries.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import msi_features_from_matrix, spatial_interaction_matrix
from habit.recipes import one_step_habitat
from habit.viz import plot_msi_matrix

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)
habitat_map = result.habitat_maps[0]
labels = habitat_map.label_array
ids = tuple(sorted({int(v) for v in labels.ravel() if int(v) != 0}))
n_classes = int(max(ids)) + 1

# %%
# MSI matrix (background row/column 0) and derived scalar features.
matrix = spatial_interaction_matrix(labels, n_classes=n_classes)
msi_table = pd.DataFrame(
    matrix,
    index=["BG"] + [f"H{i}" for i in range(1, n_classes)],
    columns=["BG"] + [f"H{i}" for i in range(1, n_classes)],
)
print("MSI matrix:")
print(msi_table.round(4).head())
msi_table.head()

features = msi_features_from_matrix(matrix)
print("MSI scalars:", {k: round(v, 4) for k, v in features.items()})

Path("out").mkdir(exist_ok=True)
fig_msi = plot_msi_matrix(matrix, habitat_ids=tuple(range(1, n_classes)))
fig_msi.savefig("out/msi_matrix_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
