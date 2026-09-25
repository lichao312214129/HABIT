"""
Multiregional spatial interaction (MSI)
=======================================

**Background.** Two tumours can have the same habitat fractions but a
different spatial arrangement: habitats mixed together, or kept apart.
MSI measures that arrangement by counting which habitats touch which.

**Purpose.** You get the MSI matrix as a table and heatmap, plus a
dictionary of scalar MSI features (one value per name) ready to use as
columns in a statistics or model table.

**Key terms.**

* **MSI** (multiregional spatial interaction, Wu et al. 2018) -- counts how
  often voxels of each habitat touch voxels of each other habitat (face
  neighbours), describing how habitats are arranged in space.
* **MSI matrix** -- entry ``[i, j]`` is the number of face-neighbour voxel
  pairs labelled ``i`` and ``j``; row/column 0 is background, so it records
  how much of each habitat lies on the tumour border.
* **MSI scalars** -- ``firstorder_*`` are the matrix entries (raw and
  normalised); ``contrast`` / ``homogeneity`` / ``correlation`` / ``energy``
  are texture-style summaries of the normalised matrix.

Atomic MSI from a habitat label map (Wu et al., *Radiology* 2018):
:func:`~habit.kernels.spatial_interaction_matrix` and
:func:`~habit.kernels.msi_features_from_matrix`.
"""

# %%
# Build a habitat map, then compute the MSI matrix and scalar summaries.
# sphinx_gallery_thumbnail_number = 1
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
# Matrix size includes background (class 0), hence the + 1.
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

# Flatten the matrix into named scalars (same keys as HABIT v0.1 MSI output).
features = msi_features_from_matrix(matrix)
print("MSI scalars:", {k: round(v, 4) for k, v in features.items()})

Path("out").mkdir(exist_ok=True)
fig_msi = plot_msi_matrix(matrix, habitat_ids=tuple(range(1, n_classes)))
fig_msi.savefig("out/msi_matrix_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
