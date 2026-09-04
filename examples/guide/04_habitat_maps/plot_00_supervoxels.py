"""
Two-step habitats with SLIC supervoxels
=======================================

Two-step habitat analysis decouples **local spatial partitioning**
from **cohort-level habitat definition**:

1. **Step 1 (Supervoxelization)**: The tumor ROI is partitioned into spatially
   compact, contiguous supervoxels using SLIC (Simple Linear Iterative Clustering).
   SLIC clusters voxels based jointly on multi-channel feature intensities and 3D
   spatial coordinates, effectively smoothing voxel-level noise while adhering to
   anatomical borders.
2. **Step 2 (Cohort clustering)**: Aggregated feature vectors from all supervoxels
   across all subjects are pooled into a shared population matrix. A cohort-level
   clustering model (such as k-means or GMM) identifies the overarching habitat centroids.
3. **Step 3 (Habitat assignment)**: Each subject's supervoxels are mapped to the
   nearest population centroid (:class:`~habit.habitat_model.NearestCentroidAssigner`),
   yielding the final discrete habitat label maps.

SLIC functions strictly as a spatial partitioner (oversegmentation), not a habitat
clustering algorithm. The biological habitat phenotypes emerge from cohort-wide
clustering of these pooled supervoxel units.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Load two demo subjects to demonstrate cohort-level two-step analysis.
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.supervoxel import SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject0 = cohort[0]
image0 = subject0.image(ROI)
Path("out").mkdir(exist_ok=True)
print(f"Cohort subjects ({len(cohort)}): {list(cohort.subject_ids)}")

# %%
# Step 1: Inspect SLIC supervoxelization on the first subject.
# Raw LAP intensity features are extracted and partitioned into 24 supervoxels.
voxel = RawVoxelFeatures(modalities=list(MODALITIES))
field0 = voxel(subject0)
slic = SlicSupervoxelizer(n_supervoxels=24, compactness=10.0)
slic_units0 = slic(field0)

unit_summary = pd.DataFrame(
    [
        {
            "subject_id": subject0.subject_id,
            "supervoxelizer": "slic",
            "n_supervoxels": int(len(slic_units0.features)),
            "label_min": int(slic_units0.label_array[slic_units0.label_array > 0].min()),
            "label_max": int(slic_units0.label_array.max()),
        }
    ]
)
print("SLIC supervoxel summary:")
print(unit_summary.to_string(index=False))
print("\nFirst 5 supervoxel mean feature vectors:")
print(slic_units0.features.head(5))

fig_slic = plot_habitat_overlay(
    image0,
    slic_units0,
    title="SLIC supervoxels (n=24)",
)
fig_slic.savefig("out/supervoxels_slic.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Step 2 & 3: Configure and run the complete two-step HabitatSpec.
# Notice how the stages chain together:
# extract_voxel_features -> preprocess -> partition (SLIC) -> pool -> fit (k-means) -> assign -> quantify.
two_step_spec = HabitatSpec(
    name="two_step_slic_kmeans",
    stages=(
        Stage(
            "extract_voxel_features",
            Spec("raw", {"modalities": list(MODALITIES)}),
        ),
        Stage(
            "preprocess1",
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        Stage(
            "partition",
            Spec("slic", {"n_supervoxels": 24, "compactness": 10.0}),
        ),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec("kmeans", {"n_habitats": 3, "n_init": 10}),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=42,
)

study_result = Study(spec=two_step_spec).fit_predict(cohort)
print("\nLearned Habitat Model:")
print(study_result.habitat_model.summary())

# %%
# Step 4: Visualize the final habitat label map on Subject 0.
hab_map0 = study_result.habitat_maps[0]
fig_hab = plot_habitat_overlay(
    image0,
    hab_map0,
    title="Habitats (SLIC partition + cohort k-means)",
)
fig_hab.savefig(
    "out/supervoxels_habitats_slic.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %%
# Step 5: Quantify volume fractions of the discovered habitats across the cohort.
cohort_volumes = study_result.features.frame
print("\nCohort habitat volume quantification:")
print(cohort_volumes.to_string(index=False))
cohort_volumes
