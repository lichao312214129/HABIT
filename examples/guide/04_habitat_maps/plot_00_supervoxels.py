"""
Supervoxels (SLIC vs k-means)
=============================

Two-step habitats first **partition** the ROI into supervoxels, then fit
habitats on supervoxel summaries. Partition choice is not the habitat
definition — the cohort fitter is — but SLIC vs feature-space k-means
change the units that get pooled.

* :class:`~habit.supervoxel.SlicSupervoxelizer` — spatially coherent
  parcels (skimage SLIC).
* :class:`~habit.supervoxel.KMeansSupervoxelizer` — parcels in voxel
  feature space (no spatial compactness term).

Both feed the same ``pool`` → ``fit`` → ``assign`` two-step chain.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Load one demo subject and build a small raw voxel field (LAP intensity).
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.supervoxel import KMeansSupervoxelizer, SlicSupervoxelizer
from habit.viz import plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
Path("out").mkdir(exist_ok=True)
print(f"Subject: {subject.subject_id}")

voxel = RawVoxelFeatures(modalities=list(MODALITIES))
field = voxel(subject)
print(field.feature_frame().head())
field.feature_frame().head()

# %%
# SLIC vs k-means partitions on the same field (small ``n_supervoxels``
# keeps the gallery fast). Print unit counts; overlays show parcel ids.
n_supervoxels = 24
slic = SlicSupervoxelizer(n_supervoxels=n_supervoxels, compactness=10.0)
kmeans = KMeansSupervoxelizer(n_supervoxels=n_supervoxels, n_init=3)
kmeans.set_random_state(0)

slic_units = slic(field)
kmeans_units = kmeans(field)
partition_table = pd.DataFrame(
    [
        {
            "partition": "slic",
            "n_units": int(len(slic_units.features)),
            "label_max": int(slic_units.label_array.max()),
        },
        {
            "partition": "kmeans",
            "n_units": int(len(kmeans_units.features)),
            "label_max": int(kmeans_units.label_array.max()),
        },
    ]
)
print(partition_table.to_string(index=False))
partition_table

image = subject.image(ROI)
fig_slic = plot_habitat_overlay(
    image,
    slic_units,
    title="supervoxels (SLIC)",
)
fig_slic.savefig("out/supervoxels_slic.png", dpi=150, bbox_inches="tight")
plt.show()

fig_kmeans = plot_habitat_overlay(
    image,
    kmeans_units,
    title="supervoxels (k-means)",
)
fig_kmeans.savefig("out/supervoxels_kmeans.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Same two-step habitat recipe; only the ``partition`` Spec name changes.
# Habitats are the scientific labels; supervoxels are the intermediate units.


def _two_step_spec(partition_name: str, seed: int) -> HabitatSpec:
    """Build a minimal two-step HabitatSpec for the given partition."""
    partition_params: Dict[str, object]
    if partition_name == "slic":
        partition_params = {"n_supervoxels": 12, "compactness": 10.0}
    else:
        partition_params = {"n_supervoxels": 12, "n_init": 3}
    return HabitatSpec(
        name=f"two_step_{partition_name}",
        stages=(
            Stage(
                "extract_voxel_features",
                Spec("raw", {"modalities": list(MODALITIES)}),
            ),
            Stage(
                "preprocess1",
                Spec(
                    "winsorize",
                    {"winsor_limits": (0.05, 0.05), "across_features": False},
                ),
            ),
            Stage("preprocess2", Spec("minmax", {"across_features": False})),
            Stage("partition", Spec(partition_name, partition_params)),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "elbow",
                        "n_init": 3,
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
        ),
        random_seed=seed,
    )


slic_result = Study(spec=_two_step_spec("slic", seed=11)).fit_predict(cohort)
kmeans_result = Study(spec=_two_step_spec("kmeans", seed=11)).fit_predict(cohort)
print("SLIC partition -> habitats:")
print(slic_result.habitat_model.summary())
print("k-means partition -> habitats:")
print(kmeans_result.habitat_model.summary())

fig_hab_slic = plot_habitat_overlay(
    image,
    slic_result.habitat_maps[0],
    title="habitats (partition=slic)",
)
fig_hab_slic.savefig(
    "out/supervoxels_habitats_slic.png", dpi=150, bbox_inches="tight"
)
plt.show()

fig_hab_kmeans = plot_habitat_overlay(
    image,
    kmeans_result.habitat_maps[0],
    title="habitats (partition=kmeans)",
)
fig_hab_kmeans.savefig(
    "out/supervoxels_habitats_kmeans.png", dpi=150, bbox_inches="tight"
)
plt.show()
