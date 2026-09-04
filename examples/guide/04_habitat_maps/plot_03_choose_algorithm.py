"""
Comparing clustering algorithms: k-means vs GMM
================================================

In habitat analysis, the habitat **definition** is established by the
cohort model fitter. The two standard feature-space clustering algorithms are:

* **k-means**: hard spherical partitions minimizing within-cluster inertia.
* **Gaussian Mixture Model (GMM)**: probabilistic ellipsoidal components
  accounting for feature covariances.

Fair comparison via Hungarian overlap matching
----------------------------------------------
To compare spatial partitions directly without confounding by different cluster
counts, both algorithms are evaluated at the same number of habitats (e.g. :math:`k=3`).
Because unsupervised clustering algorithms assign integer labels arbitrarily,
labels are aligned via Hungarian maximum-overlap matching
(:func:`~habit.kernels.habitat_label_match.match_labels_by_overlap`) before
computing per-habitat Dice and the Adjusted Rand Index (ARI).
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Load one demo subject and inspect available model fitters.
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import HabitatModelFitterRegistry
from habit.kernels.habitat_label_match import (
    adjusted_rand_index,
    habitat_dice_from_mapping,
    match_labels_by_overlap,
    present_habitat_ids,
    remap_label_array,
)
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_label_compare, use_style
from habit.viz.labels import sanitize_label

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
RANDOM_SEED = 42
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
Path("out").mkdir(exist_ok=True)
print(f"Cohort: {list(cohort.subject_ids)}")
print("Available fitters:", HabitatModelFitterRegistry.available())

# %%
# Build two-step pipelines: shared SLIC partition stage, differing only in the fitter.
N_HABITATS = 3

shared_stages = (
    Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
    Stage(
        "preprocess1",
        Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
    ),
    Stage("preprocess2", Spec("minmax", {"across_features": False})),
    Stage("partition", Spec("slic", {"n_supervoxels": 24, "compactness": 10.0})),
    Stage("pool", Spec("pool")),
)

kmeans_spec = HabitatSpec(
    name="fit_kmeans",
    stages=shared_stages
    + (
        Stage(
            "fit",
            Spec("kmeans", {"n_habitats": N_HABITATS, "n_init": 10}),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=RANDOM_SEED,
)

gmm_spec = HabitatSpec(
    name="fit_gmm",
    stages=shared_stages
    + (
        Stage(
            "fit",
            Spec("gmm", {"n_habitats": N_HABITATS, "n_init": 10}),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=RANDOM_SEED,
)

kmeans_result = Study(spec=kmeans_spec).fit_predict(cohort)
gmm_result = Study(spec=gmm_spec).fit_predict(cohort)

# %%
# Match GMM habitat IDs onto k-means IDs via Hungarian maximum overlap.
ref = np.asarray(kmeans_result.habitat_maps[0].label_array)
mov = np.asarray(gmm_result.habitat_maps[0].label_array)
mapping = match_labels_by_overlap(ref, mov)
aligned = remap_label_array(
    mov, mapping, reserved_ids=[int(v) for v in present_habitat_ids(ref)]
)

mapping_table = pd.DataFrame(
    [{"gmm_id": int(s), "matched_kmeans_id": int(d)} for s, d in sorted(mapping.items(), key=lambda x: x[1])]
)
dice_records = []
for hid, mid, dice, n_ref, n_mov in habitat_dice_from_mapping(ref, mov, mapping):
    dice_records.append(
        {
            "habitat_id": int(hid),
            "matched_gmm_id": None if mid is None else int(mid),
            "dice": float(dice),
            "kmeans_voxels": int(n_ref),
            "gmm_voxels": int(n_mov),
        }
    )
dice_table = pd.DataFrame(dice_records)

mean_dice = float(dice_table["dice"].mean()) if len(dice_table) else float("nan")
ari = float(adjusted_rand_index(ref, mov))

summary = pd.DataFrame(
    [
        {
            "subject_id": cohort[0].subject_id,
            "n_habitats": N_HABITATS,
            "mean_dice": mean_dice,
            "adjusted_rand_index": ari,
        }
    ]
)

print("Label alignment mapping:")
print(mapping_table.to_string(index=False))
print("\nPer-habitat agreement:")
print(dice_table.to_string(index=False))
print("\nOverall agreement summary:")
print(summary.to_string(index=False))
dice_table

# %%
# Visual comparison: k-means habitats vs matched GMM habitats.
fig_fit = plot_habitat_label_compare(
    cohort[0].image(ROI),
    kmeans_result.habitat_maps[0],
    aligned,
    titles=("k-means habitats", f"GMM habitats (matched, ARI={ari:.3f})"),
    align_labels=False,
)
fig_fit.savefig("out/choose_kmeans_vs_gmm.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Per-habitat Dice agreement bar plot.
with use_style("radiology"):
    fig_bar, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
    habitat_labels = [f"Habitat {row['habitat_id']}" for _, row in dice_table.iterrows()]
    dice_values = dice_table["dice"].to_numpy(dtype=float)
    bars = ax.bar(habitat_labels, dice_values, color="#0072B2", width=0.5)
    ax.axhline(mean_dice, color="#D55E00", linestyle="--", linewidth=1.5, label=f"Mean Dice = {mean_dice:.3f}")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Dice coefficient")
    ax.set_title(sanitize_label(f"k-means vs GMM spatial agreement (k={N_HABITATS})"))
    ax.legend(loc="lower right", frameon=True)
fig_bar.savefig("out/choose_algorithm_dice_bar.png", dpi=150, bbox_inches="tight")
plt.show()
