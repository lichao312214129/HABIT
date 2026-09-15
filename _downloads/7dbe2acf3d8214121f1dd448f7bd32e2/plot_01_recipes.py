"""
Recipes
=======

Three :class:`~habit.recipes.Study` designs. Same scaffolding: load a
cohort, declare :class:`~habit.spec.HabitatSpec` stages, call
:meth:`~habit.recipes.Study.fit_predict`.

Two-step includes the clustering-feature preprocessor chain used in the
Quickstart.

* **two_step** — ``partition`` + ``pool``: shared cohort definition;
  supervoxels first (typical paper pipeline).
* **direct_pooling** — ``pool`` only: shared cohort, cluster voxels
  (no supervoxels).
* **one_step** — neither: habitats **per subject**; integer ids are
  permuted — match labels before comparing patients.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Load two demo subjects.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import (
    plot_cluster_validation_from_report,
    plot_habitat_overlay,
    plot_partition_triptych,
)
import habit.recipes as recipes

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

# %%
# Two-step: ``partition`` + ``pool``. Supervoxels first, then a shared
# cohort definition. Voxel winsorize + minmax run before parcels form.
two_step = HabitatSpec(
    name="habitat_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage(
            "preprocess1",
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 5})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 6,
                    "validation": "elbow",
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
    ),
    random_seed=42,
)
two_step_result = recipes.Study(spec=two_step).fit_predict(cohort)
print(two_step_result.habitat_model.summary())
print(two_step_result.features.frame.head())
two_step_result.features.frame.head()

fig = plot_habitat_overlay(
    cohort[0].image(ROI),
    two_step_result.habitat_maps[0],
    title="habitats (two-step)",
    axis=0,
    crop_to="labels",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/two_step_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# ImageVolume + supervoxelization + HabitatMap (not ``.data``).
fig_tri = plot_partition_triptych(
    cohort[0].image(ROI),
    two_step_result.units[0],
    two_step_result.habitat_maps[0],
    axis=0,
)
fig_tri.savefig("out/two_step_triptych.png", dpi=150, bbox_inches="tight")
plt.show()

# Auto-K / elbow: cohort HabitatModel. Volume / MSI / ITH figures live in
# Habitat Guide 5. Quantify — this page only saves the selection curves.
report = (two_step_result.habitat_model.preprocessing_state or {}).get(
    "selection_report"
)
assert report is not None
fig_k = plot_cluster_validation_from_report(report)
fig_k.savefig("out/two_step_cluster_validation.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# One-step: neither ``partition`` nor ``pool``. Cluster voxels inside
# each subject (no supervoxels). Integer ids are per-subject.
one_step = HabitatSpec(
    name="habitat_one_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage(
            "preprocess1",
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 6,
                    "validation": "elbow",
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=42,
)
one_step_result = recipes.Study(spec=one_step).fit_predict(cohort[:1])
print(f"Per-subject models: {list(one_step_result.subject_models)}")
print(one_step_result.features.frame.head())
one_step_result.features.frame.head()

fig_one = plot_habitat_overlay(
    cohort[0].image(ROI),
    one_step_result.habitat_maps[0],
    title="habitats (one-step)",
)
fig_one.savefig("out/one_step_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# One-step has no cohort HabitatModel; the report is on this subject's model.
one_model = one_step_result.subject_models[cohort[0].subject_id]
one_report = (one_model.preprocessing_state or {}).get("selection_report")
assert one_report is not None
fig_one_k = plot_cluster_validation_from_report(one_report)
fig_one_k.savefig("out/one_step_cluster_validation.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Direct-pooling: ``pool`` only. Skip the cluster partition and pool
# existing voxel units across the cohort.
direct = HabitatSpec(
    name="habitat_direct_pooling",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 6,
                    "validation": "elbow",
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=42,
)
direct_result = recipes.Study(spec=direct).fit_predict(cohort[:1])
print(direct_result.habitat_model.summary())
print(direct_result.features.frame.head())
direct_result.features.frame.head()

fig_direct = plot_habitat_overlay(
    cohort[0].image(ROI),
    direct_result.habitat_maps[0],
    title="habitats (direct-pooling)",
)
fig_direct.savefig("out/direct_pooling_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

report = (direct_result.habitat_model.preprocessing_state or {}).get(
    "selection_report"
)
assert report is not None
fig_direct_k = plot_cluster_validation_from_report(report)
fig_direct_k.savefig(
    "out/direct_pooling_cluster_validation.png", dpi=150, bbox_inches="tight"
)
plt.show()
