"""
Voxel features
==============

Clustering uses the voxel field you define — not a fixed T1 image.
This page is the intensity pair: ``raw`` (concatenate modality intensities
inside the ROI) and ``concat`` (join families column-wise).

Custom formulas and plugins live on the next page. Texture maps follow
that. This page is only ``raw`` vs ``concat``.
"""

# %%
# Load two demo subjects. ``raw`` concatenates LAP and PVP intensities
# inside the ROI.
from pathlib import Path
import os

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.pipeline.assembly import build_habitat_components
from habit.spec import HabitatSpec, Spec
from habit.viz import plot_habitat_overlay
import habit.recipes as recipes

DATA = fetch_demo()
MODALITIES = ("LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

# %%
# ``raw(modalities)``: one column per series. Print the first voxel rows
# so you can see the intensities clustering will use (before any
# feature preprocessor).
raw_spec = HabitatSpec(
    name="route_raw",
    voxel_feature_extractor=Spec("raw", {"modalities": list(MODALITIES)}),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=11,
)
raw_field = (
    build_habitat_components(raw_spec)
    .pipeline(assigner=None)
    .voxel_feature_extractor(subject)
)
print("raw feature table:")
print(raw_field.feature_frame().head())
raw_field.feature_frame().head()

# %%
# Fit the ``raw`` route and overlay habitats.
raw_result = recipes.Study(spec=raw_spec).fit_predict(cohort)
print(raw_result.habitat_model.summary())
fig = plot_habitat_overlay(
    subject.image(MODALITIES[0]),
    raw_result.habitat_maps[0],
    title="habitats (raw)",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/habitat_feature_routes_overlay.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()

# %%
# ``concat`` joins two ``raw`` extractors column-wise. The head() table
# should show the same number of rows and more columns.
m0, m1 = MODALITIES
concat_spec = HabitatSpec(
    name="route_concat",
    voxel_feature_extractor=Spec(
        "concat",
        {
            "extractors": [
                {"name": "raw", "params": {"modalities": [m0]}},
                {"name": "raw", "params": {"modalities": [m1]}},
            ],
        },
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=11,
)
concat_field = (
    build_habitat_components(concat_spec)
    .pipeline(assigner=None)
    .voxel_feature_extractor(subject)
)
print("concat feature table:")
print(concat_field.feature_frame().head())
concat_field.feature_frame().head()
concat_result = recipes.Study(spec=concat_spec).fit_predict(cohort)
print(f"concat habitats: {concat_result.habitat_model.n_habitats}")
