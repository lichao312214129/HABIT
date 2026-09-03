"""
Feature chains
==============

Clustering operates on **preprocessed feature matrices**, not raw
intensities. :class:`~habit.spec.HabitatSpec` exposes three ordered chains:

* :attr:`~habit.spec.HabitatSpec.voxel_feature_preprocessors` — per
  subject, on voxel features **before** supervoxels / units form
* :attr:`~habit.spec.HabitatSpec.supervoxel_feature_preprocessors` — per
  subject, on supervoxel features **after** supervoxelization
* :attr:`~habit.spec.HabitatSpec.cohort_feature_preprocessors` — fitted
  once on pooled training rows; replayed at apply

This is clustering-feature preprocessing, not image resampling.
"""

# %%
# Load two demo subjects. Change ``DATA`` / ``MODALITIES`` / ``ROI`` for
# your tree.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.pipeline import SubjectPipeline
from habit.pipeline.assembly import build_habitat_components
from habit.spec import HabitatSpec, Spec
from habit.viz import plot_habitat_overlay
import habit.recipes as recipes

DATA = fetch_demo()
MODALITIES = ("LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]
print(f"Cohort: {list(cohort.subject_ids)}; atomic subject: {subject.subject_id}")

# %%
# Declare the three chains. Voxel: winsorize + minmax. Supervoxel:
# z-score. Cohort: uniform binning (fitted on pooled training units).
voxel_chain = (
    Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
    Spec("minmax", {"across_features": False}),
)
supervoxel_chain = (Spec("zscore", {"across_features": False}),)
cohort_chain = (
    Spec("binning", {"n_bins": 6, "bin_strategy": "uniform", "across_features": False}),
)

spec = HabitatSpec(
    name="two_step_with_chains",
    voxel_feature_extractor=Spec("raw", {"modalities": list(MODALITIES)}),
    voxel_feature_preprocessors=voxel_chain,
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
    supervoxel_feature_preprocessors=supervoxel_chain,
    cohort_feature_preprocessors=cohort_chain,
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=7,
)

# Named-field sugar expands to stages: voxel preprocess *before* partition,
# supervoxel preprocess *after* partition / *before* pool, cohort preprocess
# *after* pool. build_habitat_components reads those named fields.
components = build_habitat_components(spec)
pipe = components.pipeline(assigner=None)

# %%
# Voxel-feature table **before** the subject-level chain. Each row is one
# ROI voxel; columns are the raw modality intensities clustering would
# otherwise see.
raw = pipe.voxel_feature_extractor(subject).feature_frame()
print("Before voxel_feature_preprocessors:")
print(raw.head())
raw.head()

# %%
# Same rows after winsorize + minmax. Compare a few cells with the table
# above — the values must change if the chain ran.
processed = pipe.voxel_feature_preprocessor(raw)
print("After winsorize + minmax:")
print(processed.head())
processed.head()

# %%
# Supervoxel feature table **before** z-score. ``SubjectPipeline.units``
# applies ``supervoxel_feature_preprocessors`` last, so omit that slot to
# stop after the supervoxelizer (same parcels clustering will use).
pipe_before_zscore = SubjectPipeline(
    voxel_feature_extractor=components.voxel_feature_extractor,
    supervoxelizer=components.supervoxelizer,
    habitat_assigner=None,
    voxel_feature_preprocessor=components.voxel_feature_preprocessor,
)
sv_before_units = pipe_before_zscore.units(subject)
sv_before = sv_before_units.feature_frame()
print("Supervoxel features before zscore:")
print(sv_before.head())
sv_before.head()

# %%
# Same supervoxel rows **after** z-score. One row per parcel; columns are
# scaled before the cohort chain sees them.
sv_after = pipe.supervoxel_feature_preprocessor(sv_before)
print("Supervoxel features after zscore (before cohort binning):")
print(sv_after.head())
sv_after.head()

# %%
# Two-step study: the cohort chain is fitted on pooled training units and
# stored on :class:`~habit.contracts.HabitatModel`. ``result.pipeline``
# replays that fitted chain at assign (one subject shown here).
result = recipes.Study(spec=spec).fit_predict(cohort)
print(result.habitat_model.summary())
print("Preprocessing state keys:", sorted(result.habitat_model.preprocessing_state.keys()))
print("Habitat-level features:")
print(result.features.frame.head())

# units() on the full pipe includes zscore; assign() then applies cohort binning
zscored_units = pipe.units(subject)
_, binned_units = result.pipeline.assign(zscored_units)
binned = binned_units.feature_frame()
print("After cohort binning (fitted chain replayed on one subject):")
print(binned.head())
binned.head()

# %%
# Overlay the first subject's habitats after all three chains.
fig = plot_habitat_overlay(
    subject.image(MODALITIES[0]),
    result.habitat_maps[0],
    title="habitats",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/habitat_preprocessing_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
