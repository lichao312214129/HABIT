"""
Voxel features
==============

**Background.** The ``extract`` stage decides which numbers describe each
voxel; habitats can only separate what those columns show. The simplest
choice is the image intensities themselves.

**Purpose.** You see the voxel-feature table and matrix that clustering
receives with ``raw``, a habitat overlay fitted on it, and how
``concat`` joins two extractors into one wider table.

**Key terms.**

* **voxel feature** / **extract** -- see
  :doc:`/auto_examples/02_features/plot_01_multi_sequence_intensities`.
* **raw** -- one column per listed modality, the intensities unchanged.
* **concat** -- runs several extractors on the same ROI voxels and joins
  their columns side by side.

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

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.pipeline.assembly import build_habitat_components
from habit.spec import HabitatSpec, Spec
from habit.spec.policy import RunPolicy
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
    # extract: one column per series. Named fields are the same components
    # a Stage list would declare; the complete analysis writes them as stages.
    voxel_feature_extractor=Spec("raw", {"modalities": list(MODALITIES)}),
    # partition: few supervoxels, only so this page can show a map.
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=11,
)
# Build the components the spec names and call only the extractor on one
# subject, to look at the table before any clustering runs.
raw_field = (
    build_habitat_components(raw_spec)
    .pipeline(assigner=None)
    .voxel_feature_extractor(subject)
)
print("raw feature table:")
print(raw_field.feature_frame().head())
raw_field.feature_frame().head()

# %%
# The matrix clustering sees: one row per ROI voxel, one column per
# modality (rows subsampled so the heatmap stays legible).
frame = raw_field.feature_frame()
row_step = max(1, len(frame) // 60)
sample = frame.iloc[::row_step]
fig_matrix, ax_matrix = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)
im = ax_matrix.imshow(sample.to_numpy(dtype=float), aspect="auto", cmap="viridis")
ax_matrix.set_xticks(range(len(sample.columns)))
ax_matrix.set_xticklabels([str(column) for column in sample.columns])
ax_matrix.set_ylabel("ROI voxels (subsampled)")
ax_matrix.set_title("Voxel feature matrix (raw)")
fig_matrix.colorbar(im, ax=ax_matrix, shrink=0.8, label="Intensity")
Path("out").mkdir(exist_ok=True)
fig_matrix.savefig("out/habitat_feature_routes_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Fit the ``raw`` route and overlay habitats.
# fit_predict runs every stage of the spec on both subjects: partition,
# pooled fit (elbow over 2..3 habitats), assign, then the feature stages.
raw_result = recipes.Study(spec=raw_spec).fit_predict(
    cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print(raw_result.habitat_model.summary())
fig = plot_habitat_overlay(
    subject.image(MODALITIES[0]),
    raw_result.habitat_maps[0],
    title="habitats (raw)",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/habitat_feature_routes_overlay.png", dpi=150, bbox_inches="tight")
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
concat_result = recipes.Study(spec=concat_spec).fit_predict(
    cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print(f"concat habitats: {concat_result.habitat_model.n_habitats}")

# %%
# Elbow / Kneedle caveat and criterion table
# ------------------------------------------
# HABIT validation ``elbow`` is the same rule as ``kneedle``. It runs
# ``KneeLocator`` on k-means inertia (within-cluster sum of squares),
# curve convex, direction decreasing (Satopaa, Albrecht, Irwin, and
# Raghavan, 2011, Finding a "Kneedle" in a Haystack, IEEE ICDCS).
# Normalize k and inertia to the unit square, draw the chord from the
# first point to the last, and take the k farthest from that chord; a
# smooth curve often places this knee to the right of the bend a person
# sees. Literature "elbow" means inspecting within-cluster dispersion
# versus k (Thorndike RL, 1953, Who belongs in the family?, Psychometrika
# 18(4):267-276) — not a second-difference formula. The discrete-curvature
# elbow (visual elbow, computed) maximizes the second difference of
# inertia (HABIT pre-v1.0 elbow). Habitat maps keep the Kneedle K.
_report = concat_result.habitat_model.preprocessing_state["selection_report"]
_candidates = [int(k) for k in _report["candidates"]]
_inertia = __import__("numpy").asarray(_report["scores"][_report["methods"][0]], dtype=float)


def _discrete_curvature_elbow_k(cluster_range, inertia_scores):
    """Return k maximizing the second difference of inertia (pre-v1.0 elbow)."""
    import numpy as np
    values = np.asarray(inertia_scores, dtype=float)
    cluster_range = [int(k) for k in cluster_range]
    if values.size < 3:
        return int(cluster_range[int(np.argmin(values))])
    return int(cluster_range[int(np.argmax(np.diff(values, n=2))) + 1])


_k_kneedle = int(concat_result.habitat_model.n_habitats)
_k_discrete = _discrete_curvature_elbow_k(_candidates, _inertia)
from habit.habitat_model import KMeansHabitatModelFitter as _KMeansFitter

_k_table = {"elbow_kneedle": _k_kneedle, "discrete_curvature_elbow": _k_discrete}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _fitter = _KMeansFitter(
        min_habitats=2, max_habitats=3, validation=_name, n_init=3
    )
    _fitter.set_random_state(0)
    _k_table[_name] = _fitter.fit(concat_result.units, cohort=cohort).n_habitats
print("K by criterion (elbow/kneedle=Kneedle; sil/CH maximize; DB minimize; skip gap):")
for _name, _k in _k_table.items():
    print(f"  {_name}: {_k}")
print(f"habitat map uses Kneedle K = {_k_kneedle}")
