"""
Habitats from a single sequence
===============================

**Background.** Many studies have only one usable MR sequence, for
example a single post-contrast series. Habitat analysis still works,
but with one feature column the habitats mean something narrower: they
are intensity strata inside the tumour, not enhancement-kinetics
patterns (which need several phases).

**Purpose.** You will run the approved two-step analysis of
:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec` on the
arterial-phase (LAP) image alone, look at the number of habitats the
elbow rule picks, view the habitat maps of a training patient and of a
new patient labelled with the fitted model, and check what each habitat
means by its LAP intensity distribution.

**When to use.** Only one sequence is available, or you want a
single-sequence baseline to compare with a multi-phase definition.

**Analysis definition.** The approved two-step definition with one
deviation: ``MODALITIES`` holds only ``"LAP"`` instead of the four DCE
phases. Every other stage, the seed, and the training / new-patient
split (subj001 + subj002 train, subj003 is new) are the same as on the
Spec page.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **intensity stratum** -- with one feature, each habitat is a band of
  normalized intensity inside the tumour (for example "darkest third").
* **subject-level normalization** -- winsorize and min-max with each
  patient's own statistics. With one sequence this makes habitats
  *relative within each tumour*: the brightest habitat is the brightest
  part of that tumour, whatever its absolute signal.
* **cohort-level preprocessing** -- 10 equal-width bins learned on the
  pooled training supervoxels, stored in the model, reused on new
  patients.

.. note::

   Two training patients are a demo, not a cohort. A single-sequence
   habitat says nothing about wash-in or wash-out; do not interpret it as
   a perfusion pattern.

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.
"""

# %%
# Load one sequence
# -----------------
# ``MODALITIES`` has one entry, so every voxel has one feature: its LAP
# intensity. Two patients train the model and the third is a new
# patient, exactly as on the four-phase Spec page.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_cluster_validation_from_report, plot_habitat_overlay

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train, new_patient = cohort[:2], cohort[2:3]
print("train:      ", list(train.subject_ids))
print("new patient:", list(new_patient.subject_ids))

# %%
# The same stages, one column
# ---------------------------
# Only ``MODALITIES`` changed. Winsorize and min-max still run per
# patient (subject-level); binning still runs once on the pooled
# training supervoxels (cohort-level) and is stored in the model. With
# one column, binning turns each supervoxel into one of 10 intensity
# levels, so the fitter clusters at most 10 distinct values.
spec = HabitatSpec(
    name="single_sequence_two_step",
    stages=(
        # extract: a single intensity column (LAP), voxels inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # subject-level: clip to each patient's own 1st..99th percentile.
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        # subject-level: rescale to 0..1 with the patient's own min and max.
        # With one column, 0 is the darkest and 1 the brightest part of
        # THIS tumour.
        Stage("preprocess2", Spec("zscore")),
        # partition: 30 supervoxels per tumour.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        # pool: stack the training supervoxels.
        Stage("pool", Spec("pool")),
        # cohort-level: 10 equal-width bins, edges from training rows only.

        # fit: 2..10 habitats, elbow rule.
        Stage(
            "fit",
            Spec(
                "kmeans",
                {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        Stage("graph", Spec("graph", {"include_extended_metrics": False})),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print(result.habitat_model.summary())

# %%
# How many habitats
# -----------------
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
# inertia (HABIT pre-v1.0 elbow); it is not a Thorndike formula. The
# habitat map keeps the Kneedle K; other Ks are only in the table below.
# Full comparison: :doc:`/auto_examples/03_clustering/plot_03_clustering_algorithm`.
# With one feature the Kneedle knee can sit at a different count than
# the four-phase analysis. Read K from the model rather than assuming it.
print("habitats chosen by HABIT elbow/Kneedle:", result.habitat_model.n_habitats)
report = result.habitat_model.preprocessing_state["selection_report"]
candidates = [int(k) for k in report["candidates"]]
inertia = np.asarray(report["scores"][report["methods"][0]], dtype=float)


def discrete_curvature_elbow_k(cluster_range, inertia_scores):
    """Return k maximizing the second difference of inertia (pre-v1.0 elbow)."""
    values = np.asarray(inertia_scores, dtype=float)
    cluster_range = [int(k) for k in cluster_range]
    if values.size < 3:
        return int(cluster_range[int(np.argmin(values))])
    return int(cluster_range[int(np.argmax(np.diff(values, n=2))) + 1])


k_kneedle = int(result.habitat_model.n_habitats)
k_discrete = discrete_curvature_elbow_k(candidates, inertia)
from habit.habitat_model import KMeansHabitatModelFitter

_k_table = {"elbow_kneedle": k_kneedle, "discrete_curvature_elbow": k_discrete}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _fitter = KMeansHabitatModelFitter(
        min_habitats=2, max_habitats=10, validation=_name, n_init=10
    )
    _fitter.set_random_state(0)
    _k_table[_name] = _fitter.fit(result.units, cohort=train).n_habitats
print("K by criterion:")
for _name, _k in _k_table.items():
    print(f"  {_name}: {_k}")

fig = plot_cluster_validation_from_report(
    report, title="Inertia: x = HABIT elbow/Kneedle"
)
fig.axes[0].plot(
    k_discrete,
    float(inertia[candidates.index(k_discrete)]),
    marker="o",
    markersize=8,
    markerfacecolor="none",
    markeredgecolor="C2",
    markeredgewidth=1.4,
    linestyle="none",
    label=f"discrete-curvature k={k_discrete}",
)
fig.axes[0].legend(loc="best", fontsize=8)
fig.savefig("out/single_sequence_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# A training patient
# ------------------
subject = train[0]
habitat_map = result.habitat_maps[0]
fig = plot_habitat_overlay(
    subject.image("LAP"),
    habitat_map,
    title=f"{subject.subject_id}: single-sequence habitats",
    crop_to="labels",
)
fig.savefig("out/single_sequence_train.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# A new patient from the fitted model
# -----------------------------------
# The new patient is normalized with its own statistics, binned with the
# training bin edges stored in the model, and given the training
# centroids. Nothing is refitted.
prediction = Study.from_model(result.habitat_model).predict(new_patient)
fig = plot_habitat_overlay(
    new_patient[0].image("LAP"),
    prediction.habitat_maps[0],
    title=f"{new_patient[0].subject_id}: habitats from the fitted model",
    crop_to="labels",
)
fig.savefig("out/single_sequence_new_patient.png", dpi=150, bbox_inches="tight")
plt.show()

# Fraction of the new patient's ROI voxels in each habitat (label 0 is
# outside the ROI and is dropped).
new_labels = np.asarray(prediction.habitat_maps[0].label_array)
inside = new_labels[new_labels > 0]
ids, counts = np.unique(inside, return_counts=True)
for hid, count in zip(ids, counts):
    print(f"{new_patient[0].subject_id} habitat {int(hid)}: {count / inside.size:.3f}")

# %%
# What each habitat means
# -----------------------
# Plot the raw LAP intensity of the voxels in each habitat for the
# training patient. With one sequence the habitats are adjacent
# intensity bands. The habitat ids are arbitrary names, not a ranking:
# sort by the printed medians to read them from darkest to brightest.
lap = np.asarray(subject.image("LAP").data)
labels = np.asarray(habitat_map.label_array)
fig, ax = plt.subplots(figsize=(6, 4))
for hid in habitat_map.habitat_ids:
    # Voxels of this habitat; label 0 is outside the ROI.
    values = lap[labels == hid]
    ax.hist(values, bins=50, histtype="step", linewidth=1.5, label=f"habitat {hid} (n={values.size})")
    print(f"habitat {hid}: median LAP intensity {np.median(values):.1f}")
ax.set_xlabel("LAP intensity (raw scanner units)")
ax.set_ylabel("Voxel count")
ax.set_title(f"{subject.subject_id}: LAP intensity per habitat")
ax.legend()
fig.savefig("out/single_sequence_intensity_per_habitat.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# How to read the result
# ----------------------
# When this page was built, the LAP-only analysis gave:
#
# * K = 4 habitats by the elbow rule on subj001 + subj002 (the same
#   count as the four-phase definition on the Spec page, but these are
#   not the same habitats: each one is now an intensity band).
# * In subj001 the median raw LAP intensity per habitat was 639
#   (habitat 2), 781 (habitat 4), 924 (habitat 1) and 1068 (habitat 3):
#   four ordered bands, with ids that are names, not ranks.
# * The new patient subj003 had ROI fractions 0.325, 0.156, 0.145 and
#   0.374 for habitats 1 to 4, so every band is present in it.
#
# The histograms overlap because habitats are assigned per supervoxel
# (a patch of voxels), not per voxel, and because normalization is per
# patient. Nothing here says how these bands relate to perfusion.

# %%
# Where to go next
# ----------------
# * The four-phase analysis this page reduces to one column:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
# * Add texture so one sequence gives more than intensity strata:
#   :doc:`/auto_examples/02_features/plot_02_texture_feature_maps`.
# * Choosing the preprocessing:
#   :doc:`/auto_examples/07_advanced/plot_02_subject_preprocess`.
