"""
Pooled-voxel habitats with a Spec
=================================

**Background.** The two-step design first cuts every tumour into
supervoxels and clusters those. The pooled-voxel design (also called
direct pooling) skips that step: every ROI voxel of every training
patient goes into one matrix, and one cohort model is fitted on it.
Habitat ids are still shared across patients.

**Purpose.** You will fit one shared habitat model on the voxels of two
demo patients, let the elbow rule pick the number of habitats, view a
habitat map per patient, compare volume fractions across patients, and
look at the arterial-phase intensities inside each habitat.

**When to use.** You want shared ids without the supervoxel step (for
example because supervoxel boundaries are themselves a modelling
choice you want to avoid), and the pooled voxel matrix is small enough
to cluster: it grows with every voxel of every patient, here 44 552
rows for two patients instead of 60 supervoxels. For large cohorts the
two-step design
(:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`) clusters far
fewer rows.

**Analysis definition.** The reference analysis of
:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec` with one stage
removed: no ``partition``. Subject-level winsorize + z-score run on each
patient's voxels BEFORE ``pool``; the cohort-level 10-bin uniform
binning is learned on the pooled training voxels AFTER ``pool`` and
stored in the model. Fitter (k-means, 2..10, elbow, ``n_init=10``),
assigner, feature families and seed 0 are unchanged.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **pooled voxels / direct pooling** -- ``pool`` then ``fit`` with no
  ``partition``; each voxel is its own clustering unit.
* **pool** -- stack every training patient's rows into one matrix, so
  one model (one set of centroids) is learned for the cohort.
* **subject-level vs cohort-level preprocessing** -- before ``pool`` a
  stage uses one patient's own statistics; after ``pool`` it is learned
  once on the training rows and saved inside the model.
"""

# %%
# Load the cohort
# ---------------
# Two training patients, as on the two-step page, so the two designs
# can be compared on the same data.
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
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train = cohort[:2]
print(f"Training patients: {list(train.subject_ids)}")

# %%
# Declare the pooled-voxel analysis
# ---------------------------------
# No ``partition``: each ROI voxel is its own clustering unit. ``pool``
# stacks the voxels of every patient and one model is fitted on them.
# The preprocessing stages keep their meaning from their position:
# before ``pool`` they are subject-level, after it cohort-level.
spec = HabitatSpec(
    name="pooled_voxels",
    stages=(
        # extract: one intensity column per DCE phase, inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # preprocess, preprocess2 (subject-level): winsorize then z-score each
        # column with the patient's own statistics, BEFORE pooling, so a
        # darker scan does not end up in its own habitat.
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        # No partition. pool stacks every patient's ROI voxels.
        Stage("pool", Spec("pool")),
        # preprocess_cohort (cohort-level): 10 uniform bins per feature, edges
        # learned on the pooled training voxels and stored in the model.

        # fit: one shared k-means on the binned voxels; 2..10 candidates,
        # elbow rule, 10 restarts per candidate.
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 10,
                    "validation": "elbow",
                    "n_init": 10,
                },
            ),
        ),
        # assign: every voxel gets the id of its nearest shared centroid.
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        Stage("graph", Spec("graph", {"include_extended_metrics": False})),
    ),
    random_seed=0,
)
# One shared habitat_model, as in two-step, but its centroids were
# learned on voxels rather than supervoxels.
result = Study(spec).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print(result.habitat_model.summary())
# Each clustering unit is one voxel, so the unit count equals the ROI size.
for units in result.units:
    print(units.subject_id, "clustering units:", units.feature_frame().shape[0])

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
# The same HABIT elbow/Kneedle rule as on the two-step page, now scored
# on every pooled voxel instead of on 60 supervoxels.
report = result.habitat_model.preprocessing_state["selection_report"]
Path("out").mkdir(exist_ok=True)
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
print(
    "K by criterion (elbow/kneedle=Kneedle; sil/CH maximize; DB minimize; "
    "discrete-curvature=max second diff):"
)
for _name, _k in _k_table.items():
    print(f"  {_name}: {_k}")
print(f"habitat map uses Kneedle K = {k_kneedle}")

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
fig.savefig("out/pooled_voxels_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# One overlay per patient
# -----------------------
# Ids are shared, so the same colour is the same habitat in both maps.
for subject, habitat_map in zip(train, result.habitat_maps):
    fig = plot_habitat_overlay(
        subject.image("LAP"),
        habitat_map,
        title=f"{habitat_map.subject_id}: habitats from pooled voxels",
        crop_to="labels",
    )
    fig.savefig(f"out/pooled_voxels_{habitat_map.subject_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Feature table
# -------------
# One row per patient. Volume fractions can be compared across rows
# because the ids come from one shared model.
table = result.features.frame.set_index("subject")
columns = [c for c in table.columns if c.endswith("_volume_fraction")] + [
    "ith_score",
    "graph_num_nodes_total",
]
print(table.shape[1], "columns")
print(table[columns].round(3).to_string())

# %%
# Arterial-phase intensity inside each habitat
# --------------------------------------------
# A habitat is defined by all four phases together; this histogram
# shows only one of them (the displayed LAP image, raw values, first
# patient) to give the habitats a physical reading. The label map sits
# on the image grid, so the two arrays can be indexed together.
subject = train[0]
habitat_map = result.habitat_maps[0]
image = np.asarray(subject.image("LAP").data)
labels = np.asarray(habitat_map.label_array)
fig, ax = plt.subplots(figsize=(6.2, 3.2))
for habitat_id in habitat_map.habitat_ids:
    values = image[labels == habitat_id]
    ax.hist(values, bins=30, alpha=0.6, label=f"habitat {habitat_id}")
    print(f"{subject.subject_id} habitat {habitat_id}: median LAP = {np.median(values):.1f}")
ax.set_xlabel("LAP intensity (raw)")
ax.set_ylabel("voxels")
ax.set_title(f"{subject.subject_id}: LAP intensity by habitat")
ax.legend()
fig.savefig("out/pooled_voxels_intensity_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# How to read the result
# ----------------------
# * The model was fitted on 44 552 pooled voxels (34 694 from subj001,
#   9 858 from subj002); the elbow rule kept **K = 4** in this run.
# * Volume fractions are comparable across the two rows. In this run
#   habitat 1 covers 0.256 of subj001 and 0.431 of subj002.
# * ITH scores (0.969 and 0.846) are high: without supervoxels, single
#   voxels at habitat borders form many small fragments.
# * In the histogram, subj001's habitats differ in median LAP intensity
#   (717 to 1005 raw units), but their LAP ranges overlap, because each
#   habitat is separated using all four phases, not LAP alone.
# * The habitat numbering is not the same as on the two-step page: the
#   two designs fit different rows, so "habitat 1" here and there are
#   unrelated unless you match them.

# %%
# Where to go next
# ----------------
# * The same data with supervoxels (two-step):
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
# * Each tumour clustered on its own (one-step):
#   :doc:`/auto_examples/00_introductory_tutorials/plot_02_inside_each_subject`.
# * This design rebuilt from individual components:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_04_atomic_inside_each_subject`.
# * Fitting and assigning in detail:
#   :doc:`/auto_examples/07_advanced/plot_05_fit` and
#   :doc:`/auto_examples/07_advanced/plot_06_assign`.
