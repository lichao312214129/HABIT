"""
Habitats inside each subject with a Spec
========================================

*A per-patient heterogeneity score.*

**Background.** Some questions are about each tumour on its own: *how
heterogeneous is this lesion?* For that you do not need a habitat
definition shared by the cohort. Each tumour can be split into its own
habitats (the one-step design), and the patient is then described by
numbers that do not depend on which habitat got which id: how many
habitats it needed and how fragmented they are.

**Purpose.** You will cluster each of the five demo patients alone,
let the elbow rule pick the number of habitats per patient, and build
a one-row-per-patient table of label-free heterogeneity summaries:
number of habitats, ITH score and MSI energy, with a bar chart of the
ITH score.

**When to use.** Use one-step when your question is a per-patient
heterogeneity score, when patients are too different for one shared
definition to make sense, or when you have a single patient. Do not
use it if you need to compare *habitat 2* across patients (e.g. "the
fraction of the enhancing habitat"): use the two-step design of
:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`, or match the
per-patient habitats first
(:doc:`/auto_examples/05_validation_and_reuse/plot_02_matching_labels`).

**Analysis definition.** The reference analysis of
:doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec` with three
stages removed: no ``partition`` (voxels are clustered directly), no
``pool`` (each subject is fitted alone) and therefore no cohort-level
``binning`` (there are no pooled rows to learn bin edges from). The
subject-level winsorize + z-score stages, the k-means fitter (2..10
candidates, elbow, ``n_init=10``) and seed 0 are unchanged. The graph
family is left out to keep the page fast.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **one-step** -- ``fit`` and ``assign`` run inside each subject on its
  own voxels; there is no ``partition`` and no ``pool``, so every
  patient gets a private model with its own centroids.
* **label switching** -- two independent clusterings number similar
  tissue differently, so habitat 1 of one patient is not habitat 1 of
  another. Ids must be matched before patients are compared by habitat.
* **label-free summary** -- a number that stays the same if the habitat
  ids of a patient are renumbered (e.g. 1<->3). Only those can be
  compared across patients here.
* **ITH score** -- how fragmented the habitats are:
  ``1 - (1 / S_total) * sum_i(S_i,max / n_i)`` with ``S_i,max`` the
  largest connected piece of habitat ``i`` and ``n_i`` its number of
  pieces; 0 = every habitat is one blob.
* **MSI energy** -- a sum of squares over the habitat-contact matrix
  (MSI = how often each pair of habitats, and each habitat with itself,
  touch). Renumbering the ids only permutes the matrix entries, so the
  sum does not change.

.. note::

   The demo has no clinical outcomes. In your own study the table at
   the end is what you join to outcomes (by the ``subject`` column);
   this page stops at the table.
"""

# %%
# Load the cohort
# ---------------
# All five demo patients, four DCE phases. No patient is held out:
# nothing is learned across patients, so there is nothing to validate
# on a new one.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
print(cohort)

# %%
# Declare the one-step analysis
# -----------------------------
# The stage list has no ``partition`` and no ``pool``. Without ``pool``,
# ``fit`` runs once per subject on that subject's voxels, and the elbow
# rule picks the number of habitats separately for every patient.
#
# The two subject-level stages are valid here: they use only the
# patient's own statistics. A cohort-level stage (``binning`` after
# ``pool`` in the two-step page) does not apply: it is learned on the
# pooled training rows, and in this design nothing is pooled.
spec = HabitatSpec(
    name="inside_each_subject",
    stages=(
        # extract: one intensity column per DCE phase, voxels inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # preprocess (subject-level): clip each column to that patient's
        # own 1st..99th percentile.
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        # preprocess2 (subject-level): rescale each column to 0..1 with that
        # patient's own min and max.
        Stage("preprocess2", Spec("zscore")),
        # No partition, no pool, no preprocess_cohort.
        # fit (per subject): k-means on this patient's voxels, 2..10
        # habitats tried, elbow rule, 10 restarts per candidate count.
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
        # assign: nearest centroid of THIS patient's own model.
        Stage("assign", Spec("nearest_centroid")),
        # quantify: volume, MSI and ITH per patient.
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
    ),
    random_seed=0,
)
# Without pool there is no single habitat_model: one model per subject,
# keyed by subject id.
result = Study(spec).fit_predict(
    cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print("cohort-level model:", result.habitat_model)
print("per-subject models:", list(result.subject_models))

# %%
# Each patient picks its own number of habitats
# ---------------------------------------------
# Every private model keeps its own elbow/Kneedle report. Here is the
# first patient's curve. Maps keep that patient's Kneedle K.
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

first_id = cohort[0].subject_id
first_model = result.subject_models[first_id]
report = first_model.preprocessing_state["selection_report"]
candidates = [int(k) for k in report["candidates"]]
inertia = np.asarray(report["scores"][report["methods"][0]], dtype=float)


def discrete_curvature_elbow_k(cluster_range, inertia_scores):
    """Return k maximizing the second difference of inertia (pre-v1.0 elbow)."""
    values = np.asarray(inertia_scores, dtype=float)
    cluster_range = [int(k) for k in cluster_range]
    if values.size < 3:
        return int(cluster_range[int(np.argmin(values))])
    return int(cluster_range[int(np.argmax(np.diff(values, n=2))) + 1])


k_kneedle = int(first_model.n_habitats)
k_discrete = discrete_curvature_elbow_k(candidates, inertia)
from habit.habitat_model import KMeansHabitatModelFitter

# Compare criteria on the first patient's units (skip gap).
_units0 = result.units[0:1]
_k_table = {"elbow_kneedle": k_kneedle, "discrete_curvature_elbow": k_discrete}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _fitter = KMeansHabitatModelFitter(
        min_habitats=2, max_habitats=10, validation=_name, n_init=10
    )
    _fitter.set_random_state(0)
    _k_table[_name] = _fitter.fit(_units0, cohort=cohort[:1]).n_habitats
print(f"{first_id}: K by criterion")
for _name, _k in _k_table.items():
    print(f"  {_name}: {_k}")
print(f"habitat map for {first_id} uses Kneedle K = {k_kneedle}")

fig = plot_cluster_validation_from_report(
    report, title=f"{first_id}: inertia x = HABIT elbow/Kneedle"
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
fig.savefig("out/inside_each_subject_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# The ids present in each map run 1..K for that patient only.
for habitat_map in result.habitat_maps:
    present = sorted(int(v) for v in np.unique(habitat_map.label_array) if int(v) != 0)
    print(habitat_map.subject_id, "habitat ids:", present)

# %%
# The per-patient heterogeneity table
# -----------------------------------
# Habitat ids are private to each patient: habitat 2 in one patient is
# not habitat 2 in another, so per-habitat columns such as
# ``habitat_2_volume_fraction`` must NOT be compared across rows. Keep
# only label-free summaries:
#
# * ``n_habitats`` -- the count chosen by the elbow rule for that patient;
# * ``ith_score`` -- fragmentation, unchanged by renumbering the ids;
# * ``energy`` (MSI) -- a sum of squares over the contact matrix, also
#   unchanged by renumbering.
#
# The other MSI summaries are not used on this page.
features = result.features.frame.set_index("subject")
n_habitats = pd.Series(
    {sid: model.n_habitats for sid, model in result.subject_models.items()},
    name="n_habitats",
)
heterogeneity = pd.concat(
    [n_habitats, features[["ith_score", "energy"]].rename(columns={"energy": "msi_energy"})],
    axis=1,
)
heterogeneity.index.name = "subject"
print(heterogeneity.round(3).to_string())

# %%
# ITH score per patient
# ---------------------
# One bar per patient, with the number of habitats written above it.
# Voxels are clustered directly (no supervoxels), so a small speckle
# of isolated voxels also counts as fragmentation.
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(heterogeneity.index, heterogeneity["ith_score"], color="#4c72b0")
for bar, k in zip(bars, heterogeneity["n_habitats"]):
    # Write K on top of each bar so the two columns are read together.
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"K={int(k)}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_xlabel("Patient")
ax.set_ylabel("ITH score")
# ITH lies in [0, 1); the extra headroom keeps the K labels readable.
ax.set_ylim(0, 1.1)
ax.set_title("Intratumour heterogeneity per patient (one-step habitats)")
fig.savefig("out/inside_each_subject_ith.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Two patients' habitat maps
# --------------------------
# Colours are per patient: the same colour in two maps does not mean
# the same kind of tissue.
for subject, habitat_map in list(zip(cohort, result.habitat_maps))[:2]:
    fig = plot_habitat_overlay(
        subject.image("LAP"),
        habitat_map,
        title=f"{habitat_map.subject_id}: its own habitats",
        crop_to="labels",
    )
    fig.savefig(f"out/inside_each_subject_{habitat_map.subject_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# How to read the result
# ----------------------
# * There is no cohort-level model (``habitat_model`` is ``None``); each
#   of the five patients has its own.
# * The elbow rule chose K = 4 for subj001, subj002 and subj003 and
#   K = 5 for subj004 and subj005 in this run.
# * ITH scores ranged from 0.620 (subj002) to 0.964 (subj005). Clustering
#   single voxels leaves many small fragments, so values near 1 are
#   common in this design; compare patients with each other rather than
#   against a fixed cut-off.
# * Patients with more habitats have more ways to fragment, so read
#   ``ith_score`` together with ``n_habitats``.
# * Five demo patients and no outcomes: these numbers show the workflow,
#   not an association with anything clinical.

# %%
# Where to go next
# ----------------
# * Habitat ids shared by every patient (two-step):
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
# * Shared ids without supervoxels:
#   :doc:`/auto_examples/00_introductory_tutorials/plot_03_pooled_voxels`.
# * Matching per-patient habitats before comparing them:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_02_matching_labels`.
# * This design rebuilt from individual components:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_04_atomic_inside_each_subject`.
# * What ITH and MSI measure: :doc:`/auto_examples/04_quantifying_habitats/plot_03_ith`
#   and :doc:`/auto_examples/04_quantifying_habitats/plot_02_msi`.
