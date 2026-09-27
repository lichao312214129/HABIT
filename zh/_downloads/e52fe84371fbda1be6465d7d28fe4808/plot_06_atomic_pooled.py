"""
Pooled-voxel habitats from atomic components
============================================

**Background.** :doc:`/auto_examples/01_building_habitat_maps/plot_02_atomic_two_step`
rebuilds the two-step design from HABIT's individual components and
shows it matches ``Study``. The two other designs use the same
components with fewer steps: the per-subject (one-step) design of
:doc:`/auto_examples/00_introductory_tutorials/plot_02_inside_each_subject` fits one
model per patient on that patient's voxels, and the pooled-voxel design
of :doc:`/auto_examples/00_introductory_tutorials/plot_03_pooled_voxels` fits one
cohort model on every patient's voxels. Neither has a supervoxelizer.

**Purpose.** You will write both designs as plain per-subject loops,
then run the corresponding ``HabitatSpec`` with ``Study`` and check
that labels (voxel for voxel), model ids and feature tables agree.


**Elbow / Kneedle caveat.** HABIT ``validation="elbow"`` is the same
rule as ``kneedle`` (Satopaa, Albrecht, Irwin, and Raghavan, 2011,
Finding a "Kneedle" in a Haystack, IEEE ICDCS): KneeLocator on inertia,
curve convex, direction decreasing — not Thorndike's (1953) informal
elbow, and not the pre-v1.0 discrete-curvature second-difference rule.
See :doc:`/user_guide/building_habitat_maps` and
:doc:`/auto_examples/03_clustering/plot_03_clustering_algorithm`.

**When to use.** When you want to place one of these designs inside
your own pipeline, or to see exactly what ``Study`` does for them. For
a normal study, ``Study`` is shorter and records the run manifest.

**Analysis definition.** Both designs follow their Study pages: four
DCE phases, ROI ``LAP``, subject-level winsorize (1 % / 1 %) + z-score,
k-means fitter with 2..10 candidates, elbow rule, ``n_init=10``, nearest
centroid assignment, seed 0. The pooled-voxel design adds the
cohort-level 10-bin uniform binning after pooling; the per-subject
design has none. To keep the page fast, both use two of the smaller
demo patients (subj002 and subj003, not the subj001 + subj002 pair of
the Study pages) and the volume, MSI and ITH families (no graph).

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **voxel units** -- :func:`habit.pipeline.voxel_units` wraps one
  subject's voxel feature table as clustering units in which every ROI
  voxel is its own unit. It is what replaces the supervoxelizer when a
  design clusters voxels directly.
* **fitter / model / assigner** -- the fitter learns centroids from a
  list of units and returns a ``HabitatModel``; the model's assigner
  paints habitat ids onto one subject's units.
* **cohort-level chain** -- preprocessing fitted once on the pooled
  training rows; its fitted state is attached to the model.
"""

# %%
# Load the cohort and build the shared components
# -----------------------------------------------
# Both designs start the same way per subject: raw voxel features, then
# subject-level normalization with that subject's own statistics. The
# components are built once and reused for every subject.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import HabitatMap, Subject, Supervoxelization, cohort_from_directory
from habit.datasets import fetch_demo
from habit.feature_preprocessing import (
    ZScoreScaling,
    SubjectPreprocessingChain,
    Winsorizing,
)
from habit.habitat_features import HabitatVolumeFeatures, IthHabitatFeatures, MsiHabitatFeatures
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.execution import backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_habitat_overlay, plot_cluster_validation_from_report
from habit.voxel_features import RawVoxelFeatures

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
# subj002 and subj003: two of the smaller demo ROIs, because both designs
# cluster every voxel and each design is run twice (atomic + Study).
train = cohort[1:3]
print(f"Patients: {list(train.subject_ids)}")
# The one seed used for every random step. Study uses spec.random_seed.
SEED = 0
Path("out").mkdir(exist_ok=True)
# sphinx_gallery_thumbnail_number = 1

# extract: one intensity column per DCE phase, voxels inside the ROI.
# The extractor aligns the ROI mask onto the image grid if they differ.
extractor = RawVoxelFeatures(modalities=list(MODALITIES), roi=ROI)
# preprocess + preprocess2 (subject-level): no fitted state; each call
# only uses the numbers of the subject it is given.
subject_chain = SubjectPreprocessingChain([Winsorizing(winsor_limits=(0.01, 0.01)), ZScoreScaling()])
# quantify: three habitat feature families, one row per subject each.
feature_extractors = [HabitatVolumeFeatures(), MsiHabitatFeatures(), IthHabitatFeatures()]


def make_fitter() -> KMeansHabitatModelFitter:
    """
    Build the seeded k-means habitat fitter used by both designs.

    Returns:
        A fitter that tries 2..10 habitats and keeps the elbow. It is
        seeded because k-means starts from random centres; without the
        seed a rerun could give a different map.
    """
    fitter = KMeansHabitatModelFitter(min_habitats=2, max_habitats=10, validation="elbow", n_init=10)
    fitter.set_random_state(SEED)
    return fitter


def normalized_voxel_units(subject: Subject) -> Supervoxelization:
    """
    Turn one subject into subject-normalized, one-voxel clustering units.

    Args:
        subject: One :class:`~habit.contracts.Subject` of the cohort.

    Returns:
        A ``Supervoxelization`` in which every ROI voxel is its own unit,
        with winsorized + min-maxed features.
    """
    field = extractor(subject)
    # Swap in the normalized table; the label and fingerprint go into
    # the provenance of everything computed from it.
    field = field.with_feature_frame(
        subject_chain(field.feature_frame()),
        produced_by="feature_preprocessing.subject.voxel",
        spec_fingerprint=subject_chain.spec.fingerprint(),
    )
    # No supervoxelizer: each voxel becomes its own clustering unit.
    return voxel_units(field)


def feature_row(subject: Subject, habitat_map: HabitatMap) -> pd.DataFrame:
    """
    Compute one subject's habitat feature row.

    Args:
        subject: The subject providing the images.
        habitat_map: That subject's ``HabitatMap``.

    Returns:
        A one-row DataFrame with the volume, MSI and ITH columns joined.
    """
    table = feature_extractors[0](subject, habitat_map)
    for feature_extractor in feature_extractors[1:]:
        table = table.join(feature_extractor(subject, habitat_map))
    return table.frame


def same_table(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """
    Check two feature tables have the same columns and numeric values.

    Args:
        a: First table (one row per subject).
        b: Second table, same row order.

    Returns:
        ``True`` when column names match and numeric values are allclose
        (NaN equal to NaN).
    """
    if list(a.columns) != list(b.columns):
        return False
    numeric = a.select_dtypes("number").columns
    return bool(
        np.allclose(a[numeric].to_numpy(dtype=float), b[numeric].to_numpy(dtype=float), equal_nan=True)
    )


# %%
# Per-subject design: the atomic loop
# -----------------------------------
# Every subject gets its own fitter call on its own voxels: no pooling,
# and no cohort-level chain, because there are no pooled rows to learn
# bin edges from. ``fit`` is called without ``cohort=``: the model is
# not defined by a cohort, only by this one subject's units.
one_step_models = {}
one_step_maps = []
one_step_rows = []
fitter = make_fitter()
for subject in train:
    units = normalized_voxel_units(subject)
    # A private model for this patient; the elbow rule runs per patient.
    model = fitter.fit([units])
    one_step_models[subject.subject_id] = model
    # The assigner of THIS patient's model labels this patient only.
    habitat_map = model.assigner("nearest_centroid")(units)
    one_step_maps.append(habitat_map)
    one_step_rows.append(feature_row(subject, habitat_map))
    print(subject.subject_id, "voxel units:", units.feature_frame().shape[0], "K =", model.n_habitats)
one_step_table = pd.concat(one_step_rows, ignore_index=True)

# %%
# Per-subject design: check against Study
# ---------------------------------------
# The same design declared as a spec (no partition, no pool, no
# preprocess_cohort), run with ``Study`` on the same two patients.
one_step_spec = HabitatSpec(
    name="inside_each_subject",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
    ),
    random_seed=SEED,
)
one_step_result = Study(one_step_spec).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
for atomic_map, study_map in zip(one_step_maps, one_step_result.habitat_maps):
    sid = atomic_map.subject_id
    same_labels = np.array_equal(atomic_map.label_array, study_map.label_array)
    same_id = one_step_models[sid].model_id == one_step_result.subject_models[sid].model_id
    print(f"per-subject {sid}: labels identical = {same_labels}, model_id equal = {same_id}")
print(f"per-subject feature table: same = {same_table(one_step_table, one_step_result.features.frame)}")

# %%
# Elbow / Kneedle caveat (first training patient)
# -----------------------------------------------
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
# inertia (HABIT pre-v1.0 elbow). Each patient's map keeps that patient's
# Kneedle K.
_sid0 = train[0].subject_id
_model0 = one_step_models[_sid0]
_report = _model0.preprocessing_state["selection_report"]
_candidates = [int(k) for k in _report["candidates"]]
_inertia = np.asarray(_report["scores"][_report["methods"][0]], dtype=float)
_k_kneedle = int(_model0.n_habitats)
_k_discrete = (
    int(_candidates[int(np.argmax(np.diff(_inertia, n=2))) + 1])
    if len(_inertia) >= 3
    else _k_kneedle
)
from habit.habitat_model import KMeansHabitatModelFitter as _KMeansFitter

_units0 = [normalized_voxel_units(train[0])]
_k_table = {"elbow_kneedle": _k_kneedle, "discrete_curvature_elbow": _k_discrete}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _f = _KMeansFitter(min_habitats=2, max_habitats=10, validation=_name, n_init=10)
    _f.set_random_state(SEED)
    _k_table[_name] = _f.fit(_units0).n_habitats
print(f"{_sid0}: K by criterion (skip gap)")
for _name, _k in _k_table.items():
    print(f"  {_name}: {_k}")

_criteria = ("elbow", "silhouette", "calinski_harabasz", "davies_bouldin")
_vote = _KMeansFitter(
    min_habitats=2, max_habitats=10, validation=list(_criteria), n_init=10
)
_vote.set_random_state(SEED)
_vote_report = dict(_vote.fit(_units0).preprocessing_state["selection_report"])
_vote_report["selected"] = {
    "elbow": _k_kneedle,
    "silhouette": _k_table["silhouette"],
    "calinski_harabasz": _k_table["calinski_harabasz"],
    "davies_bouldin": _k_table["davies_bouldin"],
}
fig = plot_cluster_validation_from_report(
    _vote_report,
    title=f"{_sid0}: K criteria (elbow panel: x = Kneedle)",
)
fig.axes[0].plot(
    _k_discrete,
    float(_inertia[_candidates.index(_k_discrete)]),
    marker="o",
    markersize=8,
    markerfacecolor="none",
    markeredgecolor="C2",
    markeredgewidth=1.4,
    linestyle="none",
    label=f"discrete-curvature k={_k_discrete}",
)
fig.axes[0].legend(loc="best", fontsize=8)
fig.savefig("out/atomic_pooled_k_criteria.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Pooled-voxel design: the atomic loop
# ------------------------------------
# Same per-subject start. Then the voxel units of all training patients
# are pooled, the cohort-level chain learns its bin edges on those
# pooled TRAINING rows, and ONE fitter call defines the habitats for
# the cohort. ``cohort=train`` fingerprints the training patients into
# the model.
units = [normalized_voxel_units(subject) for subject in train]
pooled = pd.concat([one.feature_frame() for one in units], ignore_index=True)
print("pooled training voxels:", pooled.shape)
# No cohort-level preprocessing after subject z-score.
pooled_ready_units = list(units)
pooled_model = make_fitter().fit(pooled_ready_units, cohort=train)
print(pooled_model.summary())
assigner = pooled_model.assigner("nearest_centroid")
pooled_maps = [assigner(one) for one in pooled_ready_units]
pooled_table = pd.concat(
    [feature_row(subject, habitat_map) for subject, habitat_map in zip(train, pooled_maps)],
    ignore_index=True,
)

# %%
# Pooled-voxel design: check against Study
# ----------------------------------------
# The spec of the pooled-voxel page (no partition), with the same three
# feature families as the atomic loop.
pooled_spec = HabitatSpec(
    name="pooled_voxels",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        Stage("pool", Spec("pool")),

        Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
    ),
    random_seed=SEED,
)
pooled_result = Study(pooled_spec).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
print(f"pooled model_id: atomic {pooled_model.model_id} | Study {pooled_result.habitat_model.model_id}")
print(f"pooled model_id equal = {pooled_model.model_id == pooled_result.habitat_model.model_id}")
for atomic_map, study_map in zip(pooled_maps, pooled_result.habitat_maps):
    same_labels = np.array_equal(atomic_map.label_array, study_map.label_array)
    print(f"pooled {atomic_map.subject_id}: labels identical = {same_labels}")
# Centroids: the habitat definition itself, in the binned feature space.
same_centroids = np.allclose(
    np.asarray(pooled_model.centroids, dtype=float),
    np.asarray(pooled_result.habitat_model.centroids, dtype=float),
)
print(f"pooled centroids allclose = {same_centroids}")
print(f"pooled feature table: same = {same_table(pooled_table, pooled_result.features.frame)}")

# %%
# The two designs on one patient
# ------------------------------
# Same patient, same features, two definitions. Left: habitats private
# to this patient. Right: habitats shared with the other training
# patient. The colours are NOT comparable between the two maps.
subject = train[0]
for name, habitat_map in (("per-subject", one_step_maps[0]), ("pooled voxels", pooled_maps[0])):
    fig = plot_habitat_overlay(
        subject.image("LAP"),
        habitat_map,
        title=f"{subject.subject_id}: {name} habitats (atomic)",
        crop_to="labels",
    )
    stem = name.replace(" ", "_").replace("-", "_")
    fig.savefig(f"out/atomic_other_designs_{stem}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# How to read the result
# ----------------------
# * Per-subject design: the elbow rule chose K = 4 for subj002
#   (9 858 voxels) and K = 4 for subj003 (6 156 voxels), each on its
#   own voxels. Atomic and Study labels were identical voxel for voxel
#   for both patients, their per-subject model ids were equal, and the
#   feature tables matched.
# * Pooled-voxel design: 16 014 pooled voxels, K = 5 in this run. Model
#   id, centroids, labels of both patients and the feature table all
#   matched ``Study``.
# * The model id is a digest of the fitter settings and the training
#   subjects, so its equality confirms that both paths fitted the same
#   thing on the same patients; the identical labels and centroids
#   confirm that they also computed the same result.
# * The pooled-voxel K (5) differs from the per-subject K (4) because
#   the rows being clustered differ: two patients together versus one
#   at a time. It is not a contradiction between the two paths.

# %%
# Where to go next
# ----------------
# * The two-step design from atomic components:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_02_atomic_two_step`.
# * The Study pages for these designs:
#   :doc:`/auto_examples/00_introductory_tutorials/plot_02_inside_each_subject` and
#   :doc:`/auto_examples/00_introductory_tutorials/plot_03_pooled_voxels`.
# * Atomic components on a parallel backend:
#   :doc:`/auto_examples/07_advanced/plot_08_backends`.
# * Each component with its options: :doc:`/auto_examples/07_advanced/index`.
