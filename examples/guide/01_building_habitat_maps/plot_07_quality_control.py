"""
Quality control of habitat maps
===============================

**Background.** A habitat model always returns a map, even when the
definition behind it is poor. Two failure patterns are common and easy
to miss if you only read the feature table: a **one-block** map, where
most of a tumour falls into a single habitat (often because the scan is
globally brighter or darker than the training scans), and a
**fragmented** map, where habitats break into many tiny specks that no
reader would call a region. Both still produce volume fractions, MSI
and ITH numbers, so they must be caught by looking and by simple checks.

**Purpose.** You will run a short QC checklist on the approved analysis
definition: overlays on several slices and all three axes, the
supervoxel / habitat triptych, the elbow curve that chose the number of
habitats, per-habitat volume fractions per subject, and three numeric
flags per map (one-block, missing habitat, fragmented). Then you will
run a deliberately poor definition next to it and print the same flags
for both.

**When to use.** After every fit and every prediction on new patients,
before the feature table goes into statistics. The flags are cheap; the
figures are for the maps the flags point at.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **analysis definition** -- the approved two-step stage list of
  :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`, fitted on
  subj001 + subj002; subj003..subj005 are labelled with the saved model.
* **one-block flag** -- the largest habitat covers more than 60 % of the
  ROI. The 60 % is a threshold chosen for this page, not a published
  cut-off; set your own from your cohort.
* **missing-habitat flag** -- fewer habitats are present in the map than
  the model has.
* **fragmented flag** -- more than 5 % of ROI voxels sit in connected
  pieces smaller than 10 voxels (face connectivity, the same
  connectivity the ITH score uses). Both numbers are page choices.
* **ITH score** -- 0 when every habitat is one blob, close to 1 when
  habitats split into many pieces; printed next to the flags.

**What varies.** Only subject-level normalization. The "poor"
definition is the approved stage list with the two subject-level stages
(winsorize, min-max) removed; every other stage, parameter and seed is
the same. Because the elbow rule then runs on different features, it
may keep a different number of habitats; that is part of the result.

.. note::

   The demo has five patients: two train the model here and three are
   new. These are demo checks of mechanics, not a validated QC protocol.

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.
"""

# %%
# Load the cohort and split it
# ----------------------------
# subj001 and subj002 define the habitats; subj003..subj005 are new
# patients labelled with the saved model. This is the 5-subject demo
# cohort split in two.
# sphinx_gallery_thumbnail_number = 4
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from habit.contracts import HabitatMap, cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import habitat_region_stats, ith_score
from habit.execution import backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import (
    plot_cluster_validation_from_report,
    plot_habitat_overlay,
    plot_habitat_volume_fractions,
    plot_partition_triptych,
)

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train, new_patients = cohort[:2], cohort[2:5]
print("train:", list(train.subject_ids), " new:", list(new_patients.subject_ids))
Path("out").mkdir(exist_ok=True)


# %%
# Two definitions that differ in one factor
# -----------------------------------------
# ``build_spec`` returns the approved stage list. With
# ``normalize=False`` it drops the two subject-level stages and changes
# nothing else, so any difference between the two runs comes from
# per-patient normalization alone.
def build_spec(name: str, normalize: bool) -> HabitatSpec:
    """Return the approved two-step spec, optionally without normalization.

    Args:
        name: Name stored in the spec and in the run manifest.
        normalize: Keep the subject-level winsorize + z-score stages.

    Returns:
        A :class:`~habit.spec.HabitatSpec` ready for ``Study``.
    """
    stages: List[Stage] = [
        # extract: one intensity column per DCE phase, voxels inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
    ]
    if normalize:
        stages += [
            # subject-level: clip to this patient's own 1st..99th percentile.
            Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
            # subject-level: rescale to 0..1 with this patient's own min / max,
            # so a globally darker scan uses the same range as a brighter one.
            Stage("preprocess2", Spec("zscore")),
        ]
    stages += [
        # partition: 30 supervoxels per tumour.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        # pool: stack the training supervoxels into one matrix.
        Stage("pool", Spec("pool")),
        # cohort-level: 10 equal-width bins, edges learned on the pooled
        # training supervoxels and stored in the model.

        # fit: k-means for the cohort, 2..10 habitats, elbow rule.
        Stage(
            "fit",
            Spec(
                "kmeans",
                {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        # quantify: the volume table is used below for a cross-check.
        Stage("volume", Spec("volume")),
        Stage("ith", Spec("ith_score")),
    ]
    return HabitatSpec(name=name, stages=tuple(stages), random_seed=0)


# %%
# Fit the approved definition and label the new patients
# ------------------------------------------------------
# ``fit_predict`` learns the habitats on the two training patients.
# ``Study.from_model(...).predict`` labels the new patients with those
# centroids and the stored bin edges, without refitting.
spec = build_spec("qc_approved", normalize=True)
result = Study(spec).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
model = result.habitat_model
prediction = Study.from_model(model, spec=spec).predict(new_patients)
print(model.summary())

# %%
# Check 1: the number of habitats
# -------------------------------
# The elbow curve should have a visible bend at the kept count. A curve
# with no bend means the count is a guess; look at neighbouring counts
# before trusting the maps.
report = model.preprocessing_state["selection_report"]
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
    report, title="Approved definition: x = HABIT elbow/Kneedle"
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
fig.savefig("out/quality_control_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Check 2: supervoxels and habitats on one slice
# ----------------------------------------------
# The triptych shows whether habitats follow the supervoxel patches and
# the anatomy. Habitats that cut straight through uniform tissue, or
# supervoxels that ignore visible edges, are worth a second look.
subject = train[0]
habitat_map = result.habitat_maps[0]
fig = plot_partition_triptych(subject.image("LAP"), result.units[0], habitat_map, axis=0)
fig.savefig("out/quality_control_triptych.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Check 3: overlays on all three axes and several slices
# ------------------------------------------------------
# The default overlay draws the three orthogonal planes (each at its
# densest habitat slice). One slice can hide a one-block or speckled
# region, so two more axial slices are drawn near the top and bottom
# quarter of the tumour.
fig = plot_habitat_overlay(
    subject.image("LAP"),
    habitat_map,
    title=f"{subject.subject_id}: habitats, three axes",
    crop_to="labels",
)
fig.savefig("out/quality_control_axes.png", dpi=150, bbox_inches="tight")
plt.show()

# Axial (axis 0) indices at the 25th and 75th percentile of the slices
# that contain habitat voxels. Plain numpy on the label array.
z_with_labels = np.nonzero(np.asarray(habitat_map.label_array) > 0)[0]
for quantile in (25, 75):
    z_index = int(np.percentile(z_with_labels, quantile))
    fig = plot_habitat_overlay(
        subject.image("LAP"),
        habitat_map,
        title=f"{subject.subject_id}: axial slice {z_index}",
        axis=0,
        index=z_index,
        crop_to="labels",
    )
    fig.savefig(f"out/quality_control_axial_{quantile}.png", dpi=150, bbox_inches="tight")
    plt.show()


# %%
# Check 4: numeric flags for every map
# ------------------------------------
# ``qc_flags`` reads only the label array. It reports the fraction of
# each habitat, how many of the model's habitats are present, how many
# connected pieces there are, the share of voxels in tiny pieces and the
# ITH score, and turns three of them into True / False flags.
ONE_BLOCK_MAX = 0.60  # page choice: largest habitat above 60 % of the ROI
TINY_PIECE_VOXELS = 10  # page choice: a "tiny" connected piece
TINY_SHARE_MAX = 0.05  # page choice: more than 5 % of voxels in tiny pieces


def qc_flags(habitat_map: HabitatMap, n_habitats: int) -> Dict[str, object]:
    """Compute simple QC numbers and flags for one habitat map.

    Args:
        habitat_map: Label map to check (``0`` is background).
        n_habitats: Number of habitats in the model that painted the map.

    Returns:
        A flat dict: per-habitat fractions, present count, component
        count, tiny-piece share, ITH score and the three boolean flags.
    """
    labels = np.asarray(habitat_map.label_array)
    inside = labels[labels > 0]
    row: Dict[str, object] = {"subject": habitat_map.subject_id}
    # Fraction of ROI voxels per habitat id; absent habitats report 0.
    fractions = {hid: float(np.mean(inside == hid)) for hid in range(1, n_habitats + 1)}
    for hid, value in fractions.items():
        row[f"h{hid}"] = round(value, 3)
    # Connected pieces per habitat (face connectivity, as the ITH score).
    region_stats = habitat_region_stats(labels)
    tiny_voxels = 0
    for hid in region_stats:
        pieces, _ = ndimage.label(labels == hid)
        sizes = np.bincount(pieces.ravel())[1:]
        tiny_voxels += int(sizes[sizes < TINY_PIECE_VOXELS].sum())
    tiny_share = tiny_voxels / inside.size
    largest = max(fractions.values())
    present = sum(1 for value in fractions.values() if value > 0)
    row.update(
        {
            "largest": round(largest, 3),
            "present": f"{present}/{n_habitats}",
            "pieces": int(sum(n for n, _ in region_stats.values())),
            "tiny_share": round(tiny_share, 3),
            "ith": round(float(ith_score(labels)), 3),
            "one_block": largest > ONE_BLOCK_MAX,
            "missing_habitat": present < n_habitats,
            "fragmented": tiny_share > TINY_SHARE_MAX,
        }
    )
    return row


def volume_cross_check(maps: Tuple[HabitatMap, ...], frame: pd.DataFrame, n_habitats: int) -> bool:
    """True when label-array fractions equal the ``volume`` stage columns.

    Args:
        maps: Habitat maps produced by one run.
        frame: That run's ``features.frame`` (column ``subject``).
        n_habitats: Number of habitats in the model.

    Returns:
        Whether every subject's fractions agree to 1e-6.
    """
    table = frame.set_index("subject")
    for one_map in maps:
        labels = np.asarray(one_map.label_array)
        inside = labels[labels > 0]
        for hid in range(1, n_habitats + 1):
            column = f"habitat_{hid}_volume_fraction"
            stored = float(table.loc[one_map.subject_id, column]) if column in table else 0.0
            if abs(stored - float(np.mean(inside == hid))) > 1e-6:
                return False
    return True


approved_maps = tuple(result.habitat_maps) + tuple(prediction.habitat_maps)
approved_qc = pd.DataFrame([qc_flags(m, model.n_habitats) for m in approved_maps])
print(f"Approved definition (K = {model.n_habitats}):")
print(approved_qc.to_string(index=False))
print(
    "cross-check, fractions == volume stage columns:",
    volume_cross_check(tuple(result.habitat_maps), result.features.frame, model.n_habitats)
    and volume_cross_check(tuple(prediction.habitat_maps), prediction.features.frame, model.n_habitats),
)

# %%
# Check 5: volume fractions per subject
# -------------------------------------
# One bar chart per new patient. A healthy map spreads its voxels over
# several habitats; one tall bar is the one-block pattern.
for one_map in prediction.habitat_maps:
    labels = np.asarray(one_map.label_array)
    inside = labels[labels > 0]
    fractions = {hid: float(np.mean(inside == hid)) for hid in range(1, model.n_habitats + 1)}
    fig = plot_habitat_volume_fractions(fractions, title=f"{one_map.subject_id}: approved definition")
    fig.savefig(f"out/quality_control_fractions_{one_map.subject_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# A deliberately poor definition
# ------------------------------
# Same stages without the subject-level normalization. MR intensities
# are in arbitrary units, so a new patient whose scan is globally darker
# or brighter than the training scans is pushed towards the centroids
# nearest its overall intensity level.
poor_spec = build_spec("qc_no_subject_normalization", normalize=False)
poor_result = Study(poor_spec).fit_predict(
    train,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
poor_model = poor_result.habitat_model
poor_prediction = Study.from_model(poor_model, spec=poor_spec).predict(new_patients)
poor_maps = tuple(poor_result.habitat_maps) + tuple(poor_prediction.habitat_maps)
poor_qc = pd.DataFrame([qc_flags(m, poor_model.n_habitats) for m in poor_maps])
print(f"Poor definition, no subject-level normalization (K = {poor_model.n_habitats}):")
print(poor_qc.to_string(index=False))
print(
    "cross-check, fractions == volume stage columns:",
    volume_cross_check(tuple(poor_result.habitat_maps), poor_result.features.frame, poor_model.n_habitats)
    and volume_cross_check(
        tuple(poor_prediction.habitat_maps), poor_prediction.features.frame, poor_model.n_habitats
    ),
)

# %%
# The flags side by side
# ----------------------
# One row per subject and definition; only the flag columns and the
# numbers they are based on.
columns = ["subject", "largest", "present", "tiny_share", "ith", "one_block", "missing_habitat", "fragmented"]
side_by_side = pd.concat(
    [approved_qc[columns].assign(definition="approved"), poor_qc[columns].assign(definition="no normalization")]
)
print(side_by_side.set_index(["definition", "subject"]).to_string())

# %%
# Look at the worst map under both definitions
# --------------------------------------------
# The new patient with the largest single-habitat fraction under the
# poor definition, drawn under both definitions with the same crop
# window so the two figures show the same anatomy.
poor_new = poor_qc.iloc[len(train):]
worst_id = str(poor_new.loc[poor_new["largest"].idxmax(), "subject"])
worst_subject = next(s for s in new_patients if s.subject_id == worst_id)
approved_worst = next(m for m in prediction.habitat_maps if m.subject_id == worst_id)
poor_worst = next(m for m in poor_prediction.habitat_maps if m.subject_id == worst_id)
# Union of both label maps places one shared crop window.
crop_window = (np.asarray(approved_worst.label_array) > 0) | (np.asarray(poor_worst.label_array) > 0)
for name, one_map, n_habitats in (
    ("approved", approved_worst, model.n_habitats),
    ("no_normalization", poor_worst, poor_model.n_habitats),
):
    fig = plot_habitat_overlay(
        worst_subject.image("LAP"),
        one_map,
        title=f"{worst_id}: {name.replace('_', ' ')} definition",
        crop_to="labels",
        crop_labels=crop_window.astype(np.uint8),
    )
    fig.savefig(f"out/quality_control_worst_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    labels = np.asarray(one_map.label_array)
    inside = labels[labels > 0]
    fractions = {hid: float(np.mean(inside == hid)) for hid in range(1, n_habitats + 1)}
    fig = plot_habitat_volume_fractions(fractions, title=f"{worst_id}: {name.replace('_', ' ')}")
    fig.savefig(f"out/quality_control_worst_{name}_fractions.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# How to read the result
# ----------------------
# Measured on this run (approved definition: K = 4 on subj001 + subj002;
# poor definition without subject-level winsorize / z-score: K = 5):
#
# * Approved maps for all five patients: no one-block, no missing
#   habitat, no fragmented flag. Largest habitat 33--48 % of the ROI;
#   tiny-piece share 0.5--1.4 %; ITH 0.84--0.98.
# * Without subject-level normalization, held-out subj004 and subj005
#   raised the one-block flag (largest habitat 61 % and 67 %) and both
#   missed two of the five habitats. Training patients stayed clear of
#   those flags but the elbow kept a different K.
# * Volume-stage fractions matched the label-array fractions in both
#   runs (cross-check True).
#
# The 60 % / 5 % / 10-voxel thresholds are page choices. Set your own
# from your cohort before treating a flag as a reject rule.

# %%
# Where to go next
# ----------------
# * The approved analysis these checks run on:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
# * Remove tiny pieces from a map that the fragmented flag points at:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_08_cleaning_maps`.
# * How the maps react to a slightly larger or smaller ROI:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_03_precise_features`.
# * Choosing the number of habitats with other criteria:
#   :doc:`/auto_examples/03_clustering/plot_03_clustering_algorithm`.
# * The volume fraction and ITH definitions:
#   :doc:`/auto_examples/04_quantifying_habitats/plot_01_volume_fractions` and
#   :doc:`/auto_examples/04_quantifying_habitats/plot_03_ith`.
