"""
Robustness to segmentation
==========================

**Background.** Two readers rarely draw the same tumour outline. One
traces a little wider, the other a little tighter. If a habitat map
changes a lot when the outline moves by one or two millimetres, the
habitat features built on it describe the reader as much as the
tumour. Uniformly growing or shrinking the ROI is the standard way to
simulate that systematic contour difference (MIRP's
``perturbation_roi_adapt_size``).

**Purpose.** You will train the approved habitat model once, shrink and
grow the ROI of two new patients by 1 and 2 mm, label the original and
every perturbed ROI with the **same** saved model, and measure how much
the map changes: the ROI size, the volume fraction of each habitat, and
the agreement of habitat ids on the voxels that both ROIs contain
(fraction of identical ids and adjusted Rand index, ARI).

**When to use.** Before trusting habitat features in a study with
manual or semi-automatic segmentation, and when a reviewer asks how
sensitive the habitats are to the outline.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **analysis definition** -- the approved two-step stage list of
  :doc:`/auto_examples/01_complete/plot_01_two_step_spec`, fitted on
  subj001 + subj002. subj003 and subj004 are the patients whose ROI is
  perturbed (the two smallest ROIs, which keeps the page fast).
* **what varies** -- only the ROI mask of the new patients. Images,
  model, centroids, bin edges and seeds are unchanged.
* **morphological perturbation** --
  :class:`~habit.precision.MorphologicalPerturbation` dilates
  (``grow_mm > 0``) or erodes (``grow_mm < 0``) every ROI with a
  6-connected structuring element; the image is untouched. The radius in
  mm becomes an iteration count via the smallest voxel spacing (the demo
  is 1 mm isotropic, so 2 mm = 2 iterations).
* **common ROI** -- voxels inside both the original and the perturbed
  ROI. Habitat ids come from one model, so they are directly comparable
  there; no label matching is needed.
* **ARI** -- chance-corrected agreement of two partitions of the same
  voxels (1 = identical, about 0 = random).

Why can ids change inside the common ROI at all? The subject-level
stages (winsorize, min-max) and the 30 supervoxels are recomputed from
the voxels of the new ROI, so a different outline changes the
normalization and the patches, and through them the assigned habitat.

.. note::

   Two perturbed patients from the 5-subject demo are a demonstration of
   the mechanics, not a robustness study. An ICC of habitat features
   across outlines needs many patients, so it is not computed here.

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.
"""

# %%
# Load the cohort and split it
# ----------------------------
# subj001 and subj002 train the model; subj003 and subj004 are the new
# patients whose outline is perturbed. This is the 5-subject demo
# cohort split in two (subj005 is not used).
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import Cohort, HabitatMap, cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import adjusted_rand_index
from habit.precision import MorphologicalPerturbation
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_label_compare

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train, new_patients = cohort[:2], cohort[2:4]
print("train:", list(train.subject_ids), " perturbed:", list(new_patients.subject_ids))
Path("out").mkdir(exist_ok=True)

# %%
# Train the approved model once
# -----------------------------
# The same stage list as the complete analysis page. The model is fitted
# on the original training ROIs and never refitted below.
spec = HabitatSpec(
    name="segmentation_robustness_two_step",
    stages=(
        # extract: one intensity column per DCE phase, voxels inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # subject-level: clip to the patient's own 5th..95th percentile.
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.05, 0.05]})),
        # subject-level: rescale to 0..1 with the patient's own min / max.
        Stage("preprocess2", Spec("minmax")),
        # partition: 30 supervoxels per tumour.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        Stage("pool", Spec("pool")),
        # cohort-level: bin edges learned on the training supervoxels.
        Stage("preprocess_cohort", Spec("binning", {"n_bins": 10, "bin_strategy": "uniform"})),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        # volume: the per-habitat fractions compared below.
        Stage("volume", Spec("volume")),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(train)
model = result.habitat_model
# One Study object labels every ROI version, so all maps share centroids.
labeller = Study.from_model(model, spec=spec)
print(model.summary())

# %%
# Perturb the outline and relabel
# -------------------------------
# ``MorphologicalPerturbation(grow_mm=g)`` returns a copy of the subject
# with every mask grown (g > 0) or shrunk (g < 0). ``grow_mm=0`` is a
# no-op and serves as a check that relabelling is deterministic. The
# ``rng`` is required by the interface but unused when ``grow_mm`` is
# fixed.
GROW_MM = (-2.0, -1.0, 0.0, 1.0, 2.0)
maps: Dict[float, List[HabitatMap]] = {}
subjects_by_grow: Dict[float, Cohort] = {}
for grow_mm in GROW_MM:
    perturb = MorphologicalPerturbation(grow_mm=grow_mm)
    perturbed = Cohort(
        subjects=tuple(perturb(s, rng=np.random.default_rng(0)) for s in new_patients)
    )
    subjects_by_grow[grow_mm] = perturbed
    maps[grow_mm] = list(labeller.predict(perturbed).habitat_maps)


# %%
# Measure the change
# ------------------
# For every patient and radius: ROI size, per-habitat volume fraction,
# the largest absolute change of any fraction against the original ROI,
# and the id agreement on the common ROI.
def fractions_of(habitat_map: HabitatMap, n_habitats: int) -> np.ndarray:
    """Fraction of ROI voxels in each habitat id 1..n_habitats.

    Args:
        habitat_map: Label map (``0`` is background).
        n_habitats: Number of habitats in the model.

    Returns:
        Array of length ``n_habitats`` that sums to 1.
    """
    labels = np.asarray(habitat_map.label_array)
    inside = labels[labels > 0]
    return np.array([np.mean(inside == hid) for hid in range(1, n_habitats + 1)])


rows = []
for index, subject in enumerate(new_patients):
    reference = maps[0.0][index]
    ref_labels = np.asarray(reference.label_array)
    ref_fractions = fractions_of(reference, model.n_habitats)
    for grow_mm in GROW_MM:
        moved = maps[grow_mm][index]
        moved_labels = np.asarray(moved.label_array)
        # Common ROI: voxels labelled in both maps (both ROIs contain them).
        common = (ref_labels > 0) & (moved_labels > 0)
        fractions = fractions_of(moved, model.n_habitats)
        row = {
            "subject": subject.subject_id,
            "grow_mm": grow_mm,
            "roi_voxels": int(np.count_nonzero(moved_labels > 0)),
        }
        row.update({f"h{hid}": round(float(f), 3) for hid, f in enumerate(fractions, start=1)})
        row["max_abs_fraction_change"] = round(float(np.max(np.abs(fractions - ref_fractions))), 3)
        row["common_voxels"] = int(np.count_nonzero(common))
        row["same_id_in_common"] = round(float(np.mean(ref_labels[common] == moved_labels[common])), 3)
        row["ari_in_common"] = round(float(adjusted_rand_index(ref_labels, moved_labels, mask=common)), 3)
        rows.append(row)
robustness = pd.DataFrame(rows)
print(robustness.to_string(index=False))

# Cross-checks: grow_mm=0 reproduces the unperturbed labelling exactly,
# and it equals labelling the untouched patients directly.
direct = labeller.predict(new_patients).habitat_maps
print(
    "cross-check, grow 0 mm == unperturbed labelling:",
    all(
        np.array_equal(np.asarray(a.label_array), np.asarray(b.label_array))
        for a, b in zip(maps[0.0], direct)
    ),
)
# The volume stage columns agree with fractions from the label arrays.
table = labeller.predict(subjects_by_grow[2.0]).features.frame.set_index("subject")
print(
    "cross-check, +2 mm fractions == volume stage columns:",
    all(
        np.allclose(
            fractions_of(m, model.n_habitats),
            [
                float(table.loc[m.subject_id, f"habitat_{hid}_volume_fraction"])
                if f"habitat_{hid}_volume_fraction" in table
                else 0.0
                for hid in range(1, model.n_habitats + 1)
            ],
            atol=1e-6,
        )
        for m in maps[2.0]
    ),
)

# %%
# Where the ids change
# --------------------
# Original ROI (left) against the 2 mm eroded and the 2 mm dilated ROI
# (right) for the first perturbed patient. Grey-orange voxels in the
# disagreement panel changed habitat. Ids come from one model, so
# ``align_labels=False``: no relabelling is applied.
subject = new_patients[0]
for grow_mm, name in ((-2.0, "eroded"), (2.0, "dilated")):
    fig = plot_habitat_label_compare(
        subject.image("LAP"),
        maps[0.0][0],
        maps[grow_mm][0],
        titles=(f"{subject.subject_id}: original ROI", f"ROI {name} by {abs(grow_mm):.0f} mm"),
        align_labels=False,
        show_disagreement=True,
        crop_to="labels",
    )
    fig.savefig(f"out/segmentation_robustness_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Agreement against outline change
# --------------------------------
# One line per patient: share of common-ROI voxels that keep their
# habitat id, for each radius.
fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
for subject_id, group in robustness.groupby("subject"):
    ax.plot(group["grow_mm"], group["same_id_in_common"], marker="o", label=subject_id)
ax.set_xlabel("ROI change (mm, negative = eroded)")
ax.set_ylabel("Same habitat id in common ROI")
ax.set_ylim(0.0, 1.05)
ax.set_title("Habitat agreement under ROI erosion / dilation")
ax.legend(frameon=True)
fig.savefig("out/segmentation_robustness_agreement.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# How to read the result
# ----------------------
# Measured on this run (model from subj001 + subj002; ROI of subj003 and
# subj004 eroded / dilated by 1 and 2 mm; same saved model throughout):
#
# * ``grow_mm=0`` relabelling matched the unperturbed maps (cross-check
#   True): prediction is deterministic for a fixed outline.
# * At +/-1 mm, same-id agreement in the common ROI was about 0.79--0.91
#   (subj003) and 0.77--0.79 (subj004); ARI about 0.54--0.80.
# * At +/-2 mm, same-id agreement fell to about 0.68--0.83 and ARI to
#   about 0.36--0.73. Volume fractions of individual habitats shifted by
#   up to about 0.18 absolute.
# * Subject-level winsorize / min-max and the 30 supervoxels are
#   recomputed on the new ROI, so ids can change even inside voxels that
#   stay inside both outlines.
#
# Two demo patients are a mechanics check, not an ICC study of contour
# robustness. Report the radii you test with your own cohort.

# %%
# Where to go next
# ----------------
# * The approved analysis this page perturbs:
#   :doc:`/auto_examples/01_complete/plot_01_two_step_spec`.
# * Stability under a simulated rescan, and features chosen for it:
#   :doc:`/auto_examples/01_complete/plot_20_precise_features`.
# * QC flags for one-block and fragmented maps:
#   :doc:`/auto_examples/01_complete/plot_14_quality_control`.
# * Comparing maps from two separate fits (label matching):
#   :doc:`/auto_examples/01_complete/plot_19_label_matching`.
