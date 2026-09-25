"""
Habitat radiomics with volume, MSI and ITH
==========================================

**Background.** Once every voxel has a habitat id, a patient can be
described in several complementary ways: how much of each habitat there
is (volume), how the habitats border each other (MSI), how fragmented
they are (ITH score), and what the image intensities look like inside
each habitat (per-habitat radiomics). A statistics tool needs all of
them as columns of one table, one row per patient.

**Purpose.** You will add two PyRadiomics families to the quantify
stages of the approved two-step analysis, label all five demo patients,
get one combined feature table, see which column family each column
belongs to, check one relation between the per-habitat and whole-ROI
columns, and write the table as a CSV file.

**When to use.** Whenever the habitat analysis feeds a downstream
statistical or modelling step that expects a flat patient-by-feature
table.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **volume** -- ``habitat_<id>_voxel_count`` and
  ``habitat_<id>_volume_fraction`` for every habitat id of the model.
* **MSI** (multiregional spatial interaction) -- counts of face-connected
  voxel pairs between habitat classes (``firstorder_<i>_and_<j>``, and
  the same counts normalised, ``firstorder_normalized_<i>_and_<j>``),
  plus second-order summaries of that matrix (``contrast``,
  ``homogeneity``, ``correlation``, ``energy``). Compare them only between
  patients labelled by the same cohort model.
* **ITH score** -- ``ith_score``, 0 when every habitat is one connected
  piece and closer to 1 the more fragmented the habitats are.
* **per-habitat radiomics** (``each_habitat``) -- PyRadiomics features of
  each original image inside each habitat, one column per
  habitat, feature and modality: ``habitat_<id>_<feature>_of_<modality>``,
  plus ``has_habitat_<id>`` (1 when the habitat is present). A habitat
  absent from a patient gives ``NaN``, not 0.
* **whole-ROI radiomics** (``traditional``) -- the same PyRadiomics
  features inside the whole ROI (all habitats together), named
  ``<feature>_of_<modality>``: the classic radiomics reference.
* **first-order features** -- statistics of the voxel intensities alone
  (mean, percentiles, energy, entropy, ...), no texture.

**Analysis definition.** The approved two-step definition (four DCE
phases, subject-level winsorize + min-max, 30 k-means supervoxels,
pool, 10-bin cohort binning, k-means 2..10 by elbow, nearest centroid).
One deviation: the ``graph`` stage is left out to keep the page fast;
it is covered in :doc:`/auto_examples/01_complete/plot_16_graph_network`.
PyRadiomics is limited to the first-order class to stay fast, and the
whole-ROI family to the ``LAP`` phase.

.. note::

   The habitat definition is fitted on two patients and applied to the
   other three of the 5-subject demo cohort. Five rows are enough to
   show the table, not to analyse it.

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.
"""

# %%
# Load the cohort
# ---------------
# The first two demo patients define the habitats; all five get a row.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train = cohort[:2]
Path("out").mkdir(exist_ok=True)
print(cohort)

# %%
# PyRadiomics settings
# --------------------
# A PyRadiomics parameter mapping, exactly as in a PyRadiomics YAML file.
# ``"firstorder": []`` enables every first-order feature (18 of them) and
# nothing else. ``binWidth`` only matters for the two bin-based
# first-order features (Entropy, Uniformity). The features are computed
# on the ORIGINAL images: the winsorize / min-max stages above only
# prepare the clustering features and do not change what PyRadiomics
# reads. MR intensities are in arbitrary units, so intensity features
# such as Mean or Energy are only comparable between patients whose
# images were intensity-harmonised before HABIT.
radiomics_params = {
    "imageType": {"Original": {}},
    "featureClass": {"firstorder": []},
    "setting": {"binWidth": 25},
}

# %%
# Declare the analysis with every quantify family
# -----------------------------------------------
# Each quantify stage adds its columns to the same one-row-per-patient
# table. The stage label (first argument) is free text; the ``Spec`` name
# picks the registered habitat feature extractor.
spec = HabitatSpec(
    name="habitat_radiomics_two_step",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # Subject-level: clip each column to the patient's own 5th..95th percentile.
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.05, 0.05]})),
        # Subject-level: rescale each column to 0..1 with the patient's own range.
        Stage("preprocess2", Spec("minmax")),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        Stage("pool", Spec("pool")),
        # Cohort-level: bin edges learned on the training supervoxels, stored in the model.
        Stage("preprocess_cohort", Spec("binning", {"n_bins": 10, "bin_strategy": "uniform"})),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10},
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        # Label-map families: they read only the habitat map.
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        # Intensity families: they read the original images as well.
        # Per habitat, all four phases.
        Stage(
            "habitat_radiomics",
            Spec("each_habitat", {"params": radiomics_params, "modalities": list(MODALITIES)}),
        ),
        # Whole ROI, one phase: the classic radiomics reference column set.
        Stage(
            "roi_radiomics",
            Spec("traditional", {"params": radiomics_params, "modalities": ["LAP"]}),
        ),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(train)
print(result.habitat_model.summary())

# The other three patients are labelled with the fitted definition (no
# refit); ``spec=`` runs the same quantify stages on them.
prediction = Study.from_model(result.habitat_model, spec=spec).predict(cohort[2:])
table = pd.concat([result.features.frame, prediction.features.frame], ignore_index=True)
habitat_ids = result.habitat_maps[0].habitat_ids
print("habitat ids:", habitat_ids)
print("table shape (patients x columns incl. 'subject'):", table.shape)

# %%
# Column families
# ---------------
# The prefix / suffix of a column name tells which family produced it.
# The helper below only sorts names; it computes nothing.
MSI_SECOND_ORDER = {"contrast", "homogeneity", "correlation", "energy"}


def column_family(column: str) -> str:
    """Return the feature family a combined-table column belongs to.

    Args:
        column: Column name of the combined feature table.

    Returns:
        str: Human-readable family name.
    """
    if column == "subject":
        return "id"
    if column.startswith("has_habitat_"):
        return "habitat presence (each_habitat)"
    if column.startswith("habitat_") and "_original_" in column:
        return "per-habitat radiomics (each_habitat)"
    if column.startswith("habitat_") and column.endswith(("_voxel_count", "_volume_fraction")):
        return "volume"
    if column.startswith("firstorder_") or column in MSI_SECOND_ORDER:
        return "MSI"
    if column == "ith_score":
        return "ITH"
    if column.startswith("original_"):
        return "whole-ROI radiomics (traditional)"
    return "other"


families = pd.Series({c: column_family(c) for c in table.columns})
summary = families.groupby(families).agg(["count", lambda s: ", ".join(list(s.index[:2]))])
summary.columns = ["n_columns", "example columns"]
print(summary.to_string())

# %%
# A first look at the label-map families
# --------------------------------------
# Volume fractions and ITH per patient. The MSI columns are in the table
# and exported, but their values are not interpreted on this page.
fraction_columns = [f"habitat_{hid}_volume_fraction" for hid in habitat_ids]
print(table.set_index("subject")[fraction_columns + ["ith_score"]].round(3).to_string())

# %%
# Per-habitat versus whole-ROI radiomics
# --------------------------------------
# The habitats split the ROI without overlap, so the whole-ROI mean LAP
# intensity must equal the voxel-count-weighted average of the
# per-habitat mean LAP intensities. This is a consistency check between
# three families of the same table (volume, each_habitat, traditional).
check_rows: List[Dict[str, object]] = []
for _, row in table.iterrows():
    counts = np.array([row[f"habitat_{hid}_voxel_count"] for hid in habitat_ids], dtype=float)
    means = np.array(
        [row[f"habitat_{hid}_original_firstorder_Mean_of_LAP"] for hid in habitat_ids],
        dtype=float,
    )
    present = counts > 0  # absent habitats have NaN means and zero weight
    weighted = float(np.sum(counts[present] * means[present]) / np.sum(counts[present]))
    roi_mean = float(row["original_firstorder_Mean_of_LAP"])
    check_rows.append(
        {
            "subject": row["subject"],
            "ROI mean LAP": roi_mean,
            "weighted habitat mean LAP": weighted,
            "match": bool(np.isclose(roi_mean, weighted, rtol=1e-6)),
        }
    )
check = pd.DataFrame(check_rows).set_index("subject")
print(check.round(3).to_string())
print("all patients consistent:", bool(check["match"].all()))

# %%
# Mean LAP intensity inside each habitat
# --------------------------------------
# One bar group per patient, one bar per habitat, from the
# ``habitat_<id>_original_firstorder_Mean_of_LAP`` columns.
fig, ax = plt.subplots(figsize=(7.5, 3.4))
width = 0.8 / len(habitat_ids)
x = np.arange(len(table))
for offset, hid in enumerate(habitat_ids):
    ax.bar(
        x + offset * width - 0.4 + width / 2,
        table[f"habitat_{hid}_original_firstorder_Mean_of_LAP"],
        width=width,
        label=f"habitat {hid}",
    )
ax.set_xticks(x)
ax.set_xticklabels(table["subject"])
ax.set_ylabel("Mean LAP intensity (a.u.)")
ax.set_title("Per-habitat first-order Mean, LAP phase")
ax.legend(frameon=False, ncol=len(habitat_ids), fontsize=8)
fig.tight_layout()
fig.savefig("out/habitat_radiomics_mean_lap.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Export for statistics
# ---------------------
# One CSV, one row per patient, the ``subject`` column first. Any
# statistics tool can read it. Reading it back checks that nothing was
# lost on the way.
csv_path = Path("out/habitat_radiomics_features.csv")
table.to_csv(csv_path, index=False)
reread = pd.read_csv(csv_path)
print(csv_path.name, reread.shape)
print("round trip identical:", bool(reread.shape == table.shape and np.allclose(
    reread.drop(columns="subject").to_numpy(dtype=float),
    table.drop(columns="subject").to_numpy(dtype=float),
    equal_nan=True,
)))

# %%
# How to read the result
# ----------------------
# Measured on this page (K = 4 by the elbow rule on subj001 + subj002;
# habitat ids 1..4; five patients):
#
# * The combined table has 5 rows and 352 columns including ``subject``:
#   8 volume, 4 habitat-presence, 288 per-habitat radiomics (4 habitats x
#   18 first-order features x 4 phases), 18 whole-ROI radiomics (18
#   features x LAP), 32 MSI and 1 ITH column.
# * ITH scores ranged from 0.84 (subj003) to 0.98 (subj005): in every
#   patient the habitats are split into many pieces rather than one
#   blob each.
# * The whole-ROI mean LAP intensity equalled the voxel-weighted average
#   of the per-habitat means in every patient: the families describe the
#   same voxels, split differently.
# * Every habitat was present in all five patients here, so the
#   per-habitat columns contain no ``NaN``. In your data an absent habitat
#   leaves ``NaN`` in its columns; decide explicitly how your statistics
#   handle it.
# * Several hundred columns for five patients is the usual radiomics
#   situation of many more features than patients: feature selection and
#   correction for multiple testing belong to the statistics step.

# %%
# Where to go next
# ----------------
# * Each family on its own:
#   :doc:`/auto_examples/02c_metrics/plot_01_volume_fractions`,
#   :doc:`/auto_examples/02c_metrics/plot_02_msi`,
#   :doc:`/auto_examples/02c_metrics/plot_03_ith`,
#   :doc:`/auto_examples/02c_metrics/plot_05_each_habitat_radiomics`, and
#   the habitat map itself as the image in
#   :doc:`/auto_examples/02c_metrics/plot_06_whole_habitat_radiomics`.
# * Graph network features of the same maps:
#   :doc:`/auto_examples/01_complete/plot_16_graph_network`.
# * Save maps, tables and the methods text:
#   :doc:`/auto_examples/01_complete/plot_23_export_methods`.
