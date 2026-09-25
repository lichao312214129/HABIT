"""
Comparing habitat maps with and without preprocessing
=====================================================

**Background.** Preprocessing changes the distances k-means sees, so it
can change the habitats. To see by how much, fit the same design twice,
once with and once without preprocessing, and compare the two maps.

**Purpose.** You get a table that pairs the habitat ids of the two fits
with a Dice score per pair, two overlays on shared slices, and volume
fractions per matched habitat for each subject on this demo.

**Key terms.**

* **preprocess** stage -- see
  :doc:`/auto_examples/02_stages/plot_04_feature_preprocessing`.
* **label switching** -- independent clusterings number the same
  habitat differently (habitat 1 in one fit can be habitat 3 in another),
  so ids must be matched before comparing.
* **Dice** -- overlap between two label maps (0 = none, 1 = identical).
* **volume fraction** -- the share of ROI voxels that belong to one
  habitat.

Input: the same cohort, the same two-step design, the same ``k`` and
seed. The only change is the voxel-feature preprocessor chain.

Each fit numbers its habitats in its own arbitrary k-means order, so
habitat 1 of one fit need not be habitat 1 of the other. **Match the ids
first, then compare.** Both fits label the same voxels of the same
subjects, so the ids are matched by voxel overlap (Hungarian on the
overlap counts). Centroid distance is not usable here: one model lives
in raw-intensity space, the other in winsorised min-max space.
Which matcher fits which situation: :doc:`/reference/habitat_matching`.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import habitat_volume_fractions
from habit.kernels.habitat_label_match import (
    habitat_dice_from_mapping,
    match_labels_by_overlap,
    remap_label_array,
)
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_overlay

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
Path("out").mkdir(exist_ok=True)


def two_step(preprocessors: tuple) -> HabitatSpec:
    """Build the same two-step spec with or without voxel preprocessing.

    ``preprocessors`` is a tuple of :class:`~habit.spec.Spec` objects
    inserted between feature extraction and the k-means partition.
    An empty tuple leaves the raw intensities unchanged.

    Args:
        preprocessors: Preprocessor specs, in order. Empty means no
            voxel-feature preprocessing.

    Returns:
        HabitatSpec: Two-step study (partition, pool, fit, assign).
    """
    stages = [
        Stage(
            "extract_voxel_features",
            Spec("raw", {"modalities": list(MODALITIES), "roi": ROI}),
        )
    ]
    for index, preprocessor in enumerate(preprocessors, start=1):
        stages.append(Stage(f"preprocess{index}", preprocessor))
    stages.extend(
        [
            Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 3})),
            Stage("pool", Spec("pool")),
            Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 3})),
            Stage("assign", Spec("nearest_centroid")),
        ]
    )
    return HabitatSpec(name="two_step", stages=tuple(stages), random_seed=0)


def _densest_slice(labels: np.ndarray, axis: int) -> int:
    """Return the index along ``axis`` with the most foreground voxels.

    Args:
        labels: Integer label volume. Voxels ``> 0`` count as foreground.
        axis: NumPy axis to scan (``0``, ``1``, or ``2``).

    Returns:
        int: Slice index in ``[0, labels.shape[axis])``.
    """
    other = tuple(i for i in range(labels.ndim) if i != axis)
    counts = np.sum(labels > 0, axis=other)
    return int(np.argmax(counts))


# %%
# Two fits on the same cohort: one on raw intensities, one after
# winsorising and min-max scaling every feature.
plain = Study(spec=two_step(())).fit_predict(cohort)
chain = (
    Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
    Spec("minmax", {"across_features": False}),
)
prepped = Study(spec=two_step(chain)).fit_predict(cohort)

# %%
# Step 1 -- match habitat ids (before any comparison)
# ----------------------------------------------------
# A two-step fit is one cohort model, so one renaming must hold for every
# subject. Stack the voxels of all subjects into one long vector per fit
# and pair ids by maximal overlap on that pooled vector. ``dice`` scores
# how well each matched pair agrees (1 = identical voxels).
plain_voxels = np.concatenate(
    [np.asarray(m.label_array).ravel() for m in plain.habitat_maps]
)
prepped_voxels = np.concatenate(
    [np.asarray(m.label_array).ravel() for m in prepped.habitat_maps]
)
# mapping pairs each preprocessed id with the unpreprocessed id it
# overlaps most (one-to-one, Hungarian assignment).
mapping = match_labels_by_overlap(plain_voxels, prepped_voxels)

match_table = pd.DataFrame(
    habitat_dice_from_mapping(plain_voxels, prepped_voxels, mapping),
    columns=["without_id", "with_id", "dice", "n_without", "n_with"],
)
print("id match (with preprocessing -> without), pooled over the cohort:")
print(match_table.to_string(index=False))
match_table

# %%
# Rename every preprocessed map into the unpreprocessed ids. From here on
# habitat ``H1`` means the same tissue in both fits.
ids = tuple(int(v) for v in plain.habitat_maps[0].habitat_ids)
aligned_maps = []
for moving in prepped.habitat_maps:
    renamed = remap_label_array(moving.label_array, mapping, reserved_ids=ids)
    present = tuple(sorted(int(v) for v in np.unique(renamed) if int(v) > 0))
    aligned_maps.append(replace(moving, label_array=renamed, habitat_ids=present))

# %%
# Step 2 -- compare the maps of one subject
# -----------------------------------------
# Both overlays share one crop (the union of the two label maps) and one
# axial / coronal / sagittal index, so they frame the same anatomy.
# Colours follow the matched ids.
image = cohort[0].image(ROI)
reference = plain.habitat_maps[0]
aligned = aligned_maps[0]
union = np.where(
    (np.asarray(reference.label_array) > 0)
    | (np.asarray(aligned.label_array) > 0),
    1,
    0,
).astype(np.int32)
shared_index = tuple(_densest_slice(union, axis) for axis in range(union.ndim))
print("shared slice indices (axis 0, 1, 2):", shared_index)

fig_plain = plot_habitat_overlay(
    image,
    reference,
    title="habitats without preprocessing",
    crop_to="labels",
    crop_labels=union,
    index=shared_index,
)
fig_plain.savefig(
    "out/habitats_without_preprocessing.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %%
# With preprocessing, same crop and the same slices.
fig_prepped = plot_habitat_overlay(
    image,
    aligned,
    title="habitats with preprocessing",
    crop_to="labels",
    crop_labels=union,
    index=shared_index,
)
fig_prepped.savefig(
    "out/habitats_with_preprocessing.png", dpi=150, bbox_inches="tight"
)
plt.show()

# %%
# Step 3 -- compare volume fractions on the matched ids
# -----------------------------------------------------
# One row per subject and habitat. Comparing fractions on the raw k-means
# ids (or on unmatched feature-table columns) would compare different
# habitats.
rows = []
for original, renamed in zip(plain.habitat_maps, aligned_maps):
    without = habitat_volume_fractions(original.label_array, ids)
    with_prep = habitat_volume_fractions(renamed.label_array, ids)
    for habitat_id in ids:
        rows.append(
            {
                "subject": original.subject_id,
                "habitat": f"H{habitat_id}",
                "without": without[habitat_id],
                "with": with_prep[habitat_id],
            }
        )
fraction_table = pd.DataFrame(rows)
print(fraction_table.to_string(index=False))
fraction_table

subjects = list(fraction_table["subject"].unique())
fig_bar, axes = plt.subplots(
    1, len(subjects), figsize=(3.4 * len(subjects), 3.2), sharey=True
)
axes = np.atleast_1d(axes)
x = np.arange(len(ids))
for ax, subject_id in zip(axes, subjects):
    part = fraction_table[fraction_table["subject"] == subject_id]
    ax.bar(x - 0.18, part["without"], width=0.36, label="without", color="#4C78A8")
    ax.bar(x + 0.18, part["with"], width=0.36, label="with", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(part["habitat"])
    ax.set_ylim(0, 1)
    ax.set_title(subject_id)
axes[0].set_ylabel("volume fraction")
axes[-1].legend()
fig_bar.suptitle("habitat volume fractions (matched ids)")
fig_bar.tight_layout()
fig_bar.savefig("out/preprocess_volume_fractions.png", dpi=150, bbox_inches="tight")
plt.show()
