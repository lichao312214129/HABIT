"""
Match habitat labels
=====================

Independent fits permute integer ids: habitat 1 of subject B need not
be habitat 1 of subject A. :mod:`habit.kernels.habitat_label_match`
recovers a ``{moving_id: reference_id}`` map.

This is **not** :class:`~habit.contracts.HabitatModel` apply. A shared
cohort model (the Apply page) already uses one id space. Matching is
for two independent clusterings that must be named after the fact.

* :func:`~habit.kernels.habitat_label_match.match_labels_by_features` —
  cross-patient (or two seeds) using unscaled **habitat summary
  features** (means / medians of any shared voxel field: raw
  multimodality, constructed maps, or texture channels).
* :func:`~habit.kernels.habitat_label_match.match_labels_by_overlap` —
  same tumour, two masks on one grid (two observers).
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Two independent one-step fits on the same subject (different seeds).
# Same grid, so we can overlay before/after alignment. A second subject
# would use the same feature matcher; overlap matching would not apply
# across patients.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels.habitat_label_match import (
    habitat_intensity_centroids,
    match_labels_by_features,
    match_labels_by_overlap,
    remap_label_array,
)
from habit.recipes import one_step_habitat
from habit.viz import plot_habitat_label_compare, plot_habitat_overlay

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]
print(f"Subjects: {list(cohort.subject_ids)}; matching demo uses {subject.subject_id}")

result_a = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort[:1])
result_b = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=1, roi=ROI
).fit_predict(cohort[:1])
map_a = result_a.habitat_maps[0]
map_b = result_b.habitat_maps[0]
image = subject.image(ROI)

# %%
# Feature matcher: unscaled habitat summary means (here: LAP intensity),
# then Hungarian assignment after a column z-score. Multimodal / texture
# fields work the same way — pass a volume with a trailing feature axis
# into :func:`~habit.kernels.habitat_label_match.habitat_intensity_centroids`,
# or build your own ``(n_habitats, n_features)`` matrix. Do not pass
# per-tumour MinMax centres — those axes are not comparable across fits.
ids_a, feat_a = habitat_intensity_centroids(image.data, map_a.label_array)
ids_b, feat_b = habitat_intensity_centroids(image.data, map_b.label_array)
mapping = match_labels_by_features(
    ids_a,
    feat_a,
    ids_b,
    feat_b,
    metric="euclidean",
    standardize="zscore",
)
mapping_table = pd.DataFrame(
    [{"moving_id": int(src), "reference_id": int(dst)} for src, dst in mapping.items()]
)
print("match_labels_by_features (seed=1 -> seed=0):")
print(mapping_table.to_string(index=False))
mapping_table

aligned_b = remap_label_array(
    map_b.label_array, mapping, reserved_ids=ids_a.tolist()
)

# %%
# Before: raw integer ids (``align_labels=False``). After: remapped B.
# Independent one-step maps can share a model_id digest; force no extra
# overlap alignment so the figure shows *this* mapping only.
Path("out").mkdir(exist_ok=True)
fig_before = plot_habitat_label_compare(
    image,
    map_a.label_array,
    map_b.label_array,
    titles=("Fit A (seed=0)", "Fit B (seed=1)"),
    align_labels=False,
)
fig_before.savefig("out/match_labels_before.png", dpi=150, bbox_inches="tight")
plt.show()

fig_after = plot_habitat_label_compare(
    image,
    map_a.label_array,
    aligned_b,
    titles=("Fit A", "Fit B remapped"),
    align_labels=False,
)
fig_after.savefig("out/match_labels_after.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Same-tumour two observers: permute ids on one map and recover them
# with overlap (Hungarian on voxel counts). Feature matching is the
# cross-patient operator; overlap cannot name habitats across grids.
rng = np.random.default_rng(3)
present = [int(v) for v in ids_a]
shuffled = list(present)
rng.shuffle(shuffled)
observer_map = {src: int(dst) for src, dst in zip(present, shuffled)}
observer_labels = remap_label_array(
    map_a.label_array, observer_map, reserved_ids=present
)
recovered = match_labels_by_overlap(map_a.label_array, observer_labels)
overlap_table = pd.DataFrame(
    [
        {"moving_id": int(src), "reference_id": int(dst)}
        for src, dst in recovered.items()
    ]
)
print("match_labels_by_overlap (permuted observer -> original):")
print(overlap_table.to_string(index=False))
overlap_table

fig_ref = plot_habitat_overlay(image, map_a, title="reference (fit A)")
fig_ref.savefig("out/match_labels_reference.png", dpi=150, bbox_inches="tight")
plt.show()
