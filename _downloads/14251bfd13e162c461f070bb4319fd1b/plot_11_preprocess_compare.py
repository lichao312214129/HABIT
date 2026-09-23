"""
Comparing habitat maps with and without preprocessing
=====================================================

Input: the same cohort, the same two-step design, the same ``k`` and
seed. The only change is the voxel-feature preprocessor chain.

The two overlays share one crop and one set of slice indices, taken from
the union of both maps. Habitat integers are matched by voxel overlap
before the volume-fraction bars: k-means may permute ids, and unpaired
ids are not the same habitat.
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
image = cohort[0].image(ROI)
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
# Two fits on the same cohort. Overlap matching renames the preprocessed
# map into the raw map's ids (same subject, same grid). The shared window
# is the union of both label maps: one crop, and one axial / coronal /
# sagittal index in the original volume. The next two figures both use
# that window.
plain = Study(spec=two_step(())).fit_predict(cohort)
chain = (
    Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
    Spec("minmax", {"across_features": False}),
)
prepped = Study(spec=two_step(chain)).fit_predict(cohort)

reference = plain.habitat_maps[0]
moving = prepped.habitat_maps[0]
mapping = match_labels_by_overlap(reference.label_array, moving.label_array)
aligned_labels = remap_label_array(
    moving.label_array,
    mapping,
    reserved_ids=reference.habitat_ids,
)
present = tuple(sorted(int(v) for v in np.unique(aligned_labels) if int(v) > 0))
aligned = replace(moving, label_array=aligned_labels, habitat_ids=present)

union = np.where(
    (np.asarray(reference.label_array) > 0) | (aligned_labels > 0),
    1,
    0,
).astype(np.int32)
shared_index = tuple(_densest_slice(union, axis) for axis in range(union.ndim))

match_table = pd.DataFrame(
    [
        {"with_id": int(src), "without_id": int(dst)}
        for src, dst in sorted(mapping.items())
    ]
)
print("overlap match (with preprocessing -> without):")
print(match_table.to_string(index=False))
print("shared slice indices (axis 0, 1, 2):", shared_index)
match_table

# %%
# Without preprocessing. ``crop_labels`` and ``index`` are the shared
# window, so this figure frames the same anatomy as the next one.
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
# With preprocessing, same crop and the same slices. Colours follow the
# matched ids, so habitat 1 is the same colour in both figures.
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
# Volume fractions on the matched ids. The raw feature-table columns are
# not comparable: they still use each fit's own k-means integers.
ids = tuple(int(v) for v in reference.habitat_ids)
fractions_plain = habitat_volume_fractions(reference.label_array, ids)
fractions_prepped = habitat_volume_fractions(aligned_labels, ids)
fraction_table = pd.DataFrame(
    {
        "habitat": [f"H{habitat_id}" for habitat_id in ids],
        "without": [fractions_plain[habitat_id] for habitat_id in ids],
        "with": [fractions_prepped[habitat_id] for habitat_id in ids],
    }
)
print(fraction_table.to_string(index=False))
fraction_table

labels = [f"H{habitat_id}" for habitat_id in ids]
fig_bar, ax = plt.subplots(figsize=(6.2, 3.2))
x = np.arange(len(labels))
ax.bar(
    x - 0.18,
    [fractions_plain[habitat_id] for habitat_id in ids],
    width=0.36,
    label="without",
    color="#4C78A8",
)
ax.bar(
    x + 0.18,
    [fractions_prepped[habitat_id] for habitat_id in ids],
    width=0.36,
    label="with",
    color="#F58518",
)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("volume fraction")
ax.set_ylim(0, 1)
ax.set_title("habitat volume fractions")
ax.legend()
fig_bar.savefig("out/preprocess_volume_fractions.png", dpi=150, bbox_inches="tight")
plt.show()
