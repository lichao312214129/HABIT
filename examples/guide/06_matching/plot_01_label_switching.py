"""
Why habitat ids must be matched
===============================

Cluster the same tumour twice with the same features and the same ``k``,
changing only the k-means random seed. The two maps describe the same
tissue, but k-means numbers its clusters in arbitrary order, so habitat 1
of one run is habitat 2 or 3 of the other. This is **label switching**.
Comparing the maps by raw id reports disagreement that is not there.

Because both maps label the same voxels, the ids are matched by voxel
overlap: count the voxels shared by every pair of habitats, then pick the
one-to-one pairing with the largest total (Hungarian assignment).
"""

# %%
# Two k-means runs on one subject
# -------------------------------
# Relative enhancement in the arterial (LAP) and portal-venous (PVP)
# phases, ``k = 3``, seeds 0 and 1.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.precision import align_habitat_map, habitat_stability
from habit.viz import plot_habitat_label_compare, plot_label_overlap_matrix
from habit.voxel_features import ExpressionVoxelFeatures

# Change DATA / MODALITIES / ROI and the expressions to your own layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
subject = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[0]
extractor = ExpressionVoxelFeatures(
    features={
        "rel_enh_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
        "rel_enh_pvp": "(PVP - pre_contrast) / (pre_contrast + eps)",
    },
    roi=ROI,
)
units = voxel_units(extractor(subject))

runs = []
for seed in (0, 1):
    fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=1)
    fitter.set_random_state(seed)
    model = fitter.fit([units], cohort=Cohort([subject], name=subject.subject_id))
    runs.append((model, model.assigner()(units)))
    print(f"seed {seed} centroids:\n{np.asarray(model.centroids).round(3)}")
(model_a, map_a), (model_b, map_b) = runs

# %%
# Same centroids, different order: row 1 of seed 0 is row 3 of seed 1.
# Raw ids therefore disagree on most voxels.
labels_a = np.asarray(map_a.label_array)
labels_b = np.asarray(map_b.label_array)
roi = labels_a > 0
print(f"voxels with the same raw id: {np.mean(labels_a[roi] == labels_b[roi]):.1%}")

# %%
# Raw ids side by side
# --------------------
# ``align_labels=False`` draws both maps as they came out of k-means: the
# colours differ although the regions are the same.
Path("out").mkdir(exist_ok=True)
image = subject.image(ROI)
fig = plot_habitat_label_compare(
    image,
    map_a,
    map_b,
    titles=("seed 0", "seed 1 (raw ids)"),
    align_labels=False,
    crop_to="labels",
)
fig.savefig("out/switching_raw_ids.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# The overlap table decides the pairing
# -------------------------------------
# Each cell counts voxels in moving habitat ``i`` (seed 1) and reference
# habitat ``j`` (seed 0). The outlined cells are the Hungarian pairs:
# every habitat gets exactly one partner and the outlined total is the
# largest possible.
fig = plot_label_overlap_matrix(map_a, map_b, reference_name="seed 0", moving_name="seed 1")
fig.savefig("out/switching_overlap_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Rename seed 1 into the seed-0 ids
# ---------------------------------
# Both runs share a ``model_id`` (it digests the spec and subject, not the
# random seed), so :func:`~habit.precision.align_habitat_map` would treat
# them as one definition and do nothing. ``force=True`` aligns anyway.
print(f"same model_id: {model_a.model_id == model_b.model_id}")
aligned_b = align_habitat_map(map_a, map_b, force=True)
same = np.mean(labels_a[roi] == np.asarray(aligned_b.label_array)[roi])
print(f"voxels with the same id after matching: {same:.1%}")

fig = plot_habitat_label_compare(
    image,
    map_a,
    aligned_b,
    titles=("seed 0", "seed 1 (matched ids)"),
    align_labels=False,
    crop_to="labels",
)
fig.savefig("out/switching_matched_ids.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Dice per habitat
# ----------------
# :func:`~habit.precision.habitat_stability` pairs the ids by the same
# overlap rule and scores each pair. Pass the *original* moving map; it
# matches internally.
dice = habitat_stability(map_a, [map_b])
print(dice.to_string(index=False))
dice
