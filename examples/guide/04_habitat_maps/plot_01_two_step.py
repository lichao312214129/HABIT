"""
Defining habitats in two steps
==============================

Input: a cohort of at least two subjects. Output: one shared
:class:`~habit.contracts.HabitatModel` and one
:class:`~habit.contracts.HabitatMap` per subject. Stages: ``partition``
then ``pool`` then ``fit``.

SLIC is the partitioner here. ``supervoxel_algorithm="kmeans"`` is the
same study with a different partitioner, not a second habitat definition
to compare on this page. Fixed ``n_habitats=3`` draws the map. ``"auto"``
draws the elbow on the same cohort.
"""

# %%
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import two_step_habitat
from habit.viz import plot_cluster_validation_from_report, plot_partition_triptych

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {list(cohort.subject_ids)}")

# two_step_habitat partitions each ROI into supervoxels. Those
# supervoxels are the clustering units: one mean feature vector per
# supervoxel. The units are then pooled across subjects and one shared
# habitat model is fit.
result = two_step_habitat(
    modalities=MODALITIES,
    n_supervoxels=100,
    n_habitats=3,
    habitat_features=("volume",),
    random_seed=0,
    supervoxel_algorithm="slic",
    roi=ROI,
).fit_predict(cohort)
print(result.habitat_model.summary())
print(result.features.frame)
result.features.frame

fig = plot_partition_triptych(
    cohort[0].image(ROI),
    result.units[0],
    result.habitat_maps[0],
    axis=0,
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/two_step_triptych.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Elbow on the same two-step declaration. The chosen count is the knee,
# not a second overlay. A saved model is
# :doc:`/auto_examples/04_habitat_maps/plot_04_apply_saved_model`.
elbow = two_step_habitat(
    modalities=MODALITIES,
    n_supervoxels=100,
    n_habitats="auto",
    random_seed=0,
    supervoxel_algorithm="slic",
    roi=ROI,
).fit_predict(cohort)
report = (elbow.habitat_model.preprocessing_state or {}).get("selection_report")
print(elbow.habitat_model.summary())
fig_k = plot_cluster_validation_from_report(report)
fig_k.savefig("out/two_step_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
