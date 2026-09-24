"""
Defining habitats in two steps
==============================

Input: a cohort of at least two subjects. Output: one shared
:class:`~habit.contracts.HabitatModel` and one
:class:`~habit.contracts.HabitatMap` per subject. The design is the
stage list: ``partition`` then ``pool`` then ``fit``.

SLIC is the partitioner here. ``Spec("kmeans", ...)`` in the
``partition`` stage is the same study with a different partitioner, not
a second habitat definition to compare on this page. A fixed count of 3
draws the map; a 2-10 search draws the elbow on the same cohort.
``two_step_habitat(...)`` is a shortcut that builds the same stage list.
"""

# %%
# Load the cohort
# ---------------
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_cluster_validation_from_report, plot_partition_triptych

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {list(cohort.subject_ids)}")

# %%
# Fit three shared habitats
# -------------------------
# ``partition`` splits each ROI into supervoxels. Those supervoxels are
# the clustering units: one mean feature vector per supervoxel. ``pool``
# puts the units of all subjects together and ``fit`` learns one shared
# habitat model on them.
first_stages = (
    Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
    Stage("partition", Spec("slic", {"n_supervoxels": 100})),
    Stage("pool", Spec("pool")),
)
spec = HabitatSpec(
    name="two_step_slic",
    stages=first_stages + (
        Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(cohort)
print(result.habitat_model.summary())
print(result.features.frame)
result.features.frame

# %%
# Supervoxels, clustering units, and habitats
# -------------------------------------------
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
# Elbow on the same stages, with ``fit`` searching 2-10 habitats. The
# chosen count is the knee, not a second overlay. A saved model is
# :doc:`/auto_examples/04_habitat_maps/plot_04_apply_saved_model`.
elbow_spec = HabitatSpec(
    name="two_step_slic_elbow",
    stages=first_stages + (
        Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
    ),
    random_seed=0,
)
elbow = Study(elbow_spec).fit_predict(cohort)
report = (elbow.habitat_model.preprocessing_state or {}).get("selection_report")
print(elbow.habitat_model.summary())
fig_k = plot_cluster_validation_from_report(report)
fig_k.savefig("out/two_step_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
