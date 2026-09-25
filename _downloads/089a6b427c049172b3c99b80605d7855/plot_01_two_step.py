"""
Defining habitats in two steps
==============================

**Background.** The two-step design is the complete analysis used across
this Guide: each tumour is
first cut into supervoxels, then the supervoxels of all subjects are
clustered together once, so habitat ids mean the same thing in every
patient.

**Purpose.** You get one shared habitat model, a habitat map and a
volume-fraction table per subject, a supervoxel / habitat triptych, and an
elbow plot for choosing the habitat count.

**When to use.** You want habitats that are comparable across patients
(e.g. volume fractions as cohort features) and whole-cohort voxel
clustering would be slow or noisy.

**Key terms.**

* **supervoxel** -- a small patch of neighbouring voxels with similar
  features, clustered inside one subject first (``partition``); see
  :doc:`/auto_examples/02_stages/plot_12_supervoxels`.
* **pool** -- stacks every training subject's rows into one matrix so one
  model is fitted to the whole cohort.
* **SLIC** -- a supervoxel method that grows compact regions from a regular
  grid of seeds, trading intensity similarity against spatial distance.
* **elbow** -- see :doc:`/auto_quickstart/plot_quickstart_python`.

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
    # extract: one intensity column per DCE phase, voxels inside the ROI.
    Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
    # partition: SLIC supervoxels. These rows, not voxels, are clustered.
    Stage("partition", Spec("slic", {"n_supervoxels": 100})),
    # pool: stack every subject's supervoxels before a single fit.
    Stage("pool", Spec("pool")),
)
spec = HabitatSpec(
    name="two_step_slic",
    stages=first_stages + (
        # fit: one shared k-means. n_init=10 restarts; count is fixed at 3.
        Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
        # assign: nearest centroid paints habitat ids back onto voxels.
        Stage("assign", Spec("nearest_centroid")),
        # quantify: per-subject volume fractions of those ids.
        Stage("volume", Spec("volume")),
    ),
    # Seeds the stochastic stages (here the k-means fit) so a rerun paints the same map.
    random_seed=0,
)
# fit_predict: learn the shared centroids on the cohort, then label every
# subject with them in the same call.
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
# :doc:`/auto_examples/05_apply/plot_04_apply_saved_model`.
elbow_spec = HabitatSpec(
    name="two_step_slic_elbow",
    stages=first_stages + (
        Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
    ),
    random_seed=0,
)
elbow = Study(elbow_spec).fit_predict(cohort)
# The fitter stores its per-count scores inside the model, so the chosen
# count can be audited later.
report = (elbow.habitat_model.preprocessing_state or {}).get("selection_report")
print(elbow.habitat_model.summary())
fig_k = plot_cluster_validation_from_report(report)
fig_k.savefig("out/two_step_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
