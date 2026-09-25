"""
Quickstart: Python API
======================

**Background.** Habitat analysis splits a tumour into sub-regions that
behave alike across the input images (here, DCE phases), then describes
each tumour by how much of each sub-region it has and how they are arranged.

**Purpose.** You get your first habitat maps, a per-subject feature table
(volume fractions, MSI, ITH, graph), and a saved model that labels a new
patient without refitting.

**Key terms.**

* **habitat** -- a sub-region inside the tumour (the ROI) whose voxels behave
  alike across the input images; HABIT paints each ROI voxel with a habitat
  id (1, 2, 3, ...).
* **ROI / mask** -- the region of interest, usually the whole tumour, stored
  as an integer mask; only voxels inside it are analysed (0 = background).
* **voxel feature** -- the numbers that describe one voxel (here its
  intensity in each DCE phase); one column per feature.
* **supervoxel** -- a small patch of neighbouring voxels with similar
  features, clustered inside one subject first (``partition``), so the cohort
  model clusters tens of rows per subject instead of every voxel.
* **pool** -- stacks every training subject's rows into one matrix so one
  model is fitted to the whole cohort and habitat ids mean the same thing in
  every patient.
* **fit** -- learns the habitat definition (for k-means: the number of
  habitats and their centroids).
* **assign** -- gives every supervoxel / voxel the id of its nearest
  centroid, producing the habitat map.
* **Stage / Spec / HabitatSpec** -- a ``Stage`` is one step with a label you
  choose; a ``Spec`` names a registered component and its parameters; a
  ``HabitatSpec`` is the ordered list of stages plus ``random_seed``, i.e.
  the whole study definition.
* **elbow** -- a rule for picking the number of habitats: the candidate
  count after which adding one more habitat stops reducing within-cluster
  spread much.

A short list of stages declares the analysis; one call fits it on a
cohort. Everything after that is looking at the result: the habitat map,
the per-habitat features a paper would report, and reusing the fitted
model on a new patient. No YAML.

Install first (:doc:`/tutorial/installation`). This page uses the
official demo pack (five liver lesions, four DCE phases). Change ``DATA``
/ ``MODALITIES`` / ``ROI`` to your own preprocessed tree.
"""

# %%
# Load the images
# ---------------
# Four subjects define the habitats; the fifth plays a new patient later.
# The downloaded pack is cached after the first run.
# sphinx_gallery_thumbnail_number = 4
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import HabitatModel, cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import habitat_ith_dispersion, ith_score, spatial_interaction_matrix
from habit.recipes import Study, two_step_habitat
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import (
    plot_cluster_validation_from_report,
    plot_habitat_graph_slice,
    plot_habitat_overlay,
    plot_habitat_volume_fractions,
    plot_intensity_slice,
    plot_ith_summary,
    plot_msi_matrix,
    plot_partition_triptych,
)

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train, new_patient = cohort[:4], cohort[4:]
print(cohort)

subject = train[0]
Path("out").mkdir(exist_ok=True)
fig = plot_intensity_slice(
    subject.image("LAP"),
    roi_mask=subject.mask(ROI),
    roi_contour=True,
    image_label="LAP",
    title=f"{subject.subject_id}: arterial phase and ROI",
)
fig.savefig("out/quickstart_input.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Build the analysis step by step
# -------------------------------
# A habitat analysis is a list of stages. This is the two-step design:
# each tumour is split into 30 supervoxels (``partition``), the
# supervoxels of all subjects are put together (``pool``) and clustered
# once (``fit``), so habitat 2 means the same enhancement pattern in every
# patient. Drop ``pool`` and every subject is clustered on its own
# (one-step); drop ``partition`` and voxels are clustered directly
# (direct pooling). The fitter tries 2 to 10 habitats and keeps the elbow.
spec = HabitatSpec(
    name="quickstart_two_step",
    stages=(
        # extract: one intensity column per DCE phase, inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # partition: 30 supervoxels per tumour; these rows are clustered.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        # pool: one matrix for every training subject, then one fit.
        Stage("pool", Spec("pool")),
        # fit: search 2..10 habitats and keep the elbow. n_init=10 restarts.
        Stage("fit", Spec("kmeans", {"min_habitats": 2, "max_habitats": 10, "validation": "elbow", "n_init": 10})),
        # assign: nearest centroid writes the shared habitat ids.
        Stage("assign", Spec("nearest_centroid")),
        # quantify: one row per subject. Spec name is the feature family.
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        Stage("graph", Spec("graph", {"include_extended_metrics": False})),
    ),
    # Seeds partition and fit. It is not a RunPolicy setting.
    random_seed=0,
)
# Study runs the spec: fit_predict learns the habitats on the four training
# subjects and labels those same subjects in one call.
result = Study(spec).fit_predict(train)
print(result.habitat_model.summary())

# %%
# ``two_step_habitat(modalities=..., roi=..., n_supervoxels=30,
# habitat_features=[...], random_seed=0)`` is a shortcut for the same
# spec; ``one_step_habitat`` and ``direct_pooling_habitat`` do the same for
# the other designs. Both give the same habitat maps:
shortcut = two_step_habitat(modalities=MODALITIES, roi=ROI, n_supervoxels=30, random_seed=0).fit_predict(train)
same = all(np.array_equal(a.label_array, b.label_array) for a, b in zip(result.habitat_maps, shortcut.habitat_maps))
print("stages == shortcut:", same)

# %%
# How many habitats, and why
# --------------------------
# The fitter scores every candidate count; the marked point is the one
# kept. The report travels inside the model, so the choice is auditable.
report = result.habitat_model.preprocessing_state["selection_report"]
fig = plot_cluster_validation_from_report(report)
fig.savefig("out/quickstart_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Habitat maps
# ------------
# From supervoxels to habitats on one slice, then the habitat map of two
# patients. Same colour, same habitat, in both.
habitat_map = result.habitat_maps[0]
fig = plot_partition_triptych(subject.image("LAP"), result.units[0], habitat_map, axis=0)
fig.savefig("out/quickstart_triptych.png", dpi=150, bbox_inches="tight")
plt.show()

for other in (0, 1):
    fig = plot_habitat_overlay(
        train[other].image("LAP"),
        result.habitat_maps[other],
        title=f"{train[other].subject_id}: habitats",
        crop_to="labels",
    )
    fig.savefig(f"out/quickstart_habitats_{train[other].subject_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# The feature table
# -----------------
# One row per subject, ready for statistics: volume fractions, the MSI
# (how habitats touch each other), the ITH score (how fragmented the
# tumour is) and graph topology. A few of the columns:
table = result.features.frame.set_index("subject")
print(table.shape[1], "columns")
columns = [c for c in table.columns if c.endswith("_volume_fraction")] + ["ith_score", "contrast", "graph_num_nodes_total"]
print(table[columns].round(3).to_string())

# %%
# What those numbers look like for one subject
# --------------------------------------------
# Volume fractions come straight from the table. MSI and ITH are drawn from
# the same label map with the kernels the table uses.
labels = habitat_map.label_array
fractions = {hid: float(table.loc[subject.subject_id, f"habitat_{hid}_volume_fraction"]) for hid in habitat_map.habitat_ids}
fig = plot_habitat_volume_fractions(fractions, title=f"{subject.subject_id}: volume fractions")
fig.savefig("out/quickstart_volume_fractions.png", dpi=150, bbox_inches="tight")
plt.show()

n_classes = max(habitat_map.habitat_ids) + 1
fig = plot_msi_matrix(spatial_interaction_matrix(labels, n_classes=n_classes), habitat_ids=habitat_map.habitat_ids)
fig.savefig("out/quickstart_msi.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_ith_summary(float(ith_score(labels)), dispersion=habitat_ith_dispersion(labels))
fig.savefig("out/quickstart_ith.png", dpi=150, bbox_inches="tight")
plt.show()

# The graph features cut the tumour into 8-voxel cubes (the grid); each
# cube is a node coloured by its habitat, and touching cubes share an edge.
fig = plot_habitat_graph_slice(labels, block_size=8)
fig.savefig("out/quickstart_graph.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Reuse the model on a new patient
# --------------------------------
# The ``.habitatmodel`` file is the habitat definition: load it anywhere
# and the new patient gets the same habitat names. Nothing is refitted.
# save writes the model archive and result tables under out/quickstart,
# plus the habitat maps because write_maps=True.
result.save("out/quickstart", write_maps=True)
model = HabitatModel.load("out/quickstart/habitat_model.habitatmodel")
# predict labels the held-out fifth subject with the saved centroids.
prediction = Study.from_model(model).predict(new_patient)
fig = plot_habitat_overlay(
    new_patient[0].image("LAP"),
    prediction.habitat_maps[0],
    title=f"{new_patient[0].subject_id}: habitats from the saved model",
    crop_to="labels",
)
fig.savefig("out/quickstart_new_patient.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Where to go next
# ----------------
# * The same analysis from a YAML file:
#   :doc:`/auto_quickstart/plot_quickstart_yaml`, and from the shell with
#   ``habit get-habitat --config config/habitat/config_habitat_quickstart_v1.yaml``
#   (:doc:`/tutorial/quickstart`). Both fit the same four subjects and give
#   the same habitat maps as this page.
# * The same stages, then each one opened up:
#   :doc:`/auto_examples/00_full_pipeline/plot_01_full_pipeline`.
# * Interactive 3-D view (``pip install "habitat-analysis[view]"``)::
#
#     from habit.viz import view_habitat_napari
#     view_habitat_napari(subject.image("LAP"), habitat_map)
