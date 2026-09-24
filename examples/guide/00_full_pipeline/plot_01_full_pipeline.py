"""
A complete habitat analysis
===========================

One :class:`~habit.spec.HabitatSpec` declares the whole study. One
``fit_predict`` runs it. The stage list below is the same two-step
analysis as :doc:`/auto_quickstart/plot_quickstart_python`: raw DCE
intensities, 30 supervoxels, one shared model, then volume, MSI, ITH
and graph features.

Later Guide pages change one stage, or drop ``partition`` / ``pool``.
They are not a second analysis. Change ``DATA`` / ``MODALITIES`` /
``ROI`` to your preprocessed tree.
"""

# %%
# Load the cohort
# ---------------
# Two subjects define the habitats. The third is a new patient at the
# end of the page. The official demo pack is cached after the first
# download.
# sphinx_gallery_thumbnail_number = 3
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import HabitatModel, cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import habitat_ith_dispersion, ith_score, spatial_interaction_matrix
from habit.recipes import Study
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
train, new_patient = cohort[:2], cohort[2:3]
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
fig.savefig("out/full_pipeline_input.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Declare every stage
# -------------------
# ``Stage``'s first argument is a label you choose. ``Spec`` names a
# registered component and its parameters. ``random_seed`` seeds every
# stochastic stage (partition and fit), so a rerun paints the same map.
#
# ``two_step_habitat(...)`` builds this list for you. Drop ``pool`` and
# each subject is clustered alone (one-step). Drop ``partition`` and
# voxels are clustered directly (direct pooling). Those two designs are
# :doc:`/auto_examples/04_designs/plot_02_inside_each_subject` and
# :doc:`/auto_examples/04_designs/plot_03_pool_voxels`.
spec = HabitatSpec(
    name="full_pipeline_two_step",
    stages=(
        # extract: one intensity column per DCE phase, voxels inside the ROI.
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        # partition: 30 supervoxels per tumour. These rows, not voxels,
        # are what the shared model clusters.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 30})),
        # pool: stack every training subject's supervoxels into one matrix.
        Stage("pool", Spec("pool")),
        # fit: one k-means for the cohort. Try 2..10 habitats, keep the elbow.
        # n_init=10 restarts each candidate count.
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 10,
                    "validation": "elbow",
                    "n_init": 10,
                },
            ),
        ),
        # assign: nearest centroid paints a habitat id onto every supervoxel,
        # then onto the voxels that belong to it.
        Stage("assign", Spec("nearest_centroid")),
        # quantify: one row per subject. The Spec name is the feature family.
        Stage("volume", Spec("volume")),
        Stage("msi", Spec("msi")),
        Stage("ith", Spec("ith_score")),
        # graph: 8-voxel cubes, habitat-labelled nodes. Extended metrics off
        # so the table stays the short default set.
        Stage("graph", Spec("graph", {"include_extended_metrics": False})),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(train)
print(result.habitat_model.summary())

# %%
# How many habitats
# -----------------
# The fitter scores every candidate count. The marked point is the one
# kept, and the report is stored on the model.
report = result.habitat_model.preprocessing_state["selection_report"]
fig = plot_cluster_validation_from_report(report)
fig.savefig("out/full_pipeline_elbow.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Habitat maps
# ------------
# Supervoxels, then habitats, on one slice. The same colour is the same
# habitat in every patient because ``pool`` and ``fit`` ran once.
habitat_map = result.habitat_maps[0]
fig = plot_partition_triptych(subject.image("LAP"), result.units[0], habitat_map, axis=0)
fig.savefig("out/full_pipeline_triptych.png", dpi=150, bbox_inches="tight")
plt.show()

for one, one_map in zip(train, result.habitat_maps):
    fig = plot_habitat_overlay(
        one.image("LAP"),
        one_map,
        title=f"{one.subject_id}: habitats",
        crop_to="labels",
    )
    fig.savefig(f"out/full_pipeline_habitats_{one.subject_id}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Feature table
# -------------
# One row per subject: volume fractions, MSI, ITH, and graph columns
# produced by the quantify stages above.
table = result.features.frame.set_index("subject")
print(table.shape[1], "columns")
columns = [
    c
    for c in table.columns
    if c.endswith("_volume_fraction")
] + ["ith_score", "contrast", "graph_num_nodes_total"]
print(table[columns].round(3).to_string())

# %%
# One subject, four views of that table
# -------------------------------------
# Volume fractions are columns of the table. MSI, ITH and the graph
# slice are drawn from the same label map the table used.
labels = habitat_map.label_array
fractions = {
    hid: float(table.loc[subject.subject_id, f"habitat_{hid}_volume_fraction"])
    for hid in habitat_map.habitat_ids
}
fig = plot_habitat_volume_fractions(fractions, title=f"{subject.subject_id}: volume fractions")
fig.savefig("out/full_pipeline_volume_fractions.png", dpi=150, bbox_inches="tight")
plt.show()

n_classes = max(habitat_map.habitat_ids) + 1
fig = plot_msi_matrix(
    spatial_interaction_matrix(labels, n_classes=n_classes),
    habitat_ids=habitat_map.habitat_ids,
)
fig.savefig("out/full_pipeline_msi.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_ith_summary(float(ith_score(labels)), dispersion=habitat_ith_dispersion(labels))
fig.savefig("out/full_pipeline_ith.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_habitat_graph_slice(labels, block_size=8)
fig.savefig("out/full_pipeline_graph.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Save the model and label a new patient
# --------------------------------------
# The ``.habitatmodel`` file is the habitat definition. Loading it does
# not refit. The new patient gets the training centroids.
result.save("out/full_pipeline", write_maps=True)
model = HabitatModel.load("out/full_pipeline/habitat_model.habitatmodel")
prediction = Study.from_model(model).predict(new_patient)
fig = plot_habitat_overlay(
    new_patient[0].image("LAP"),
    prediction.habitat_maps[0],
    title=f"{new_patient[0].subject_id}: habitats from the saved model",
    crop_to="labels",
)
fig.savefig("out/full_pipeline_new_patient.png", dpi=150, bbox_inches="tight")
plt.show()
