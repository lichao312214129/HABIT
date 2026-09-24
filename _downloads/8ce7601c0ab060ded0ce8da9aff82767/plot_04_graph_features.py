"""
Graph features
===============

After a habitat map exists, :func:`~habit.kernels.extract_graph_features`
summarises region topology (lattice nodes, closest-voxel edges).
The same family is available as the scikit-learn-style component
:class:`~habit.habitat_features.GraphHabitatFeatures`, or as
``Spec("graph")`` on a study.

**Cross-tumour id alignment.** With ``one_step`` clustering each subject
is clustered independently, so integer habitat ids are permuted across
patients: cluster 1 in subject A need not be the same phenotype as
cluster 1 in subject B. Before extracting subject-level features that
name habitats (especially graph columns ``single_h*``, ``pair_h*_*``),
name every subject against shared prototypes with
:func:`~habit.precision.align_habitat_maps_to_prototypes`. Only then does
``single_h1`` mean the same habitat across the cohort. Method and
caveats: :doc:`/auto_examples/06_matching/plot_07_match_labels` and
:doc:`/reference/habitat_matching`.

2-D network figures are display-only (one representative slice). Tables
use the full 3-D :class:`~habit.contracts.HabitatMap`.
"""

# %%
# One-step habitats with a known K so the graph has a fixed number of
# labels. Graph option fields are passed as flat kwargs — no separate
# options object is required.
# sphinx_gallery_thumbnail_number = 3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_features import GraphHabitatFeatures
from habit.kernels import extract_graph_features
from habit.precision import align_habitat_maps_to_prototypes
from habit.recipes import one_step_habitat
from habit.spec import Spec
from habit.viz import (
    plot_habitat_graph_network_2d,
    plot_habitat_graph_slice,
    plot_habitat_overlay,
)

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {list(cohort.subject_ids)}")

result = one_step_habitat(
    modalities=MODALITIES,
    n_habitats=3,
    random_seed=0,
    roi=ROI,
    habitat_features=[
        "volume",
        Spec("graph", {"include_extended_metrics": False}),
    ],
).fit_predict(cohort)
print("Study graph columns before cross-tumour alignment (head):")
print(result.features.frame.head())
result.features.frame.head()

# %%
# Name every subject's habitats against shared prototypes so habitat
# integers mean one phenotype across the cohort. ``models=`` reads each
# subject's fitted clustering centroids. This demo clusters raw LAP to
# stay short; in a real study cluster on features that are comparable
# across patients (see the matching page).
matched = align_habitat_maps_to_prototypes(
    list(result.habitat_maps),
    models=[result.subject_models[subject.subject_id] for subject in cohort],
)
aligned_maps = list(matched.habitat_maps)
print(matched.assignments)

# %%
# Two idiomatic extraction paths on the **aligned** full 3-D label arrays.
# Do not extract from a 2-D slice — the network figure is display-only.
#
# 1) Direct kernel function with flat kwargs (sklearn-style keyword API):
rows = []
for subject, habitat_map in zip(cohort, aligned_maps):
    feats = extract_graph_features(
        habitat_map.label_array,
        expected_labels=habitat_map.habitat_ids,
        block_size=8,
        include_extended_metrics=False,
    )
    rows.append({"subject_id": subject.subject_id, **feats})
table = pd.DataFrame(rows)
print("Kernel extract_graph_features (flat kwargs) after alignment:")
print(table.head())
table.head()

# %%
# 2) Scikit-learn style component — construct once, call per subject:
graph_extractor = GraphHabitatFeatures(
    block_size=8,
    include_extended_metrics=False,
)
component_rows = []
for subject, habitat_map in zip(cohort, aligned_maps):
    feature_table = graph_extractor(subject, habitat_map)
    component_rows.append(feature_table.frame)
component_table = pd.concat(component_rows, ignore_index=True)
print("GraphHabitatFeatures component (same options as constructor):")
print(component_table.head())
component_table.head()

# %%
# Overlay, lattice slice, and 2-D network — also flat kwargs, no options object.
# ``block_size=8`` is the library default for both extraction and display.
Path("out").mkdir(exist_ok=True)
labels = aligned_maps[0].label_array
fig = plot_habitat_overlay(
    cohort[0].image(MODALITIES[0]),
    aligned_maps[0],
    title="One-step habitats (K=3, prototype ids)",
)
fig.savefig("out/graph_habitat_slice_2d.png", dpi=150, bbox_inches="tight")
plt.show()

fig_slice = plot_habitat_graph_slice(
    labels,
    block_size=8,
    show_grid=True,
    grid_linestyle="--",
)
fig_slice.savefig("out/graph_habitat_lattice_2d.png", dpi=150, bbox_inches="tight")
plt.show()

fig_net = plot_habitat_graph_network_2d(
    labels,
    block_size=8,
    show_grid=True,
    grid_linestyle="--",
)
if fig_net is not None:
    fig_net.savefig("out/graph_habitat_network_2d.png", dpi=150, bbox_inches="tight")
    plt.show()
