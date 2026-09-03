"""
Graph features
===============

After a habitat map exists, :func:`~habit.kernels.extract_graph_features`
summarises region topology (lattice nodes, closest-voxel edges).
The same family is ``Spec("graph")`` on a study.

2-D network figures are display-only (one representative slice). Tables
use the full 3-D :class:`~habit.contracts.HabitatMap`.
"""

# sphinx_gallery_thumbnail_number = 3

# %%
# One-step habitats with a known K so the graph has a fixed number of
# labels. ``include_extended_metrics=False`` keeps the table narrow.
from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels import HabitatGraphFeatureOptions, extract_graph_features
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

options = HabitatGraphFeatureOptions(include_extended_metrics=False)
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
print("Study graph columns (head):")
print(result.features.frame.head())
result.features.frame.head()

# %%
# Kernel extract on the full 3-D label array (same options as the plot).
# Do not extract from a 2-D slice — the network figure is display-only.
rows = []
for subject, habitat_map in zip(cohort, result.habitat_maps):
    feats = extract_graph_features(
        habitat_map.label_array,
        options=options,
        expected_labels=habitat_map.habitat_ids,
    )
    rows.append({"subject_id": subject.subject_id, **feats})
table = pd.DataFrame(rows)
print(table.head())
table.head()

# %%
# Overlay, lattice slice, and 2-D network. ``block_size=8`` is the
# library default (same as :class:`~habit.kernels.HabitatGraphFeatureOptions`).
Path("out").mkdir(exist_ok=True)
labels = result.habitat_maps[0].label_array
fig = plot_habitat_overlay(
    cohort[0].image(MODALITIES[0]),
    result.habitat_maps[0],
    title="One-step habitats (K=3)",
)
fig.savefig("out/graph_habitat_slice_2d.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()

fig_slice = plot_habitat_graph_slice(
    labels, options=options, show_grid=True, block_size=8, grid_linestyle="--"
)
fig_slice.savefig("out/graph_habitat_lattice_2d.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()

fig_net = plot_habitat_graph_network_2d(
    labels,
    options=options,
    show_grid=True,
    block_size=8,
    grid_linestyle="--",
)
if fig_net is not None:
    fig_net.savefig("out/graph_habitat_network_2d.png", dpi=150, bbox_inches="tight")
    if os.environ.get("HABIT_NO_VIEW") != "1":
        plt.show()
