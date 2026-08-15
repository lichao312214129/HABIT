#!/usr/bin/env python
"""
Three subjects → one-step habitats (K=4) → graph features + plots.

This gallery uses the library graph defaults: an 8-voxel cube lattice
(``node_method='uniform_grid'``) with one node per in-cell subregion
centroid, and closest-voxel edges (``edge_method='min_distance'``,
``distance_threshold=5``). It only fixes ``n_habitats=4``.

Heatmaps use the **same representative axial slice** as the 2D network
(not a full-volume extract). ``include_extended_metrics=False`` keeps
the gallery interactive. For a 3D cohort table, load
``habitat_graph_features.csv`` from ``habit extract`` and pass
``subjects=`` / ``features=`` to the same plot function.

Accompanies ``docs/source/examples/graph_features.rst``.
Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import pandas as pd

from habit import (
    HabitatGraphFeatureOptions,
    cohort_from_directory,
    extract_graph_features,
    one_step_habitat,
)

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"
# Visualization: which people to show. Add ids (and load more rows) for all 5.
SUBJECTS = ("subj001", "subj002", "subj003")

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:3]
# Fixed K=4 (not "auto") so the graph has a known number of habitats.
# One fit for overlay, 2D network, and the heatmap table.
# Extended metrics OFF: they dominate runtime even at block_size=8.
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=4, random_seed=0, roi=ROI
).fit_predict(cohort)

options = HabitatGraphFeatureOptions(include_extended_metrics=False)
# Same representative axial slice as plot_habitat_graph_network_2d.
rows = []
for subject, habitat_map in zip(cohort, result.habitat_maps):
    labels_3d = habitat_map.label_array
    slice_index = int(
        (labels_3d > 0).reshape(labels_3d.shape[0], -1).sum(axis=1).argmax()
    )
    feats = extract_graph_features(
        labels_3d[slice_index],
        options=options,
        expected_labels=habitat_map.habitat_ids,
    )
    rows.append({"subject_id": subject.subject_id, **feats})
table = pd.DataFrame(rows)
print(
    table.shape[0],
    "subjects x",
    table.shape[1] - 1,
    "graph features from representative slices",
)
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort, result, table, options, SUBJECTS, MODALITIES.
from habit.viz import (
    plot_graph_feature_heatmap,
    plot_habitat_graph_network_2d,
    plot_habitat_overlay,
)

Path("out").mkdir(exist_ok=True)
# Overlay + 2D network: first of the three subjects.
labels = result.habitat_maps[0].label_array
fig = plot_habitat_overlay(
    cohort[0].image(MODALITIES[0]),
    result.habitat_maps[0],
    title="One-step habitats (K=4)",
)
fig.savefig("out/graph_habitat_slice_2d.png", dpi=150, bbox_inches="tight")
# Same edge options as extract_graph_features so the 2D plot matches.
# block_size=8 is the library default (8-voxel cubes on the legend).
fig = plot_habitat_graph_network_2d(
    labels,
    options=options,
    show_grid=True,
    block_size=8,
    grid_linestyle="--",
)
fig.savefig("out/graph_habitat_network_2d.png", dpi=150, bbox_inches="tight")
# Cohort heatmaps: set people and features here (do not dump all columns).
fig = plot_graph_feature_heatmap(
    table,
    subjects=SUBJECTS,
    n_features=40,
    feature_group="single",
    select="variance",
    title="Single-habitat graph features (column z-score)",
)
fig.savefig("out/graph_feature_heatmap_single.png", dpi=150, bbox_inches="tight")
fig = plot_graph_feature_heatmap(
    table,
    subjects=SUBJECTS,
    n_features=40,
    feature_group="pair",
    select="variance",
    title="Pairwise graph features (column z-score)",
)
fig.savefig("out/graph_feature_heatmap_pair.png", dpi=150, bbox_inches="tight")
print(
    "Wrote out/graph_habitat_slice_2d.png, "
    "out/graph_habitat_network_2d.png, "
    "out/graph_feature_heatmap_single.png, and "
    "out/graph_feature_heatmap_pair.png"
)
# END figures

if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    # Gallery = copy of out/ from the visible block (same composition).
    copy_out_figures_to_gallery(
        (
            "graph_habitat_slice_2d.png",
            "graph_habitat_network_2d.png",
            "graph_feature_heatmap_single.png",
            "graph_feature_heatmap_pair.png",
        )
    )

    # Optional 3D renders need [view] / PyVista; keep them out of the recipe.
    try:
        from habit.viz import (
            render_habitat_graph_network_3d,
            render_habitat_graph_surface_3d,
        )

        gallery = Path("docs/source/_static/images/examples")
        gallery.mkdir(parents=True, exist_ok=True)
        vol = cohort[0].image(MODALITIES[0])
        spacing_zyx = (1.0, 1.0, 1.0)
        if vol.spacing is not None and len(vol.spacing) == 3:
            sx, sy, sz = (float(v) for v in vol.spacing)
            spacing_zyx = (sz, sy, sx)
        surface = render_habitat_graph_surface_3d(
            labels, spacing=spacing_zyx, black_background=False, render_window=1200
        )
        network = render_habitat_graph_network_3d(
            labels,
            options=options,
            spacing=spacing_zyx,
            black_background=False,
            render_window=1200,
        )
        if surface is not None:
            plt.imsave(gallery / "graph_habitat_surface_3d.png", surface)
        if network is not None:
            plt.imsave(gallery / "graph_habitat_network_3d.png", network)
    except Exception as exc:  # pragma: no cover
        print(f"3D gallery assets skipped: {exc}")
