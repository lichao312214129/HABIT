#!/usr/bin/env python
"""
Three subjects → one-step habitats (K=4) → graph features + plots.

This gallery uses the library graph defaults: an 8-voxel cube lattice
(``node_method='uniform_grid'``) with one node per in-cell subregion
centroid, and closest-voxel edges (``edge_method='min_distance'``,
``distance_threshold=5``). It only fixes ``n_habitats=4``.

Tables, heatmaps, and the 5-minus-8 delta use **full 3D** habitat maps
(``HabitatMap.label_array``). The 2D network figures are display-only:
they draw a representative axial slice so you can see the lattice.
``include_extended_metrics=False`` keeps the 3D extract tractable.
A second 3D extract uses ``block_size=5`` only as a comparison override
(library default stays 8). For a larger cohort table, load
``habitat_graph_features.csv`` from ``habit extract`` and pass
``subjects=`` / ``features=`` to the same plot function.

Accompanies ``docs/source/examples/graph_features.rst``.
Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

# BEGIN example
import time
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
# One fit for overlay, 2D display networks, and both 3D heatmap tables.
# Extended metrics OFF: they dominate runtime on a full 3D map.
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=4, random_seed=0, roi=ROI
).fit_predict(cohort)

# Library default stays block_size=8. The 5-voxel options are a
# comparison override only (same min_distance / threshold / coverage).
options = HabitatGraphFeatureOptions(include_extended_metrics=False)
options_5 = HabitatGraphFeatureOptions(
    include_extended_metrics=False,
    block_size=5,
)
# Full 3D HabitatMap.label_array for tables / heatmaps / delta.
# Do not extract from a 2D slice — the 2D network is display-only.
rows_8 = []
rows_5 = []
t0 = time.perf_counter()
for subject, habitat_map in zip(cohort, result.habitat_maps):
    labels_3d = habitat_map.label_array
    expected = habitat_map.habitat_ids
    feats_8 = extract_graph_features(
        labels_3d,
        options=options,
        expected_labels=expected,
    )
    feats_5 = extract_graph_features(
        labels_3d,
        options=options_5,
        expected_labels=expected,
    )
    rows_8.append({"subject_id": subject.subject_id, **feats_8})
    rows_5.append({"subject_id": subject.subject_id, **feats_5})
elapsed_s = time.perf_counter() - t0
table = pd.DataFrame(rows_8)
table_5 = pd.DataFrame(rows_5)
print(
    table.shape[0],
    "subjects x",
    table.shape[1] - 1,
    "graph features from full 3D habitat maps",
)
print(
    "block_size=8 nodes:",
    list(table["graph_num_nodes_total"]),
    "| block_size=5 nodes:",
    list(table_5["graph_num_nodes_total"]),
)
print(
    f"3D graph extract (block_size=8 and 5, {table.shape[0]} subjects): "
    f"{elapsed_s:.1f}s"
)
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort, result, table, options, SUBJECTS, MODALITIES.
from habit.viz import (
    plot_graph_feature_heatmap,
    plot_habitat_graph_network_2d,
    plot_habitat_graph_slice,
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
# Lattice on the representative slice (display-only; same options as extract).
fig = plot_habitat_graph_slice(
    labels, options=options, show_grid=True, block_size=8, grid_linestyle="--"
)
fig.savefig("out/graph_habitat_lattice_2d.png", dpi=150, bbox_inches="tight")
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
    "out/graph_habitat_lattice_2d.png, "
    "out/graph_habitat_network_2d.png, "
    "out/graph_feature_heatmap_single.png, and "
    "out/graph_feature_heatmap_pair.png"
)
# END figures

# BEGIN compare
# 2D networks are display-only (representative slice). Tables are 3D.
# Same HabitatMap / one fit. Only block_size changes (8 vs 5).
fig = plot_habitat_graph_network_2d(
    labels,
    options=options,
    show_grid=True,
    block_size=8,
    grid_linestyle="--",
)
fig.savefig("out/graph_habitat_network_2d.png", dpi=150, bbox_inches="tight")
fig = plot_habitat_graph_network_2d(
    labels,
    options=options_5,
    show_grid=True,
    block_size=5,
    grid_linestyle="--",
)
fig.savefig("out/graph_habitat_network_2d_block5.png", dpi=150, bbox_inches="tight")
# Same people and columns on both 3D tables (and the 5-minus-8 delta).
# Rank once on the 8-voxel 3D table; do not re-rank the 5-voxel table.
numeric_8 = table.set_index("subject_id").loc[list(SUBJECTS)]
single_cols = [
    name for name in numeric_8.columns if str(name).startswith("single_h")
]
compare_features = tuple(
    numeric_8[single_cols].var(axis=0, skipna=True).nlargest(40).index
)
fig = plot_graph_feature_heatmap(
    table,
    subjects=SUBJECTS,
    features=compare_features,
    zscore=True,
    title="Graph features: 8-voxel cubes (3D)",
)
fig.savefig("out/graph_feature_heatmap_block8.png", dpi=150, bbox_inches="tight")
fig = plot_graph_feature_heatmap(
    table_5,
    subjects=SUBJECTS,
    features=compare_features,
    zscore=True,
    title="Graph features: 5-voxel cubes (3D)",
)
fig.savefig("out/graph_feature_heatmap_block5.png", dpi=150, bbox_inches="tight")
# 3D table_5 minus 3D table (8-voxel): column z-score of the raw delta.
# star_significant marks features (not cells) after paired t + FDR-BH.
fig = plot_graph_feature_heatmap(
    table_5,
    reference=table,
    subjects=SUBJECTS,
    features=compare_features,
    zscore=True,
    star_significant=True,
    title="Graph features: 5-voxel minus 8-voxel",
    cbar_label="Z-scored difference (5 - 8)",
)
fig.savefig("out/graph_feature_delta_5_minus_8.png", dpi=150, bbox_inches="tight")
print(
    "Wrote out/graph_habitat_network_2d.png, "
    "out/graph_habitat_network_2d_block5.png, "
    "out/graph_feature_heatmap_block8.png, "
    "out/graph_feature_heatmap_block5.png, and "
    "out/graph_feature_delta_5_minus_8.png"
)
# END compare

if __name__ == "__main__":
    import sys

    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    # Gallery = copy of out/ from the visible blocks (same composition).
    copy_out_figures_to_gallery(
        (
            "graph_habitat_slice_2d.png",
            "graph_habitat_lattice_2d.png",
            "graph_habitat_network_2d.png",
            "graph_habitat_network_2d_block5.png",
            "graph_feature_heatmap_single.png",
            "graph_feature_heatmap_pair.png",
            "graph_feature_heatmap_block8.png",
            "graph_feature_heatmap_block5.png",
            "graph_feature_delta_5_minus_8.png",
        )
    )

    # Optional 3D renders need [view] / PyVista; keep them out of the recipe.
    # HABIT_NO_VIEW=1 skips them so the gallery smoke stays on the 2D path.
    import os

    if os.environ.get("HABIT_NO_VIEW", "").strip():
        print("HABIT_NO_VIEW set: skipped optional 3D gallery assets")
    else:
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
                labels,
                spacing=spacing_zyx,
                black_background=False,
                render_window=1200,
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
