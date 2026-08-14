#!/usr/bin/env python
"""
One subject → one-step habitats (K=4) → graph features + plots.

This gallery uses the library graph defaults: equal-volume cubes
(``node_method='uniform_grid'``, ``block_size=5`` voxels) and closest-voxel
edges (``edge_method='min_distance'``, ``distance_threshold=5``).
It only fixes ``n_habitats=4``.

Accompanies ``docs/source/examples/graph_features.rst``.
Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

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

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
# Fixed K=4 (not "auto") so the graph has a known number of habitats.
# Graph options are the library defaults (uniform 5-voxel cubes + min-distance).
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=4, random_seed=0, roi=ROI
).fit_predict(cohort)

labels = result.habitat_maps[0].label_array
options = HabitatGraphFeatureOptions()
feats = extract_graph_features(labels, options=options)
print(len(feats), "graph features")
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort, result, labels, options, and MODALITIES.
from habit.viz import plot_habitat_graph_network_2d, plot_habitat_overlay

Path("out").mkdir(exist_ok=True)
# Overlay: ImageVolume + HabitatMap (3-panel orthogonal default).
fig = plot_habitat_overlay(
    cohort[0].image(MODALITIES[0]),
    result.habitat_maps[0],
    title="One-step habitats (K=4)",
)
fig.savefig("out/graph_habitat_slice_2d.png", dpi=150, bbox_inches="tight")
# Same edge options as extract_graph_features so the 2D plot matches.
fig = plot_habitat_graph_network_2d(
    labels,
    options=options,
    show_grid=True,
    block_size=5,
    grid_linestyle="--",
)
fig.savefig("out/graph_habitat_network_2d.png", dpi=150, bbox_inches="tight")
print("Wrote out/graph_habitat_slice_2d.png and out/graph_habitat_network_2d.png")
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
