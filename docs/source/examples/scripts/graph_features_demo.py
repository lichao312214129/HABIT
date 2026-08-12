#!/usr/bin/env python
"""
One subject → one-step habitats → graph features + 2D plot (sklearn-short).

Accompanies ``docs/source/examples/graph_features.rst``.
Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

# BEGIN example
import matplotlib.pyplot as plt

from habit import HabitatGraphFeatureOptions, HabitatSpec, Spec, Stage, extract_graph_features
import habit.recipes as recipes
from habit.viz import plot_habitat_graph_network_2d, use_style

from _example_roi import crop_pair, examples_image_dir, one_subject_cohort

# 1. One subject (demo_data/preprocessed if present, else synthetic)
cohort, modalities, _ = one_subject_cohort()
modality = modalities[0]

# 2. One-step habitats
spec = HabitatSpec(
    name="one_step_graph",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(modalities)})),
        Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 5})),
        Stage("assign", Spec("nearest_centroid")),
    ),
    random_seed=0,
)
result = recipes.Study(spec=spec).fit_predict(cohort)
image, labels = crop_pair(cohort[0].image(modality).data, result.habitat_maps[0].label_array)

# 3. Graph features + 2D network plot
opts = HabitatGraphFeatureOptions(distance_threshold=8.0, subdivide_region_voxels=1000)
feats = extract_graph_features(labels, options=opts)
print(f"{len(feats)} graph features; sample:", list(feats.items())[:3])

out = examples_image_dir()
with use_style("nature"):
    fig = plot_habitat_graph_network_2d(labels, options=opts)
fig.savefig(out / "graph_habitat_network_2d.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
# END example

if __name__ == "__main__":
    from habit.viz import plot_habitat_overlay, use_style as _use_style

    with _use_style("radiology"):
        fig = plot_habitat_overlay(
            image, labels, axis=0, alpha=0.45, title="One-step habitats"
        )
    fig.savefig(out / "graph_habitat_slice_2d.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    try:
        from habit.viz import (
            render_habitat_graph_network_3d,
            render_habitat_graph_surface_3d,
        )

        vol = cohort[0].image(modality)
        spacing_zyx = (1.0, 1.0, 1.0)
        if vol.spacing is not None and len(vol.spacing) == 3:
            sx, sy, sz = (float(v) for v in vol.spacing)
            spacing_zyx = (sz, sy, sx)
        surface = render_habitat_graph_surface_3d(
            labels, spacing=spacing_zyx, black_background=False, render_window=1200
        )
        network = render_habitat_graph_network_3d(
            labels,
            options=opts,
            spacing=spacing_zyx,
            black_background=False,
            render_window=1200,
        )
        if surface is not None:
            plt.imsave(out / "graph_habitat_surface_3d.png", surface)
        if network is not None:
            plt.imsave(out / "graph_habitat_network_3d.png", network)
    except Exception as exc:  # pragma: no cover
        print(f"3D gallery assets skipped: {exc}")
