#!/usr/bin/env python
"""
One subject → one-step habitats → graph features + 2D plot.

Accompanies ``docs/source/examples/graph_features.rst``.
Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

from habit import cohort_from_directory, extract_graph_features, one_step_habitat
from habit.viz import plot_habitat_graph_network_2d

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)

labels = result.habitat_maps[0].label_array
feats = extract_graph_features(labels)
print(len(feats), "graph features")

fig = plot_habitat_graph_network_2d(labels)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/graph_habitat_network_2d.png", dpi=150, bbox_inches="tight")
# END example

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from habit.viz import plot_habitat_overlay

    # Docs gallery assets (maintainers); not part of the user-facing recipe.
    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    image = cohort[0].image(MODALITIES[0]).data
    fig.savefig(gallery / "graph_habitat_network_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plot_habitat_overlay(
        image, labels, axis=0, alpha=0.45, title="One-step habitats"
    )
    fig.savefig(gallery / "graph_habitat_slice_2d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    try:
        from habit.viz import (
            render_habitat_graph_network_3d,
            render_habitat_graph_surface_3d,
        )

        vol = cohort[0].image(MODALITIES[0])
        spacing_zyx = (1.0, 1.0, 1.0)
        if vol.spacing is not None and len(vol.spacing) == 3:
            sx, sy, sz = (float(v) for v in vol.spacing)
            spacing_zyx = (sz, sy, sx)
        surface = render_habitat_graph_surface_3d(
            labels, spacing=spacing_zyx, black_background=False, render_window=1200
        )
        network = render_habitat_graph_network_3d(
            labels, spacing=spacing_zyx, black_background=False, render_window=1200
        )
        if surface is not None:
            plt.imsave(gallery / "graph_habitat_surface_3d.png", surface)
        if network is not None:
            plt.imsave(gallery / "graph_habitat_network_3d.png", network)
    except Exception as exc:  # pragma: no cover
        print(f"3D gallery assets skipped: {exc}")
