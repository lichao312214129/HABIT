#!/usr/bin/env python
"""
Synthetic habitat-graph topology features with publication figures.

Shows the preferred public paths:

* Kernel: ``habit.extract_graph_features`` / ``HabitatGraphFeatureOptions``
* Domain: ``HabitatFeatureExtractorRegistry.create("graph", ...)``
* Viz: ``habit.viz.habitat_graph`` 2D matplotlib panels and 3D PyVista renders

Writes English-labelled PNGs under ``docs/source/_static/images/examples/`` for
the Sphinx Examples gallery. Requires matplotlib; 3D assets require pyvista and
scikit-image (``pip install 'habit[view]'`` or ``pip install pyvista scikit-image``).

Accompanies ``docs/source/examples/graph_features.rst``.

Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

import habit.domain  # registers built-in habitat feature extractors
from habit import HabitatGraphFeatureOptions, extract_graph_features
from habit.contracts import ArrayImageRef, Geometry, HabitatMap, Provenance, Subject
from habit.domain import HabitatFeatureExtractorRegistry
from habit.viz import (
    plot_habitat_graph_network_2d,
    plot_habitat_graph_slice,
    render_habitat_graph_network_3d,
    render_habitat_graph_surface_3d,
    use_style,
)

SHAPE: Tuple[int, int, int] = (40, 40, 40)
OUT_DIR = (
    Path(__file__).resolve().parents[2] / "_static" / "images" / "examples"
)


def make_synthetic_labels() -> np.ndarray:
    """
    Build a visually readable 3D multi-habitat map for graph demos.

    All three habitats occupy the mid axial slab so 2D slice/network figures
    show every class; the full volume remains rich enough for 3D renders.

    Returns:
        Integer label array; background is 0, habitats are 1, 2, and 3.
    """
    labels = np.zeros(SHAPE, dtype=np.int32)
    # Paint a thick mid slab (axis 0) so the representative / mid slice is busy.
    z0, z1 = 14, 26
    # Habitat 1: three nearby islands (intra edges under a generous threshold).
    labels[z0:z1, 4:12, 4:12] = 1
    labels[z0:z1, 4:11, 16:24] = 1
    labels[z0:z1, 14:22, 6:13] = 1
    # Habitat 2: two larger masses that flank habitat 3.
    labels[z0:z1, 14:24, 18:28] = 2
    labels[z0:z1, 26:34, 8:18] = 2
    # Habitat 3: central bridge plus a satellite for intra edges.
    labels[z0:z1, 16:24, 12:20] = 3
    labels[z0:z1, 28:35, 22:30] = 3
    return labels


def _demo_options() -> HabitatGraphFeatureOptions:
    """Return deterministic options that keep small synthetic regions as nodes."""
    return HabitatGraphFeatureOptions(
        edge_method="centroid_distance",
        distance_threshold=14.0,
        erosion_radius=0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
        pairwise_include_intra_edges=True,
    )


def kernel_path(labels: np.ndarray) -> Dict[str, float]:
    """
    Extract graph features from a plain label array (arrays in, dict out).

    Args:
        labels: Habitat label map.

    Returns:
        Flat feature dictionary from the L0 kernel.
    """
    return extract_graph_features(
        labels,
        options=_demo_options(),
        expected_labels=(1, 2, 3),
    )


def domain_path(labels: np.ndarray) -> Dict[str, float]:
    """
    Extract graph features via the domain registry (Subject + HabitatMap).

    Args:
        labels: Habitat label map.

    Returns:
        One-row feature mapping from the returned FeatureTable frame.
    """
    geometry = Geometry.from_array(SHAPE, spacing=(1.0, 1.0, 1.0))
    subject = Subject(
        subject_id="synth_graph_001",
        images={},
        masks={
            "tumor": ArrayImageRef(
                array=(labels > 0).astype(np.int32),
                geometry=geometry,
            )
        },
    )
    habitat_map = HabitatMap(
        subject_id=subject.subject_id,
        label_array=labels,
        geometry=geometry,
        model_id="synthetic-graph-demo",
        habitat_ids=(1, 2, 3),
        provenance=Provenance.source("docs.graph_features_demo"),
    )
    options = _demo_options()
    extractor = HabitatFeatureExtractorRegistry.create(
        "graph",
        edge_method=options.edge_method,
        distance_threshold=options.distance_threshold,
        erosion_radius=options.erosion_radius,
        subdivide_region_voxels=options.subdivide_region_voxels,
        include_extended_metrics=options.include_extended_metrics,
    )
    table = extractor(subject, habitat_map)
    row = table.frame.iloc[0]
    return {str(key): float(row[key]) for key in table.feature_columns}


def save_publication_figures(labels: np.ndarray) -> Dict[str, Path]:
    """
    Render 2D and 3D habitat-graph figures and save them for the docs gallery.

    Args:
        labels: 3D synthetic habitat label volume.

    Returns:
        Mapping from asset stem to the written PNG path.

    Raises:
        RuntimeError: When a required 2D figure cannot be built, or when 3D
            dependencies / renders fail (examples must ship both 2D and 3D).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    options = _demo_options()
    written: Dict[str, Path] = {}

    # --- 2D: habitat slice + network layout (journal styles) ---
    # Leave slice_index=None so viz picks the largest cross-section after its
    # padded foreground crop (explicit indices must account for that pad).
    with use_style("radiology"):
        fig_slice = plot_habitat_graph_slice(labels)
    slice_path = OUT_DIR / "graph_habitat_slice_2d.png"
    fig_slice.savefig(slice_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig_slice)
    written["slice_2d"] = slice_path
    print(f"Wrote {slice_path} ({slice_path.stat().st_size} bytes)")

    with use_style("nature"):
        fig_network = plot_habitat_graph_network_2d(labels, options=options)
    if fig_network is None:
        raise RuntimeError("2D network figure is empty; check synthetic labels.")
    network_path = OUT_DIR / "graph_habitat_network_2d.png"
    fig_network.savefig(network_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig_network)
    written["network_2d"] = network_path
    print(f"Wrote {network_path} ({network_path.stat().st_size} bytes)")

    # --- 3D: surface + spatial network (off-screen PyVista) ---
    try:
        import pyvista  # noqa: F401
        import skimage  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "3D example assets require pyvista and scikit-image. "
            "Install with: pip install pyvista scikit-image "
            "(or pip install 'habit[view]')."
        ) from exc

    surface_rgb = render_habitat_graph_surface_3d(
        labels,
        spacing=(1.0, 1.0, 1.0),
        black_background=False,
        render_window=1400,
        surface_smooth_iter=20,
    )
    network_rgb = render_habitat_graph_network_3d(
        labels,
        options=options,
        spacing=(1.0, 1.0, 1.0),
        black_background=False,
        render_window=1400,
    )
    if surface_rgb is None or network_rgb is None:
        raise RuntimeError(
            "3D render returned None; enlarge synthetic habitats or relax options."
        )

    surface_path = OUT_DIR / "graph_habitat_surface_3d.png"
    network_3d_path = OUT_DIR / "graph_habitat_network_3d.png"
    plt.imsave(surface_path, surface_rgb)
    plt.imsave(network_3d_path, network_rgb)
    written["surface_3d"] = surface_path
    written["network_3d"] = network_3d_path
    print(f"Wrote {surface_path} ({surface_path.stat().st_size} bytes)")
    print(f"Wrote {network_3d_path} ({network_3d_path.stat().st_size} bytes)")
    return written


def main() -> None:
    """Print representative columns and regenerate gallery PNG assets."""
    labels = make_synthetic_labels()
    kernel_feats = kernel_path(labels)
    domain_feats = domain_path(labels)

    keys = (
        "single_h1_n_nodes",
        "single_h1_n_edges",
        "single_h2_n_nodes",
        "single_h3_n_nodes",
        "pair_h1_h2_n_edges",
        "pair_h1_h3_n_edges",
    )
    print("Kernel path (extract_graph_features):")
    for key in keys:
        print(f"  {key}: {kernel_feats.get(key)}")
    print("Domain path (HabitatFeatureExtractorRegistry.create('graph')):")
    for key in keys:
        print(f"  {key}: {domain_feats.get(key)}")
    print(f"Kernel feature count: {len(kernel_feats)}")
    print(f"Domain feature count: {len(domain_feats)}")

    written = save_publication_figures(labels)
    print(f"\nGallery assets ({len(written)}) under {OUT_DIR}")


if __name__ == "__main__":
    main()
