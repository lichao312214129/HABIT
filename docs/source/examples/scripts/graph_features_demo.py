#!/usr/bin/env python
"""
Habitat-graph topology features on real demo_data habitat maps.

Shows the preferred public paths:

* Kernel: ``habit.extract_graph_features`` / ``HabitatGraphFeatureOptions``
* Domain: ``HabitatFeatureExtractorRegistry.create("graph", ...)``
* Viz: ``habit.viz`` anatomy overlay, 2D network panels, and 3D PyVista renders

Writes English-labelled PNGs under ``docs/source/_static/images/examples/`` for
the Sphinx Examples gallery. Requires ``demo_data/`` (not shipped in the git
tree) plus matplotlib; 3D assets require pyvista and scikit-image
(``pip install 'habit[view]'`` or ``pip install pyvista scikit-image``).

Accompanies ``docs/source/examples/graph_features.rst``.

Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import habit.domain  # registers built-in habitat feature extractors
from habit import HabitatGraphFeatureOptions, extract_graph_features
from habit.contracts import ArrayImageRef, Geometry, HabitatMap, Provenance, Subject
from habit.domain import HabitatFeatureExtractorRegistry
from habit.viz import (
    plot_habitat_graph_network_2d,
    plot_habitat_overlay,
    render_habitat_graph_network_3d,
    render_habitat_graph_surface_3d,
    use_style,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = (
    Path(__file__).resolve().parents[2] / "_static" / "images" / "examples"
)

# Preferred maintainer demo paths (subject subj001, two-step habitats + LAP).
HABITAT_CANDIDATES: Tuple[Path, ...] = (
    REPO_ROOT
    / "demo_data"
    / "results"
    / "habitat_two_step"
    / "subj001_habitats.nrrd",
    REPO_ROOT
    / "demo_data"
    / "results"
    / "habitat_two_step_v1"
    / "subj001_habitats.nrrd",
    REPO_ROOT
    / "demo_data"
    / "results"
    / "examples"
    / "habitat_v1_two_step_demo"
    / "subj001_habitats.nrrd",
)
IMAGE_CANDIDATES: Tuple[Path, ...] = (
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "images"
    / "subj001"
    / "LAP"
    / "WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd",
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "images"
    / "subj001"
    / "PVP"
    / "WATER__WATER__WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd",
)

# Crop padding (voxels) around the non-background habitat bbox for figures.
BBOX_PAD: int = 5


def _first_existing(paths: Sequence[Path]) -> Path | None:
    """Return the first path that exists, or ``None``."""
    for path in paths:
        if path.is_file():
            return path
    return None


def _demo_data_missing_message() -> str:
    """Build a clear error when local demo_data habitat maps are absent."""
    tried = "\n".join(f"  - {path}" for path in HABITAT_CANDIDATES)
    return (
        "demo_data habitat map not found. This example requires local "
        "demo_data (not committed to git).\n"
        "Looked for:\n"
        f"{tried}\n"
        "Generate habitats with the two-step demo / CLI, e.g.\n"
        "  habit get-habitat -c config/habitat/config_habitat_two_step.yaml\n"
        "Committed gallery PNGs under docs/source/_static/images/examples/ "
        "were produced from demo_data on the maintainer machine."
    )


def load_demo_volumes() -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float], Path, Path]:
    """
    Load subj001 habitat labels and a matching anatomy image from demo_data.

    Returns:
        Tuple of ``(labels, image, spacing_xyz, habitat_path, image_path)`` where
        arrays are NumPy ``(z, y, x)`` from SimpleITK and spacing is SimpleITK
        ``(x, y, z)`` in millimetres.

    Raises:
        FileNotFoundError: When neither habitat maps nor a usable image exist.
    """
    habitat_path = _first_existing(HABITAT_CANDIDATES)
    if habitat_path is None:
        raise FileNotFoundError(_demo_data_missing_message())

    image_path = _first_existing(IMAGE_CANDIDATES)
    if image_path is None:
        raise FileNotFoundError(
            "demo_data anatomy image not found for subj001. Looked for:\n"
            + "\n".join(f"  - {path}" for path in IMAGE_CANDIDATES)
        )

    habitat_img = sitk.ReadImage(str(habitat_path))
    anatomy_img = sitk.ReadImage(str(image_path))
    labels = np.asarray(sitk.GetArrayFromImage(habitat_img), dtype=np.int32)
    image = np.asarray(sitk.GetArrayFromImage(anatomy_img), dtype=np.float32)
    if labels.shape != image.shape:
        raise RuntimeError(
            "Habitat and anatomy shapes differ; cannot overlay. "
            f"labels={labels.shape} image={image.shape} "
            f"({habitat_path} vs {image_path})"
        )
    spacing_xyz = tuple(float(v) for v in habitat_img.GetSpacing())
    if len(spacing_xyz) != 3:
        raise RuntimeError(f"Expected 3D spacing, got {spacing_xyz!r}")
    return labels, image, spacing_xyz, habitat_path, image_path


def crop_to_habitat_bbox(
    labels: np.ndarray,
    image: np.ndarray,
    *,
    pad: int = BBOX_PAD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop both volumes to the padded bounding box of non-background habitats.

    Args:
        labels: Full habitat label volume ``(z, y, x)``.
        image: Matching anatomy volume.
        pad: Voxel padding on each side (clipped to array bounds).

    Returns:
        Cropped ``(labels, image)`` arrays sharing the same shape.
    """
    foreground = labels > 0
    if not np.any(foreground):
        raise RuntimeError("Habitat map has no non-background voxels.")
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        start = max(0, int(lo) - pad)
        stop = min(int(labels.shape[axis]), int(hi) + pad + 1)
        slices.append(slice(start, stop))
    indexer = tuple(slices)
    return labels[indexer].copy(), image[indexer].copy()


def _demo_options() -> HabitatGraphFeatureOptions:
    """
    Return graph options suited to real demo habitats.

    Defaults keep connected regions that exceed 1000 voxels subdivided so the
    network figures stay readable on the cropped tumour ROI.
    """
    return HabitatGraphFeatureOptions(
        edge_method="centroid_distance",
        distance_threshold=8.0,
        erosion_radius=0,
        subdivide_region_voxels=1000,
        include_extended_metrics=False,
        pairwise_include_intra_edges=True,
    )


def _expected_labels(labels: np.ndarray) -> Tuple[int, ...]:
    """Sorted positive habitat IDs present in ``labels``."""
    present = np.unique(labels)
    return tuple(int(v) for v in present if int(v) > 0)


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
        expected_labels=_expected_labels(labels),
    )


def domain_path(labels: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> Dict[str, float]:
    """
    Extract graph features via the domain registry (Subject + HabitatMap).

    Args:
        labels: Habitat label map.
        spacing_xyz: SimpleITK spacing ``(x, y, z)`` in millimetres.

    Returns:
        One-row feature mapping from the returned FeatureTable frame.
    """
    # Geometry.spacing uses SimpleITK physical axis order (x, y, z).
    geometry = Geometry.from_array(labels.shape, spacing=spacing_xyz)
    subject = Subject(
        subject_id="subj001",
        images={},
        masks={
            "tumor": ArrayImageRef(
                array=(labels > 0).astype(np.int32),
                geometry=geometry,
            )
        },
    )
    habitat_ids = _expected_labels(labels)
    habitat_map = HabitatMap(
        subject_id=subject.subject_id,
        label_array=labels,
        geometry=geometry,
        model_id="demo-habitat-two-step",
        habitat_ids=habitat_ids,
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


def save_publication_figures(
    labels: np.ndarray,
    image: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Dict[str, Path]:
    """
    Render 2D and 3D habitat-graph figures and save them for the docs gallery.

    Args:
        labels: Cropped 3D habitat label volume.
        image: Matching cropped anatomy volume.
        spacing_xyz: SimpleITK spacing ``(x, y, z)``.

    Returns:
        Mapping from asset stem to the written PNG path.

    Raises:
        RuntimeError: When a required 2D figure cannot be built, or when 3D
            dependencies / renders fail (examples must ship both 2D and 3D).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    options = _demo_options()
    written: Dict[str, Path] = {}

    # --- 2D: anatomy + habitat overlay (densest axial slice) ---
    with use_style("radiology"):
        fig_slice = plot_habitat_overlay(
            image,
            labels,
            axis=0,
            alpha=0.45,
            spacing=spacing_xyz,
            title="Habitat overlay on LAP (demo_data subj001)",
        )
    slice_path = OUT_DIR / "graph_habitat_slice_2d.png"
    fig_slice.savefig(slice_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig_slice)
    written["slice_2d"] = slice_path
    print(f"Wrote {slice_path} ({slice_path.stat().st_size} bytes)")

    # --- 2D: region network on the representative habitat cross-section ---
    with use_style("nature"):
        fig_network = plot_habitat_graph_network_2d(labels, options=options)
    if fig_network is None:
        raise RuntimeError("2D network figure is empty; check demo habitat labels.")
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

    # NumPy (z, y, x) spacing for viz helpers that accept array-axis spacing.
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    surface_rgb = render_habitat_graph_surface_3d(
        labels,
        spacing=spacing_zyx,
        black_background=False,
        render_window=1400,
        surface_smooth_iter=20,
    )
    network_rgb = render_habitat_graph_network_3d(
        labels,
        options=options,
        spacing=spacing_zyx,
        black_background=False,
        render_window=1400,
    )
    if surface_rgb is None or network_rgb is None:
        raise RuntimeError(
            "3D render returned None; check cropped demo habitats or options."
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
    labels_full, image_full, spacing_xyz, habitat_path, image_path = load_demo_volumes()
    labels, image = crop_to_habitat_bbox(labels_full, image_full)
    habitat_ids = _expected_labels(labels)
    print(f"Habitat map: {habitat_path.relative_to(REPO_ROOT)}")
    print(f"Anatomy:     {image_path.relative_to(REPO_ROOT)}")
    print(
        f"Cropped ROI shape: {labels.shape} "
        f"(from {labels_full.shape}); habitats={habitat_ids}"
    )

    kernel_feats = kernel_path(labels)
    domain_feats = domain_path(labels, spacing_xyz)

    # Print a few columns for the lowest habitat IDs present in the map.
    sample_keys = []
    for hid in habitat_ids[:3]:
        sample_keys.append(f"single_h{hid}_n_nodes")
        sample_keys.append(f"single_h{hid}_n_edges")
    if len(habitat_ids) >= 2:
        sample_keys.append(f"pair_h{habitat_ids[0]}_h{habitat_ids[1]}_n_edges")

    print("Kernel path (extract_graph_features):")
    for key in sample_keys:
        print(f"  {key}: {kernel_feats.get(key)}")
    print("Domain path (HabitatFeatureExtractorRegistry.create('graph')):")
    for key in sample_keys:
        print(f"  {key}: {domain_feats.get(key)}")
    print(f"Kernel feature count: {len(kernel_feats)}")
    print(f"Domain feature count: {len(domain_feats)}")

    written = save_publication_figures(labels, image, spacing_xyz)
    print(f"\nGallery assets ({len(written)}) under {OUT_DIR}")


if __name__ == "__main__":
    main()
