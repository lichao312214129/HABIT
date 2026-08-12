#!/usr/bin/env python
"""
Synthetic habitat-graph topology features (no demo_data required).

Shows the preferred public paths:

* Kernel: ``habit.extract_graph_features`` / ``HabitatGraphFeatureOptions``
* Domain: ``HabitatFeatureExtractorRegistry.create("graph", ...)``

Accompanies ``docs/source/examples/graph_features.rst``.

Run from the repository root::

    python docs/source/examples/scripts/graph_features_demo.py
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import habit.domain  # registers built-in habitat feature extractors
from habit import HabitatGraphFeatureOptions, extract_graph_features
from habit.contracts import ArrayImageRef, Geometry, HabitatMap, Provenance, Subject
from habit.domain import HabitatFeatureExtractorRegistry


SHAPE: Tuple[int, int, int] = (24, 24, 24)


def make_synthetic_labels() -> np.ndarray:
    """
    Build a small 3D label map with two fragmented habitats.

    Returns:
        Integer label array; background is 0, habitats are 1 and 2.
    """
    labels = np.zeros(SHAPE, dtype=np.int32)
    # Habitat 1: two separated blobs (become multiple graph nodes after CC).
    labels[4:10, 4:10, 4:10] = 1
    labels[4:10, 14:20, 14:20] = 1
    # Habitat 2: one larger contiguous region near habitat 1.
    labels[10:18, 8:16, 8:16] = 2
    return labels


def kernel_path(labels: np.ndarray) -> Dict[str, float]:
    """
    Extract graph features from a plain label array (arrays in, dict out).

    Args:
        labels: Habitat label map.

    Returns:
        Flat feature dictionary from the L0 kernel.
    """
    options = HabitatGraphFeatureOptions(
        edge_method="centroid_distance",
        distance_threshold=8.0,
        erosion_radius=0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )
    return extract_graph_features(
        labels,
        options=options,
        expected_labels=(1, 2),
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
    # GraphHabitatFeatures uses subject_id only; intensities are unused.
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
        habitat_ids=(1, 2),
        provenance=Provenance.source("docs.graph_features_demo"),
    )
    extractor = HabitatFeatureExtractorRegistry.create(
        "graph",
        edge_method="centroid_distance",
        distance_threshold=8.0,
        erosion_radius=0,
        subdivide_region_voxels=0,
        include_extended_metrics=False,
    )
    table = extractor(subject, habitat_map)
    row = table.frame.iloc[0]
    return {str(key): float(row[key]) for key in table.feature_columns}


def main() -> None:
    """Print a few representative graph columns from both call paths."""
    labels = make_synthetic_labels()
    kernel_feats = kernel_path(labels)
    domain_feats = domain_path(labels)

    keys = (
        "single_h1_n_nodes",
        "single_h1_n_edges",
        "single_h2_n_nodes",
        "pair_h1_h2_n_edges",
    )
    print("Kernel path (extract_graph_features):")
    for key in keys:
        print(f"  {key}: {kernel_feats.get(key)}")
    print("Domain path (HabitatFeatureExtractorRegistry.create('graph')):")
    for key in keys:
        print(f"  {key}: {domain_feats.get(key)}")
    print(f"Kernel feature count: {len(kernel_feats)}")
    print(f"Domain feature count: {len(domain_feats)}")


if __name__ == "__main__":
    main()
