#!/usr/bin/env python
"""
Real-data habitat-feature contrast: existing maps → each_habitat + graph.

Loads ``demo_data/preprocessed`` plus already-written
``*_habitats.nrrd`` maps, crops each pair to the habitat foreground
(full-FOV demo volumes are large), extracts the **first-order**
``each_habitat`` bank (gallery-fast; not a full IBSI texture set)
together with ``volume`` and the default ``graph`` family, then melts
two wide blocks:

* ``habitat_{id}_{feature}`` → :func:`~habit.to_habitat_feature_panel`
* ``single_h{id}_{metric}`` → :func:`~habit.to_graph_habitat_panel`

Pair columns ``pair_h*_h*`` stay on the joined table and are drawn with
:func:`~habit.viz.plot_habitat_graph_pair_matrix`.

This script accompanies ``docs/source/examples/habitat_feature_compare.rst``.
This is a software demo, not a clinical claim.

Run from the repository root::

    python docs/source/examples/scripts/habitat_feature_compare_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from habit import (
    cohort_from_directory,
    compare_habitat_features,
    to_graph_habitat_panel,
    to_habitat_feature_panel,
)
from habit.adapters import (
    discover_habitat_map_paths,
    read_habitat_map,
    resolve_n_habitats,
)
from habit.contracts import ArrayImageRef, FeatureTable, Geometry, HabitatMap, Subject
from habit.domain.habitat_features import (
    EachHabitatRadiomicsFeatures,
    GraphHabitatFeatures,
    HabitatVolumeFeatures,
)
from habit.utils.progress_utils import CustomTqdm

# Change DATA / MAPS / MODALITIES / ROI to your preprocessed layout.
# MAPS is the get-habitat out_dir (files named <sid>_habitats.nrrd).
DATA = "demo_data/preprocessed"
MAPS = "demo_data/results/habitat_two_step"
MODALITIES = ("LAP",)
ROI = "LAP"
# Demo pack has five subjects. Paired Cliff's delta / BH-FDR need n >= 3.
N_SUBJECTS = 5

# First-order bank + volume: enough columns for a reviewer heatmap without
# a full IBSI texture wait. A paper pipeline would add glcm / glrlm here.
LIGHT_RADIOMICS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": [
            "Mean",
            "Median",
            "Minimum",
            "Maximum",
            "Range",
            "Variance",
            "StandardDeviation",
            "Skewness",
            "Kurtosis",
            "Energy",
            "TotalEnergy",
            "Entropy",
            "Uniformity",
            "10Percentile",
            "90Percentile",
            "InterquartileRange",
            "MeanAbsoluteDeviation",
            "RobustMeanAbsoluteDeviation",
            "RootMeanSquared",
        ],
    },
    "setting": {"additionalInfo": False, "correctMask": True},
}

# Interpretable single-habitat graph metrics for the topology heatmap.
GRAPH_NODE_METRICS = (
    "n_nodes",
    "n_edges",
    "edge_density",
    "avg_degree",
    "connected_components",
    "avg_edge_distance",
    "degree_cv",
    "n_nodes_per_habitat_volume",
)


def stack_subject_tables(tables: Sequence[FeatureTable]) -> FeatureTable:
    """
    Stack one-row-per-subject tables from the same feature family.

    Args:
        tables: Per-subject tables that share id and feature columns.

    Returns:
        One cohort table (rows = subjects).

    Raises:
        ValueError: If ``tables`` is empty.
    """
    if not tables:
        raise ValueError("stack_subject_tables: no tables to stack.")
    frame = pd.concat([table.frame for table in tables], ignore_index=True)
    return FeatureTable(
        frame=frame,
        id_columns=tables[0].id_columns,
        feature_columns=tables[0].feature_columns,
    )


def crop_subject_and_map(
    subject: Subject,
    habitat_map: HabitatMap,
    modality: str,
    *,
    pad: int = 5,
) -> Tuple[Subject, HabitatMap]:
    """
    Crop anatomy and habitat labels to a padded foreground bbox.

    Demo volumes are full-FOV. Radiomics and graph topology only need the
    ROI; cropping keeps this gallery on real data without a full-volume
    wait. Swap DATA / MAPS as usual -- the crop follows the habitat mask.

    Args:
        subject: Imaging subject (one or more modalities).
        habitat_map: Habitat labels aligned with the subject.
        modality: Image key to keep on the cropped subject.
        pad: Voxel padding around the habitat foreground.

    Returns:
        Cropped ``(subject, habitat_map)`` sharing one geometry.
    """
    volume = subject.image(modality)
    image = np.asarray(volume.data, dtype=np.float32)
    labels = np.asarray(habitat_map.label_array)
    foreground = labels > 0
    if not np.any(foreground):
        raise ValueError(f"No habitat voxels for {subject.subject_id!r}.")
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        start = max(0, int(lo) - pad)
        stop = min(int(labels.shape[axis]), int(hi) + pad + 1)
        slices.append(slice(start, stop))
    indexer = tuple(slices)
    image_c = image[indexer].copy()
    labels_c = labels[indexer].copy()
    spacing = tuple(float(v) for v in volume.spacing)
    geometry = Geometry.from_array(image_c.shape, spacing=spacing)
    cropped_subject = Subject(
        subject_id=subject.subject_id,
        images={modality: ArrayImageRef(array=image_c, geometry=geometry)},
        masks={
            modality: ArrayImageRef(
                array=(labels_c > 0).astype(np.int32),
                geometry=geometry,
            )
        },
    )
    cropped_map = HabitatMap(
        subject_id=habitat_map.subject_id,
        label_array=labels_c.astype(np.int32, copy=False),
        geometry=geometry,
        model_id=habitat_map.model_id,
        habitat_ids=habitat_map.habitat_ids,
        provenance=habitat_map.provenance,
    )
    return cropped_subject, cropped_map


def extract_one_subject(
    subject: Subject,
    habitat_map: HabitatMap,
    *,
    each: EachHabitatRadiomicsFeatures,
    graph: GraphHabitatFeatures,
    volume: HabitatVolumeFeatures,
) -> FeatureTable:
    """
    Run each_habitat + graph + volume on one subject and join the rows.

    Args:
        subject: Imaging subject (intensities used by each_habitat).
        habitat_map: Existing habitat labels for that subject.
        each: Per-habitat radiomics extractor.
        graph: Topology extractor (library defaults).
        volume: Per-habitat voxel-count / fraction extractor.

    Returns:
        One-row table: wide ``habitat_{id}_{feature}`` columns plus
        ``single_h*`` / ``pair_h*_h*`` graph columns.
    """
    each_table = each(subject, habitat_map)
    volume_table = volume(subject, habitat_map)
    graph_table = graph(subject, habitat_map)
    return each_table.join(volume_table).join(graph_table)


cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:N_SUBJECTS]
n_habitats = resolve_n_habitats(MAPS, None)
habitat_ids = tuple(range(1, int(n_habitats) + 1))
map_paths = discover_habitat_map_paths(MAPS, "*_habitats.nrrd")

# Domain extractors: graph uses library defaults (now a default light family).
each_extractor = EachHabitatRadiomicsFeatures(
    params=LIGHT_RADIOMICS,
    modalities=list(MODALITIES),
)
graph_extractor = GraphHabitatFeatures()
volume_extractor = HabitatVolumeFeatures()

per_subject: List[FeatureTable] = []
for subject in CustomTqdm(cohort, total=len(cohort), desc="Extract features"):
    sid = str(subject.subject_id)
    if sid not in map_paths:
        raise FileNotFoundError(
            f"No habitat map for {sid!r} under {MAPS}. "
            "Run get-habitat first or point MAPS at your *_habitats.nrrd folder."
        )
    habitat_map = read_habitat_map(
        map_paths[sid],
        subject_id=sid,
        habitat_ids=habitat_ids,
    )
    cropped_subject, cropped_map = crop_subject_and_map(
        subject, habitat_map, MODALITIES[0]
    )
    per_subject.append(
        extract_one_subject(
            cropped_subject,
            cropped_map,
            each=each_extractor,
            graph=graph_extractor,
            volume=volume_extractor,
        )
    )

table = stack_subject_tables(per_subject)
wide_cols = [
    name
    for name in table.feature_columns
    if str(name).startswith("habitat_")
]
graph_cols = [
    name
    for name in table.feature_columns
    if str(name).startswith("single_h") or str(name).startswith("pair_h")
]
print(
    f"Subjects: {list(cohort.subject_ids)}; habitats 1..{n_habitats}"
)
print(
    f"Joined table: {table.frame.shape[0]} rows x "
    f"{len(table.feature_columns)} columns "
    f"({len(wide_cols)} wide habitat_*, {len(graph_cols)} graph)"
)

# Two melts, same contrast API. Radiomics / volume use habitat_{id}_*;
# graph node metrics use single_h{id}_*. Pair columns stay on `table`.
panel = to_habitat_feature_panel(table)
graph_panel = to_graph_habitat_panel(table)
comparison = compare_habitat_features(panel)
graph_comparison = compare_habitat_features(graph_panel)
pair = comparison.strongest_pair()
subject_id = str(cohort[0].subject_id)
print(
    f"Panel: {panel.n_subjects} subjects, "
    f"habitats={panel.habitat_ids}, "
    f"features={len(panel.feature_names)}"
)
print(
    f"Cohort contrast: n={comparison.n_subjects}, "
    f"paired={comparison.paired}, effect={comparison.effect}, "
    f"strongest pair=H{pair[0]} vs H{pair[1]}"
)
print("Top absolute-effect features:", comparison.top_features(k=6, pair=pair))
print(
    f"Graph panel: {graph_panel.n_subjects} subjects, "
    f"features={len(graph_panel.feature_names)}; "
    f"pair columns remain on the wide table for the contact matrix"
)
# END example

# BEGIN figures
# Paste after the Script block. Uses comparison, graph_comparison, table,
# pair, subject_id, GRAPH_NODE_METRICS, and Path.
from habit.viz import (
    plot_habitat_feature_effect,
    plot_habitat_feature_heatmap,
    plot_habitat_feature_violin,
    plot_habitat_graph_pair_matrix,
    use_style,
)

Path("out").mkdir(exist_ok=True)


def _save(fig: object, name: str) -> None:
    """Save a figure under out/ and close it."""
    import matplotlib.pyplot as plt

    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote out/{name}")


graph_metric_names = [
    name for name in GRAPH_NODE_METRICS if name in graph_panel.feature_names
]

with use_style("radiology"):
    # Figure 1 -- cohort overview (all first-order + volume, z-scored).
    _save(
        plot_habitat_feature_heatmap(comparison),
        "habitat_feature_compare_heatmap.png",
    )
    # Figure 2 -- the claim: one pair, ranked Cliff's delta.
    _save(
        plot_habitat_feature_effect(comparison, pair=pair, top_k=20),
        "habitat_feature_compare_effect.png",
    )
    # Figure 3 -- only the top features that separate that pair.
    _save(
        plot_habitat_feature_violin(
            comparison, pair=pair, max_features=4, kind="box"
        ),
        "habitat_feature_compare_violin.png",
    )
    # Graph node metrics: same reviewer claim, topology instead of intensity.
    _save(
        plot_habitat_feature_heatmap(
            graph_comparison,
            features=graph_metric_names,
            title="Cohort graph metrics by habitat (z-scored)",
        ),
        "habitat_feature_compare_graph_heatmap.png",
    )
    # Graph pair values cannot melt; this is the honest contact figure.
    _save(
        plot_habitat_graph_pair_matrix(table, metric="contact_voxels_sum"),
        "habitat_feature_compare_graph_pairs.png",
    )
    # One case only -- not the cohort claim.
    _save(
        plot_habitat_feature_heatmap(
            comparison,
            subject_id=subject_id,
            title=f"One case ({subject_id})",
        ),
        "habitat_feature_compare_subject_heatmap.png",
    )
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    # Gallery = copy of out/ from the visible block (same composition).
    copy_out_figures_to_gallery(
        (
            "habitat_feature_compare_heatmap.png",
            "habitat_feature_compare_effect.png",
            "habitat_feature_compare_violin.png",
            "habitat_feature_compare_graph_heatmap.png",
            "habitat_feature_compare_graph_pairs.png",
            "habitat_feature_compare_subject_heatmap.png",
        )
    )
