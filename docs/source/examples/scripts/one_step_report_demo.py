#!/usr/bin/env python
"""
One-step habitats with a Report: persist and draw as each subject completes.

Accompanies ``docs/source/examples/one_step_habitat.rst`` (Stream per
subject) and ``docs/source/examples/persistence.rst``.

Run from the repository root::

    python docs/source/examples/scripts/one_step_report_demo.py
"""

from __future__ import annotations

# BEGIN example
from dataclasses import asdict
from pathlib import Path

from habit import (
    ClusterValidation,
    GraphNetwork2D,
    GraphSlice,
    HabitatGraphFeatureOptions,
    HabitatSpec,
    ITH,
    MSI,
    Overlay,
    Report,
    Spec,
    Stage,
    Study,
    VolumeFractions,
    make_synthetic_cohort,
)
from habit.adapters import DirectoryResultWriter
from habit.execution import CheckpointStore

# Synthetic cohort so this script stays offline / smoke-safe.
# Swap for cohort_from_directory(...) on a real preprocessed tree.
cohort = make_synthetic_cohort(
    n_subjects=2,
    modalities=("T1", "T2"),
    shape=(16, 16, 16),
    n_subregions=3,
    rng=0,
)
# Same options object for Spec("graph") and the 2D Report atoms.
# 2D PNGs are a representative slice (display-only); metrics use the 3D volume.
# Library default is include_extended_metrics=True. Pin False so this
# synthetic smoke stays fast.
graph = HabitatGraphFeatureOptions(
    edge_method="min_distance",
    block_size=8,
    distance_threshold=5.0,
    include_extended_metrics=False,
)
spec = HabitatSpec(
    name="one_step_report",
    stages=(
        Stage(
            "extract_voxel_features",
            Spec("raw", {"modalities": ["T1", "T2"], "roi": "tumor"}),
        ),
        Stage(
            "fit",
            Spec("kmeans", {"n_habitats": 3, "n_init": 2}),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("graph", asdict(graph))),
    ),
    random_seed=0,
)

out_dir = Path("out/one_step_report")
writer = DirectoryResultWriter(out_dir)
# Report is not a HabitatSpec stage: changing figures does not invalidate
# scientific checkpoints. Each completed subject is persisted and drawn
# before the next one accumulates in memory.
# BEGIN report
report = Report(
    persist=("habitat_map", "subject_model"),
    retain="tables",
    figures=(
        Overlay(modality="T1"),
        VolumeFractions(),
        MSI(),
        ITH(),
        ClusterValidation(),
        GraphSlice(options=graph),
        GraphNetwork2D(options=graph),
    ),
    writer=writer,
    figure_layout="by_subject",
)
# END report
result = Study(spec=spec, design="one_step").fit_predict(
    cohort,
    checkpoint=CheckpointStore(out_dir / ".ckpt"),
    report=report,
)
# Maps and per-subject models are already on disk; save writes tables + manifest.
result.save(out_dir, write_cluster_plots=False)
print(f"In-memory maps (retain='tables'): {len(result.habitat_maps)}")
print(f"Per-subject models: {len(result.subject_models)}")
print("On disk:")
for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
    print(f"  {path.relative_to(out_dir)}")
# END example

if __name__ == "__main__":
    print("Done. See docs/source/examples/one_step_habitat.rst")
