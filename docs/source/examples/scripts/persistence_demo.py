#!/usr/bin/env python
"""
Persist and reload study artefacts with ``StudyResult.save``.

Writes NRRD habitat maps, ``habitat_model.habitatmodel``, feature tables
(parquet/csv), ``habitats.parquet`` unit table, run manifest, and optional
clustering figures — the same layout the CLI produces.

This script accompanies ``docs/source/examples/persistence.rst``.

Run from the repository root::

    python docs/source/examples/scripts/persistence_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from habit import HabitatModel, HabitatSpec, Spec, make_synthetic_cohort
import habit.recipes as recipes

cohort = make_synthetic_cohort(n_subjects=4, shape=(18, 18, 18), rng=99)
# Keyword order follows the runtime pipeline (not HabitatSpec field order).
spec = HabitatSpec(
    name="persistence_demo",
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi")),
    random_seed=99,
)

result = recipes.two_step(cohort, spec)

with tempfile.TemporaryDirectory(prefix="habit_persist_") as tmp:
    out_dir = Path(tmp) / "study_out"
    saved = result.save(
        out_dir,
        table_format="parquet",
        write_cluster_plots=True,
    )
    print(f"Saved study to {saved}")

    artefacts = sorted(p for p in out_dir.rglob("*") if p.is_file())
    print(f"\nArtefacts ({len(artefacts)} files):")
    for path in artefacts:
        print(f"  {path.relative_to(out_dir)}")

    manifest_path = out_dir / "run_manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"\nRunManifest keys: {sorted(manifest_data.keys())}")
    print("Methods snippet:", result.manifest.describe_methods()[:120], "...")

    model = HabitatModel.load(out_dir / "habitat_model.habitatmodel")
    print(f"\nReloaded HabitatModel: {model.model_id}, {model.n_habitats} habitats")

    prediction = recipes.apply_habitat_model(cohort, spec, model)
    print(f"Apply round-trip maps: {len(prediction.habitat_maps)}")
