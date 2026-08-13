#!/usr/bin/env python
"""
Persist and reload study artefacts with ``StudyResult.save``.

Writes NRRD habitat maps, ``habitat_model.habitatmodel``, feature tables
(parquet/csv), ``habitats.parquet`` unit table, run manifest, and optional
clustering figures — the same layout the CLI produces.

Primary API: HabitatSpec.stages + recipes.Study(...).fit_predict.

This script accompanies ``docs/source/examples/persistence.rst``.

Run from the repository root::

    python docs/source/examples/scripts/persistence_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from habit import HabitatModel, HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes

cohort = make_synthetic_cohort(n_subjects=4, shape=(18, 18, 18), rng=99)
spec = HabitatSpec(
    name="persistence_demo",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "elbow",
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Stage("quantify5", Spec("traditional")),
        # Stage("quantify6", Spec("whole_habitat")),
        # Stage("quantify7", Spec("each_habitat")),
    ),
    random_seed=99,
)

result = recipes.Study(spec=spec).fit_predict(cohort)

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

    prediction = recipes.Study.from_model(model, spec).predict(cohort)
    print(f"Apply round-trip maps: {len(prediction.habitat_maps)}")

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, result)
