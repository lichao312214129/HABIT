#!/usr/bin/env python
"""
Feature composition: trees, combiners, statistics, and aliases.

Demonstrates the v1.0 feature-tree layer on a synthetic two-modality cohort:

* ``parse_feature_expression`` -- strict expression form (quoted modalities,
  explicit ``key=value`` parameters, nested combiner calls).
* Structured YAML / Python form -- identical Spec and fingerprint.
* Voxel trees -- ``concat`` / ``ratio`` / ``weighted_concat`` combiners and
  the ``as_`` alias, evaluated atomically on one subject.
* Supervoxel statistics -- ``mean`` / ``std`` / ``percentile`` inside a
  ``SubjectPipeline`` (fields are bound automatically).

This script accompanies ``docs/source/examples/feature_composition.rst``.

Run from the repository root::

    python docs/source/examples/scripts/feature_composition_demo.py
"""

from __future__ import annotations

# BEGIN example
from habit import HabitatSpec, Spec, make_synthetic_cohort, parse_feature_expression
from habit.domain import build_voxel_extractor
from habit.domain.assembly import build_habitat_components

cohort = make_synthetic_cohort(n_subjects=2, shape=(12, 12, 12), rng=7)
subject = cohort[0]

# --- 1. Expression form == structured form -----------------------------------
expression = 'concat(raw("T1"), local_entropy("T2", kernel_size=3))'
parsed = parse_feature_expression(expression)
structured = Spec(
    "concat",
    {
        "children": [
            {"name": "raw", "params": {"modality": "T1"}},
            {"name": "local_entropy", "params": {"modality": "T2", "kernel_size": 3}},
        ],
    },
)
print("=== expression -> Spec ===")
print(f"  expression: {expression}")
print(f"  parsed == structured: {parsed == structured}")
print(f"  fingerprint equal: {parsed.fingerprint() == structured.fingerprint()}")

# --- 2. Voxel tree, atomic call ----------------------------------------------
voxel_fx = build_voxel_extractor(parsed)
field = voxel_fx(subject)
print("\n=== voxel tree (atomic) ===")
print(f"  columns: {list(field.feature_frame().columns)}")

# --- 3. Combiners and aliases ------------------------------------------------
combo = parse_feature_expression(
    'concat('
    'weighted_concat(raw("T1", as_="t1w"), raw("T2", as_="t2w"), weights=[2.0, 1.0]), '
    'ratio(raw("T1"), raw("T2"), as_="t1_over_t2"))'
)
combo_field = build_voxel_extractor(combo)(subject)
print("\n=== weighted_concat + ratio with as_ aliases ===")
print(f"  columns: {list(combo_field.feature_frame().columns)}")

# --- 4. Supervoxel statistics inside SubjectPipeline -------------------------
spec = HabitatSpec(
    name="composition_stats",
    voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
    supervoxel_feature_extractor=Spec(
        "concat",
        {
            "children": [
                {"name": "mean", "params": {"modality": "T1"}},
                {"name": "std", "params": {"modality": "T1", "as_": "t1_spread"}},
                {"name": "percentile", "params": {"modality": "T2", "q": 90}},
            ],
        },
    ),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "inertia", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(
        Spec("volume"),
        Spec("msi"),
        Spec("ith_score"),
        Spec("non_radiomics"),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Spec("traditional"),
        # Spec("whole_habitat"),
        # Spec("each_habitat"),
    ),
    random_seed=7,
)
pipeline = build_habitat_components(spec).pipeline(assigner=None)
units = pipeline.units(subject)
print("\n=== supervoxel statistics tree (pipeline) ===")
print(f"  columns: {list(units.feature_frame().columns)}")
print(f"  units: {units.feature_frame().shape[0]} supervoxels")

# --- 5. YAML dual form -------------------------------------------------------
via_expression = HabitatSpec.from_dict(
    {
        "name": "dual",
        "voxel_feature_extractor": 'concat(raw("T1"), raw("T2"))',
        "supervoxelizer": {"name": "kmeans", "params": {"n_supervoxels": 6}},
        "habitat_model_fitter": {"name": "kmeans", "params": {"max_habitats": 3}},
        "habitat_assigner": {"name": "nearest_centroid"},
        "habitat_features": [
            {"name": "volume"},
            {"name": "msi"},
            {"name": "ith_score"},
            {"name": "non_radiomics"},
            # {"name": "traditional"},
            # {"name": "whole_habitat"},
            # {"name": "each_habitat"},
        ],
    }
)
via_structured = HabitatSpec.from_dict(
    {
        "name": "dual",
        "voxel_feature_extractor": {
            "name": "concat",
            "params": {
                "children": [
                    {"name": "raw", "params": {"modality": "T1"}},
                    {"name": "raw", "params": {"modality": "T2"}},
                ],
            },
        },
        "supervoxelizer": {"name": "kmeans", "params": {"n_supervoxels": 6}},
        "habitat_model_fitter": {"name": "kmeans", "params": {"max_habitats": 3}},
        "habitat_assigner": {"name": "nearest_centroid"},
        "habitat_features": [
            {"name": "volume"},
            {"name": "msi"},
            {"name": "ith_score"},
            {"name": "non_radiomics"},
            # {"name": "traditional"},
            # {"name": "whole_habitat"},
            # {"name": "each_habitat"},
        ],
    }
)
print("\n=== YAML dual form ===")
print(f"  fingerprint equal: "
      f"{via_expression.fingerprint() == via_structured.fingerprint()}")
# END example

# BEGIN figures
# Paste after the Script block. Uses spec, cohort, and subject.
from pathlib import Path

import habit.recipes as recipes
from habit.viz import plot_habitat_overlay

result = recipes.Study(spec=spec).fit_predict(cohort)
fig = plot_habitat_overlay(
    subject.image("T1").data,
    result.habitat_maps[0].label_array,
    axis=0,
    title="Habitats from composed feature tree",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/feature_composition_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/feature_composition_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig, "feature_composition_overlay.png")
