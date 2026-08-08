#!/usr/bin/env python
"""
Habitat voxel-feature routes: raw, concat, radiomics, and SLIC supervoxels.

Demonstrates the main feature-construction paths referenced in habitat YAML:

* ``raw`` — concatenate modality intensities (always runnable).
* ``concat`` — join heterogeneous families (here two ``raw`` branches).
* ``slic`` — spatially coherent supervoxels instead of k-means.
* ``voxel_radiomics`` / ``supervoxel_radiomics`` — PyRadiomics texture
  (requires ``demo_data/`` and PyRadiomics; skipped gracefully otherwise).

Both **batch** (``recipes.two_step(cohort, spec)``) and **atomic**
(``SubjectPipeline.units(subject)``) calls are shown.

This script accompanies ``docs/source/examples/habitat_feature_routes.rst``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_feature_routes_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

from habit import HabitatSpec, Spec, cohort_from_directory, make_synthetic_cohort
from habit.domain.assembly import build_habitat_components
import habit.recipes as recipes

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
IMAGING_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed"
PARAMS_VOXEL: Path = (
    REPO_ROOT / "habit" / "resources" / "radiomics" / "params_voxel_radiomics.yaml"
)
PARAMS_SV: Path = (
    REPO_ROOT / "habit" / "resources" / "radiomics" / "params_supervoxel_radiomics.yaml"
)


def _load_cohort() -> Tuple[object, Sequence[str]]:
    """Return a cohort and modality list (demo or synthetic)."""
    if IMAGING_ROOT.is_dir():
        modalities = ("pre_contrast", "LAP", "PVP", "delay_3min")
        cohort = cohort_from_directory(
            IMAGING_ROOT,
            modalities=modalities,
            roi="LAP",
        )
        print(f"Cohort: {len(cohort)} subjects from demo_data ({modalities})")
        return cohort, modalities
    cohort = make_synthetic_cohort(n_subjects=3, shape=(14, 14, 14), rng=11)
    modalities = ("T1", "T2")
    print(f"Cohort: {len(cohort)} synthetic subjects ({modalities})")
    return cohort, modalities


def _atomic_units(spec: HabitatSpec, subject: object) -> int:
    """Run fit-time SubjectPipeline.units on one subject."""
    components = build_habitat_components(spec)
    pipeline = components.pipeline(assigner=None)
    units = pipeline.units(subject)
    frame = units.feature_frame()
    return int(frame.shape[1])


cohort, modalities = _load_cohort()
subject = cohort[0]
m0, m1 = modalities[0], modalities[1]

# --- raw (batch + atomic) ----------------------------------------------------
raw_spec = HabitatSpec(
    name="route_raw",
    voxel_feature_extractor=Spec("raw", {"modalities": list(modalities)}),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 5, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
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
    random_seed=11,
)
print("\n=== raw(modalities) ===")
print(f"  atomic n_features: {_atomic_units(raw_spec, subject)}")
raw_result = recipes.two_step(cohort, raw_spec)
print(f"  batch: {len(raw_result.habitat_maps)} maps, "
      f"{raw_result.habitat_model.n_habitats} habitats")

# Eye-check first route (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, raw_result)

# --- concat(raw(m0), raw(m1)) — heterogeneous join ---------------------------
concat_spec = HabitatSpec(
    name="route_concat",
    voxel_feature_extractor=Spec(
        "concat",
        {
            "extractors": [
                {"name": "raw", "params": {"modalities": [m0]}},
                {"name": "raw", "params": {"modalities": [m1]}},
            ],
        },
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 5, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
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
    random_seed=11,
)
print("\n=== concat(raw, raw) per modality ===")
print(f"  atomic n_features: {_atomic_units(concat_spec, subject)}")
concat_result = recipes.two_step(cohort, concat_spec)
print(f"  batch: {len(concat_result.habitat_maps)} maps")

# --- slic supervoxelizer (spatial coherence) ---------------------------------
slic_spec = HabitatSpec(
    name="route_slic",
    voxel_feature_extractor=Spec("raw", {"modalities": list(modalities)}),
    supervoxelizer=Spec("slic", {"n_supervoxels": 12, "compactness": 0.05}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
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
    random_seed=11,
)
print("\n=== supervoxelizer: slic ===")
try:
    print(f"  atomic n_features: {_atomic_units(slic_spec, subject)}")
    slic_result = recipes.two_step(cohort, slic_spec)
    print(f"  batch: {len(slic_result.habitat_maps)} maps, "
          f"{slic_result.habitat_model.n_habitats} habitats")
except Exception as exc:  # noqa: BLE001 - demo script reports and continues
    print(f"  slic skipped: {exc}")

# --- voxel_radiomics / supervoxel_radiomics (demo_data + PyRadiomics) ---------
if IMAGING_ROOT.is_dir() and PARAMS_VOXEL.is_file() and PARAMS_SV.is_file():
    print("\n=== voxel_radiomics (demo_data, may take ~30s) ===")
    vr_spec = HabitatSpec(
        name="route_voxel_radiomics",
        voxel_feature_extractor=Spec(
            "voxel_radiomics",
            {"modalities": [m0], "params_file": str(PARAMS_VOXEL)},
        ),
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
        habitat_model_fitter=Spec(
            "kmeans",
            {"min_habitats": 2, "max_habitats": 4, "validation": "elbow", "n_init": 3},
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
        random_seed=11,
    )
    single = cohort[0:1]
    try:
        n_feat = _atomic_units(vr_spec, subject)
        print(f"  atomic n_features: {n_feat}")
        vr_result = recipes.two_step(single, vr_spec)
        print(f"  batch (1 subject): {vr_result.habitat_model.n_habitats} habitats")
    except Exception as exc:  # noqa: BLE001
        print(f"  voxel_radiomics skipped: {exc}")

    print("\n=== supervoxel_radiomics (demo_data) ===")
    svr_spec = HabitatSpec(
        name="route_supervoxel_radiomics",
        voxel_feature_extractor=Spec("raw", {"modalities": [m0]}),
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
        supervoxel_feature_extractor=Spec(
            "supervoxel_radiomics",
            {"modalities": [m0], "params_file": str(PARAMS_SV)},
        ),
        habitat_model_fitter=Spec(
            "kmeans",
            {"min_habitats": 2, "max_habitats": 4, "validation": "elbow", "n_init": 3},
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
        random_seed=11,
    )
    try:
        n_feat = _atomic_units(svr_spec, subject)
        print(f"  atomic n_features: {n_feat}")
        svr_result = recipes.two_step(single, svr_spec)
        print(f"  batch (1 subject): {svr_result.habitat_model.n_habitats} habitats")
    except Exception as exc:  # noqa: BLE001
        print(f"  supervoxel_radiomics skipped: {exc}")
else:
    print("\n=== voxel_radiomics / supervoxel_radiomics ===")
    print("  skipped (need demo_data/preprocessed/ and PyRadiomics)")

print("\nYAML equivalents: config/habitat/config_habitat_two_step_voxel_radiomics_*.yaml")
