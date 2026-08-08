#!/usr/bin/env python
"""
Habitat feature extraction after a two-step training run.

Demonstrates:

* **Batch** — ``extract_habitat_features(config)`` over a directory of maps.
* **Train path** — ``recipes.two_step`` with preprocessing chains (aligned
  with ``demo_data/results/api/05_extract_features`` when demo_data exists).

Feature families: ``traditional``, ``non_radiomics``, ``whole_habitat``,
``each_habitat``, ``msi``, ``ith_score``.

This script accompanies ``docs/source/examples/feature_extraction.rst``.

Run from the repository root (PyRadiomics required for ``traditional``)::

    python docs/source/examples/scripts/feature_extraction_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from habit import HabitatSpec, Spec, cohort_from_directory, make_synthetic_cohort
import habit.recipes as recipes

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
IMAGING_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed"


def _load_cohort() -> Tuple[object, Sequence[str]]:
    """Load demo or synthetic cohort."""
    if IMAGING_ROOT.is_dir():
        modalities = ("pre_contrast", "LAP", "PVP", "delay_3min")
        return (
            cohort_from_directory(IMAGING_ROOT, modalities=modalities, roi="LAP"),
            modalities,
        )
    return make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=11), ("T1", "T2")


def main() -> None:
    """Run two-step training then batch feature extraction."""
    cohort, modalities = _load_cohort()
    print(f"Cohort: {len(cohort)} subjects, modalities={list(modalities)}")

    # Keyword order follows the runtime pipeline (not HabitatSpec field order).
    spec = HabitatSpec(
        name="extract_demo",
        voxel_feature_extractor=Spec("raw", {"modalities": list(modalities)}),
        voxel_feature_preprocessors=(
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
            Spec("minmax", {"across_features": False}),
        ),
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
        cohort_feature_preprocessors=(
            Spec("binning", {"n_bins": 6, "bin_strategy": "uniform", "across_features": False}),
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

    train_result = recipes.two_step(cohort, spec)
    print(f"Trained: {train_result.habitat_model.n_habitats} habitats, "
          f"{len(train_result.habitat_maps)} maps")

    # Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _habitat_eye_check import eye_check_study
    eye_check_study(cohort, train_result)

    work_dir = Path(tempfile.mkdtemp(prefix="habit_extract_demo_"))
    maps_dir = work_dir / "habitat_maps"
    images_dir = work_dir / "images"
    out_dir = work_dir / "features"

    train_result.save(maps_dir, write_maps=True, write_units_table=True)
    print(f"Saved habitat maps to {maps_dir}")

    # Images folder for radiomics-backed families.
    if IMAGING_ROOT.is_dir():
        raw_img_folder = str(IMAGING_ROOT)
    else:
        for subject in cohort:
            for modality, image_ref in subject.images.items():
                folder = images_dir / "images" / subject.subject_id / modality
                folder.mkdir(parents=True, exist_ok=True)
                array = image_ref.load().astype(np.float32)
                image = sitk.GetImageFromArray(array)
                image.SetSpacing(tuple(float(v) for v in image_ref.geometry.spacing))
                sitk.WriteImage(image, str(folder / f"{subject.subject_id}_{modality}.nrrd"))
            for roi, mask_ref in subject.masks.items():
                folder = images_dir / "masks" / subject.subject_id / roi
                folder.mkdir(parents=True, exist_ok=True)
                mask_array = mask_ref.load().astype(np.uint8)
                mask_image = sitk.GetImageFromArray(mask_array)
                mask_image.SetSpacing(tuple(float(v) for v in mask_ref.geometry.spacing))
                sitk.WriteImage(mask_image, str(folder / f"{subject.subject_id}_mask.nrrd"))
        raw_img_folder = str(images_dir)

    # Light families on by default; heavy radiomics stay commented / opt-in.
    feature_types: List[str] = [
        "volume",
        "msi",
        "ith_score",
        "non_radiomics",
        # "traditional",
        # "whole_habitat",
        # "each_habitat",
    ]
    # Uncomment to also run traditional radiomics when a real imaging root exists:
    # if IMAGING_ROOT.is_dir():
    #     feature_types.insert(0, "traditional")

    config: Dict[str, Any] = {
        "raw_img_folder": raw_img_folder,
        "habitats_map_folder": str(maps_dir),
        "out_dir": str(out_dir),
        "n_processes": 1,
        "habitat_pattern": "*_habitats.nrrd",
        "feature_types": feature_types,
        "n_habitats": train_result.habitat_model.n_habitats,
    }

    print(f"\nExtracting feature families: {feature_types}")
    extract_result = recipes.extract_habitat_features(config)
    print(f"Output: {extract_result.output_dir}")
    csv_outputs = sorted(out_dir.glob("*.csv"))
    print(f"CSV artefacts: {len(csv_outputs)}")
    for path in csv_outputs[:6]:
        print(f"  {path.name}")
    if len(csv_outputs) > 6:
        print(f"  ... and {len(csv_outputs) - 6} more")

    print("\nStandalone ROI radiomics: traditional_radiomics_demo.py")


if __name__ == "__main__":
    main()
