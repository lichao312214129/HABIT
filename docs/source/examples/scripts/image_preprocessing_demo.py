#!/usr/bin/env python
"""
Batch image preprocessing via ``habit.recipes.preprocess_images``.

Two entry points:

* **Batch (directory pipeline)** — ``preprocess_images(config)`` is the
  programmatic twin of ``habit preprocess``; scans ``data_dir`` and writes
  ``processed_images/``.
* **Atomic (in-memory)** — there is no separate subject-level image recipe
  yet; embed HABIT by building a one-subject directory layout or by
  operating on :class:`~habit.api.image.ImageVolume` objects upstream and
  passing the processed cohort to :func:`~habit.cohort_from_directory`.

This example writes a tiny synthetic cohort and runs **resample + z-score**
so it completes in seconds anywhere. When ``demo_data/`` is present, it
also shows the resample-only path used in ``demo_data/results/api/01_preprocess``.

This script accompanies ``docs/source/examples/image_preprocessing.rst``.

Run from the repository root::

    python docs/source/examples/scripts/image_preprocessing_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import SimpleITK as sitk

from habit.recipes import preprocess_images

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
IMAGING_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed" / "processed_images"


def _synthetic_dataset(data_root: Path, n_subjects: int = 3) -> List[str]:
    """Write a tiny HABIT layout under ``data_root`` and return modalities."""
    rng = np.random.default_rng(42)
    modalities = ["T1", "T2"]
    for index in range(n_subjects):
        subject_id = f"S{index:03d}"
        for modality in modalities:
            folder = data_root / "images" / subject_id / modality
            folder.mkdir(parents=True)
            array = rng.normal(loc=100.0, scale=20.0, size=(12, 12, 12)).astype(np.float32)
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((2.0, 2.0, 2.0))
            sitk.WriteImage(image, str(folder / f"{subject_id}_{modality}.nrrd"))
        mask_folder = data_root / "masks" / subject_id / "T1"
        mask_folder.mkdir(parents=True)
        mask = np.zeros((12, 12, 12), dtype=np.uint8)
        mask[3:9, 3:9, 3:9] = 1
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.SetSpacing((2.0, 2.0, 2.0))
        sitk.WriteImage(mask_image, str(mask_folder / f"{subject_id}_mask.nrrd"))
    return modalities


def _run_preprocess(
    data_dir: Path,
    out_dir: Path,
    modalities: List[str],
    *,
    resample_spacing: List[float],
    with_zscore: bool,
) -> None:
    """Execute preprocess_images with a resample (+ optional z-score) chain."""
    preprocessing: Dict[str, Any] = {
        "resample": {
            "images": modalities,
            "target_spacing": resample_spacing,
            "img_mode": "bilinear",
        },
    }
    if with_zscore:
        preprocessing["zscore_normalization"] = {
            "images": modalities,
            "mask_keyword": "masks",
            "use_mask": True,
        }
    config: Dict[str, Any] = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "auto_select_first_file": True,
        "processes": 1,
        "preprocessing": preprocessing,
    }
    result = preprocess_images(config)
    written = sorted(out_dir.rglob("*.nii.gz"))
    print(f"  output: {result.output_dir}")
    print(f"  manifest: {result.manifest_path}")
    print(f"  NIfTI files: {len(written)}")

print("=== Batch: synthetic cohort (resample + z-score) ===")
work_dir = Path(tempfile.mkdtemp(prefix="habit_preprocess_demo_"))
data_root = work_dir / "dataset"
out_dir = work_dir / "processed"
modalities = _synthetic_dataset(data_root)
print(f"Wrote synthetic cohort under {data_root}")
_run_preprocess(data_root, out_dir, modalities, resample_spacing=[1.0, 1.0, 1.0], with_zscore=True)

if IMAGING_ROOT.is_dir():
    print("\n=== Batch: demo_data DCE-MRI (resample only, mirrors API 01_preprocess) ===")
    demo_out = work_dir / "demo_resample"
    demo_modalities = ["delay2", "delay3", "delay5"]
    _run_preprocess(
        IMAGING_ROOT,
        demo_out,
        demo_modalities,
        resample_spacing=[3.0, 3.0, 3.0],
        with_zscore=False,
    )
else:
    print("\n(demo_data absent — skip real-data resample example)")

print("\nFull MRI pipeline (N4 + ANTs registration): "
      "config/preprocessing/config_preprocessing_n4_reg_resample_zscore.yaml")
print("Atomic note: pass processed images to cohort_from_directory; "
      "see habitat_preprocessing.rst for subject-level habitat chains.")
