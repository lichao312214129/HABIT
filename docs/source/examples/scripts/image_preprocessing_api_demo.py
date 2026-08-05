#!/usr/bin/env python
"""
Image preprocessing API: batch directory pipeline + atomic in-memory operators.

* **Batch** — :func:`habit.recipes.preprocess_images` (CLI twin of
  ``habit preprocess``).
* **Atomic subject** — :func:`habit.preprocess_subject` / recipe twin
  ``recipes.preprocess_subject``: one :class:`~habit.contracts.Subject` in,
  one Subject out. No ``data_dir`` / ``out_dir`` / YAML.
* **Atomic volume** — :func:`habit.preprocess_image` for a single
  :class:`~habit.api.image.ImageVolume`.

Accompanies ``docs/source/examples/image_preprocessing_api.rst``.

Run from the repository root::

    python docs/source/examples/scripts/image_preprocessing_api_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import SimpleITK as sitk

from habit import preprocess_image, preprocess_subject
from habit.datasets import make_synthetic_cohort
from habit.recipes import preprocess_images

REPO_ROOT: Path = Path(__file__).resolve().parents[4]


def _write_tiny_dataset(root: Path, n_subjects: int = 2) -> List[str]:
    """Write a minimal HABIT images/masks layout for the batch path."""
    rng = np.random.default_rng(0)
    modalities = ["T1", "T2"]
    for index in range(n_subjects):
        subject_id = f"S{index:03d}"
        for modality in modalities:
            folder = root / "images" / subject_id / modality
            folder.mkdir(parents=True)
            array = rng.normal(100.0, 15.0, size=(10, 10, 10)).astype(np.float32)
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((2.0, 2.0, 2.0))
            sitk.WriteImage(image, str(folder / f"{modality}.nrrd"))
        mask_folder = root / "masks" / subject_id / "tumor"
        mask_folder.mkdir(parents=True)
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.SetSpacing((2.0, 2.0, 2.0))
        sitk.WriteImage(mask_image, str(mask_folder / "mask.nrrd"))
    return modalities


print("=== Atomic: preprocess_subject (in-memory) ===")
cohort = make_synthetic_cohort(n_subjects=2, modalities=("T1", "T2"), rng=7)
subject = cohort[0]
steps: Dict[str, Any] = {
    "resample": {"target_spacing": [2.0, 2.0, 2.0], "img_mode": "bilinear"},
}
processed = preprocess_subject(subject, steps)
vol = processed.image("T1")
print(f"  {subject.subject_id}: shape={vol.data.shape}, "
      f"spacing={tuple(round(float(v), 3) for v in vol.spacing)}")

print("=== Atomic: preprocess_image (single volume) ===")
single = preprocess_image(
    subject.image("T1"),
    {"resample": {"target_spacing": [2.0, 2.0, 2.0], "img_mode": "nearest"}},
    mask=subject.mask("tumor"),
    modality="T1",
)
print(f"  single volume spacing={tuple(round(float(v), 3) for v in single.spacing)}")

print("=== Batch: preprocess_images (directory pipeline) ===")
work = Path(tempfile.mkdtemp(prefix="habit_preprocess_api_"))
modalities = _write_tiny_dataset(work / "dataset")
out_dir = work / "processed"
config: Dict[str, Any] = {
    "data_dir": str(work / "dataset"),
    "out_dir": str(out_dir),
    "auto_select_first_file": True,
    "processes": 1,
    "preprocessing": {
        "resample": {
            "images": modalities,
            "target_spacing": [1.0, 1.0, 1.0],
            "img_mode": "bilinear",
        },
    },
}
result = preprocess_images(config)
n_files = len(list(out_dir.rglob("*.nii.gz"))) + len(list(out_dir.rglob("*.nrrd")))
print(f"  output_dir={result.output_dir}")
print(f"  written files≈{n_files}")
