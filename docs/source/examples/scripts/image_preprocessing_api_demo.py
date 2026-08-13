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

Change DATA / MODALITIES / ROI to your preprocessed tree. Accompanies
``docs/source/examples/image_preprocessing_api.rst``.

Run from the repository root::

    python docs/source/examples/scripts/image_preprocessing_api_demo.py
"""

from __future__ import annotations

# BEGIN example
import tempfile
from pathlib import Path
from typing import Any, Dict

from habit import cohort_from_directory, preprocess_image, preprocess_subject
from habit.recipes import preprocess_images
from habit.viz import plot_intensity_slice

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
modality = MODALITIES[0]

print("=== Atomic: preprocess_subject (in-memory) ===")
steps: Dict[str, Any] = {
    "resample": {"target_spacing": [2.0, 2.0, 2.0], "img_mode": "bilinear"},
}
processed = preprocess_subject(subject, steps)
vol = processed.image(modality)
print(
    f"  {subject.subject_id}: shape={vol.data.shape}, "
    f"spacing={tuple(round(float(v), 3) for v in vol.spacing)}"
)

print("=== Atomic: preprocess_image (single volume) ===")
single = preprocess_image(
    subject.image(modality),
    {"resample": {"target_spacing": [2.0, 2.0, 2.0], "img_mode": "nearest"}},
    mask=subject.mask(ROI),
    modality=modality,
)
print(f"  single volume spacing={tuple(round(float(v), 3) for v in single.spacing)}")

print("=== Batch: preprocess_images (directory pipeline) ===")
out_dir = Path(tempfile.mkdtemp(prefix="habit_preprocess_api_"))
config: Dict[str, Any] = {
    "data_dir": DATA,
    "out_dir": str(out_dir),
    "auto_select_first_file": True,
    "processes": 1,
    "preprocessing": {
        "resample": {
            "images": list(MODALITIES),
            "target_spacing": [3.0, 3.0, 3.0],
            "img_mode": "bilinear",
        },
    },
}
result = preprocess_images(config)
n_files = len(list(out_dir.rglob("*.nii.gz"))) + len(list(out_dir.rglob("*.nrrd")))
print(f"  output_dir={result.output_dir}")
print(f"  written files≈{n_files}")

# Z-score keeps the grid, so original | processed is an honest before/after.
# Resample (above) changes spacing/shape and cannot share one slice figure.
# Whole-FOV greyscale: do not pass roi_mask (that would imply an ROI crop).
zscored = preprocess_subject(
    subject, {"zscore_normalization": {"only_inmask": True, "mask_key": ROI}}
)
fig = plot_intensity_slice(
    zscored.image(modality),
    before=subject.image(modality),
    axis=0,
    cmap="gray",
    image_label=f"Z-scored {modality}",
    before_label=f"Original {modality}",
    title="Image preprocess: original | z-scored",
    colorbar_label="Z-score",
    before_colorbar_label="Intensity",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/image_preprocess_api_slice.png", dpi=150, bbox_inches="tight")
print("Wrote out/image_preprocess_api_slice.png")
# END example

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig, "image_preprocess_api_slice.png")
