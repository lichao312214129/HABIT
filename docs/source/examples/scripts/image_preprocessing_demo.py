#!/usr/bin/env python
"""
Batch image preprocessing via ``habit.recipes.preprocess_images``.

Two entry points:

* **Batch (directory pipeline)** — ``preprocess_images(config)`` is the
  programmatic twin of ``habit preprocess``; scans ``data_dir`` and writes
  ``processed_images/``.
* **Atomic (in-memory)** — :func:`~habit.preprocess_subject` on one
  :class:`~habit.contracts.Subject` (no YAML / directory layout).

Change DATA / MODALITIES / ROI to your preprocessed tree. This script
accompanies ``docs/source/examples/image_preprocessing.rst``.

Run from the repository root::

    python docs/source/examples/scripts/image_preprocessing_demo.py
"""

from __future__ import annotations

# BEGIN example
import tempfile
from pathlib import Path
from typing import Any, Dict

from habit import cohort_from_directory, preprocess_subject
from habit.recipes import preprocess_images

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

print("=== Batch: demo_data (resample + z-score) ===")
out_dir = Path(tempfile.mkdtemp(prefix="habit_preprocess_demo_"))
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
        "zscore_normalization": {
            "images": list(MODALITIES),
            "mask_keyword": "masks",
            "use_mask": True,
        },
    },
}
result = preprocess_images(config)
written = sorted(out_dir.rglob("*.nii.gz")) + sorted(out_dir.rglob("*.nrrd"))
print(f"  output: {result.output_dir}")
print(f"  manifest: {result.manifest_path}")
print(f"  image files: {len(written)}")

print("=== Atomic: z-score one subject ===")
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
modality = MODALITIES[0]
processed = preprocess_subject(
    subject, {"zscore_normalization": {"only_inmask": True, "mask_key": ROI}}
)
print(f"  {subject.subject_id} {modality}: shape={processed.image(modality).data.shape}")
# END example

# BEGIN figures
# Paste after the Script block. Uses subject, processed, and modality.
from habit.viz import plot_intensity_slice

# Whole-FOV greyscale: z-score is an intensity transform, not an ROI crop.
# Independent colorbars show raw intensity vs z-score (do not share clim).
fig = plot_intensity_slice(
    processed.image(modality),
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
fig.savefig("out/image_preprocess_slice.png", dpi=150, bbox_inches="tight")
print("Wrote out/image_preprocess_slice.png")
# END figures

print(
    "Full MRI pipeline (N4 + ANTs registration): "
    "config/preprocessing/config_preprocessing_n4_reg_resample_zscore.yaml"
)

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig, "image_preprocess_slice.png")
