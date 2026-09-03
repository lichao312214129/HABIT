#!/usr/bin/env python
"""
Atomic image-preprocessing steps: one copyable call per capability + figures.

Accompanies ``docs/source/examples/image_preprocessing_api.rst`` and
``docs/source/how_to/preprocess.rst``. Change DATA / MODALITIES / ROI to
your preprocessed tree.

Figures are whole-FOV greyscale anatomy (not ROI crops, not sequential
colormaps). Computation stays 3D; each PNG shows one teaching slice.

Run from the repository root::

    python docs/source/examples/scripts/image_preprocessing_steps_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.api.preprocessing import preprocess_image, preprocess_subject
from habit.viz import plot_intensity_slice, use_style

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
modality = MODALITIES[0]
mask = subject.mask(ROI).data if ROI in subject.masks else None
print(f"Subject {subject.subject_id} modality={modality}")
# END example

# BEGIN resample
resampled = preprocess_image(
    subject.image(modality),
    {"resample": {"target_spacing": [2.0, 2.0, 2.0], "img_mode": "bilinear"}},
    mask=subject.mask(ROI) if ROI in subject.masks else None,
    modality=modality,
)
print(f"resample: spacing {tuple(round(float(v), 3) for v in resampled.spacing)}")
# END resample

# BEGIN zscore
zscored = preprocess_subject(
    subject,
    {"zscore_normalization": {"only_inmask": True, "mask_key": ROI}},
)
roi_mean = float(np.mean(zscored.image(modality).data[mask > 0]))
print(f"zscore: mean (ROI stats)={roi_mean:.3f}")
# END zscore

# BEGIN histogram
hist = preprocess_subject(
    subject,
    {"histogram_standardization": {"target_min": 0.0, "target_max": 100.0}},
)
print(
    "histogram_standardization: "
    f"range=({float(hist.image(modality).data.min()):.1f}, "
    f"{float(hist.image(modality).data.max()):.1f})"
)
# END histogram

# BEGIN clahe
clahe = preprocess_subject(
    subject,
    {"adaptive_histogram_equalization": {"alpha": 0.3, "beta": 0.3, "radius": 5}},
)
print("adaptive_histogram_equalization: done")
# END clahe

# BEGIN n4
n4 = preprocess_subject(
    subject,
    {"n4_correction": {"num_fitting_levels": 2, "shrink_factor": 4}},
)
print("n4_correction: done")
# END n4

# BEGIN reorient
reoriented = preprocess_subject(
    subject,
    {"reorientation": {"target_orientation": "RAS", "mode": "closest"}},
)
print(
    "reorientation: direction[0:3]="
    f"{tuple(round(float(v), 3) for v in reoriented.image(modality).direction[:3])}"
)
# END reorient

# BEGIN registration
# SimpleITK affine (no ANTs extra). Fixed = first modality; others move.
# The same transform is applied to the mask (contour overlay on the figure).
reg_ok = False
try:
    registered = preprocess_subject(
        subject,
        {
            "registration": {
                "fixed_image": modality,
                "backend": "simpleitk",
                "type_of_transform": "Affine",
            }
        },
    )
    print("registration: SimpleITK Affine done")
    reg_ok = True
except Exception as exc:
    registered = subject
    print(f"registration skipped: {exc}")
# END registration


def _as_array(volume) -> np.ndarray:
    """Return the NumPy array from an ImageVolume or a raw array."""
    return np.asarray(volume.data if hasattr(volume, "data") else volume)


def _save_slice(
    processed,
    filename: str,
    label: str,
    *,
    before=None,
    roi_mask=None,
    roi_contour: bool = False,
    colorbar_label: str = "Intensity",
    before_colorbar_label: str = "Intensity",
) -> None:
    """
    Write a whole-FOV greyscale teaching slice.

    Same-grid steps (N4, z-score, histogram, CLAHE) show original | processed
    with an independent colorbar per panel (native units, not a shared [0, 1]
    window). Resample / reorient often change the grid: those figures are
    processed-only. Registration may overlay the transformed ROI as a contour.
    """
    from _example_roi import save_example_figure

    proc_vol = processed.image(modality) if hasattr(processed, "image") else processed
    before_vol = None
    if before is not None:
        before_vol = before.image(modality) if hasattr(before, "image") else before
        if _as_array(before_vol).shape != _as_array(proc_vol).shape:
            # Grid changed (resample / reorient): do not force a mismatched pair.
            before_vol = None
    contour_mask = None
    if roi_contour and roi_mask is not None:
        contour_mask = roi_mask
        if _as_array(contour_mask).shape != _as_array(proc_vol).shape:
            contour_mask = None
            roi_contour = False
    title = f"Original | {label}" if before_vol is not None else None
    fig = plot_intensity_slice(
        proc_vol,
        before=before_vol,
        roi_mask=contour_mask,
        axis=0,
        cmap="gray",
        image_label=label,
        before_label="Original",
        title=title,
        roi_contour=bool(roi_contour and contour_mask is not None),
        colorbar_label=colorbar_label,
        before_colorbar_label=before_colorbar_label,
    )
    save_example_figure(fig, filename)
    print(f"Wrote {filename}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    with use_style("radiology"):
        # Whole-image geometry / intensity: no ROI on the figure.
        _save_slice(resampled, "preprocess_resample.png", f"Resampled {modality}")
        _save_slice(
            zscored,
            "preprocess_zscore.png",
            f"Z-scored {modality}",
            before=subject,
            colorbar_label="Z-score",
        )
        _save_slice(
            hist,
            "preprocess_histogram.png",
            f"Histogram-standardized {modality}",
            before=subject,
        )
        _save_slice(
            clahe,
            "preprocess_clahe.png",
            "Adaptive histogram equalization",
            before=subject,
        )
        _save_slice(
            n4,
            "preprocess_n4.png",
            f"N4 bias-field corrected {modality}",
            before=subject,
        )
        _save_slice(reoriented, "preprocess_reorient.png", "Reoriented to RAS")
        # Registration: ROI contour on anatomy (mask follows the transform).
        reg_mask = registered.mask(ROI) if ROI in registered.masks else None
        if reg_ok:
            _save_slice(
                registered,
                "preprocess_register.png",
                "Registered (SimpleITK Affine)",
                before=subject,
                roi_mask=reg_mask,
                roi_contour=True,
            )
        else:
            _save_slice(
                subject,
                "preprocess_register.png",
                "Registration (input; step skipped)",
                roi_mask=mask,
                roi_contour=True,
            )
