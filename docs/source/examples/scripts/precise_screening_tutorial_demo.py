#!/usr/bin/env python
"""
Precise-screening perturbation figures on real preprocessed anatomy.

Accompanies ``docs/source/tutorial/precise_screening.rst`` and
``docs/source/examples/precise_features.rst``.

Each Prior 2024 / MIRP 1.2.0 component is applied with the public
``image_perturbation`` API. The ROI-contour figure is a separate optional
step: MONAI ``Rand3DElastic`` free-form / elastic warp
(``BSplineDeformPerturbation``), not MIRP morphological grow/shrink and
not the Prior 2024 default chain.

Run from the repository root::

    python docs/source/examples/scripts/precise_screening_tutorial_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from habit import cohort_from_directory
from habit.contracts import ArrayImageRef, Geometry
from habit.domain import (
    BSplineDeformPerturbation,
    GaussianNoisePerturbation,
    RotationPerturbation,
    TranslationPerturbation,
    prior2024_retest_perturbation,
)
from habit.kernels import local_entropy_map


def _gallery_dir() -> Path:
    """Return ``docs/source/_static/images/examples`` (created if needed)."""
    out = Path(__file__).resolve().parents[2] / "_static" / "images" / "examples"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _densest_roi_index(mask_volume: np.ndarray, axis: int = 0) -> int:
    """
    Return the slice index with the most ROI voxels along ``axis``.

    Args:
        mask_volume: Label / binary mask, NumPy ``(z, y, x)``.
        axis: Axis along which to reduce.

    Returns:
        Slice index in ``[0, length)``.
    """
    other = tuple(i for i in range(mask_volume.ndim) if i != axis)
    counts = np.sum(np.asarray(mask_volume) > 0, axis=other)
    if int(np.max(counts)) == 0:
        return int(mask_volume.shape[axis] // 2)
    return int(np.argmax(counts))


def _roi_crop_slices(
    mask_volume: np.ndarray,
    *,
    pad: int = 12,
) -> tuple[slice, ...]:
    """
    Padded bounding-box slices around ``mask_volume > 0``.

    Display-only crop so a one-voxel contour shift is visible. HABIT still
    transforms the full grid.

    Args:
        mask_volume: Foreground indicator / label volume.
        pad: Voxel padding on each side.

    Returns:
        Tuple of slices for NumPy indexing.
    """
    foreground = np.asarray(mask_volume) > 0
    if not np.any(foreground):
        raise RuntimeError("ROI crop: mask has no foreground voxels.")
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        start = max(0, int(lo) - pad)
        stop = min(int(mask_volume.shape[axis]), int(hi) + pad + 1)
        slices.append(slice(start, stop))
    return tuple(slices)


def _window_grey(panel: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale a 2D slice to ``[0, 1]`` with a shared intensity window."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(panel.shape, dtype=np.float32)
    scaled = (np.asarray(panel, dtype=np.float64) - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _orient(
    slice_2d: np.ndarray,
    *,
    direction,
    axis: int = 0,
) -> np.ndarray:
    """Apply HABIT's radiological display convention to one slice."""
    from habit.viz.orientation import (
        direction_matrix,
        orient_slice_for_display,
    )

    matrix = direction_matrix(direction, ndim=3) if direction is not None else None
    return orient_slice_for_display(
        np.asarray(slice_2d),
        slice_axis=axis,
        direction=matrix,
        convention="radiological",
    )


def _save_figure(fig, filename: str) -> Path:
    """
    Write ``fig`` to the gallery directory and ``out/``.

    Args:
        fig: Matplotlib figure.
        filename: Basename only.

    Returns:
        Gallery path written.
    """
    import matplotlib.pyplot as plt

    Path("out").mkdir(exist_ok=True)
    dest = _gallery_dir() / filename
    fig.savefig(dest, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(Path("out") / filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {filename}")
    return dest


def _perturbation_triptych(
    original: np.ndarray,
    perturbed: np.ndarray,
    *,
    index: int,
    direction,
    spacing,
    before_label: str,
    after_label: str,
    title: str,
    filename: str,
) -> None:
    """
    Original | perturbed | |difference| on one axial slice (shared window).

    Args:
        original: Intensity volume ``(z, y, x)``.
        perturbed: Same-grid perturbed intensity volume.
        index: Axial slice index.
        direction: SimpleITK direction cosines (9 floats) or ``None``.
        spacing: SimpleITK spacing ``(x, y, z)`` in mm.
        before_label: Left panel title.
        after_label: Middle panel title.
        title: Figure title (English, ASCII).
        filename: Gallery basename.
    """
    import matplotlib.pyplot as plt

    from habit.viz import use_style
    from habit.viz.labels import sanitize_label
    from habit.viz.orientation import imshow_physical_extent

    orig_sl = _orient(np.take(original, index, axis=0), direction=direction)
    pert_sl = _orient(np.take(perturbed, index, axis=0), direction=direction)
    finite = np.concatenate(
        [orig_sl[np.isfinite(orig_sl)].ravel(), pert_sl[np.isfinite(pert_sl)].ravel()]
    )
    lo, hi = np.percentile(finite, (1.0, 99.0))
    diff_sl = np.abs(pert_sl.astype(np.float64) - orig_sl.astype(np.float64))
    diff_finite = diff_sl[np.isfinite(diff_sl)]
    d_hi = float(np.percentile(diff_finite, 99.5)) if diff_finite.size else 1.0
    if d_hi <= 0.0:
        d_hi = 1.0
    extent = imshow_physical_extent(
        (int(orig_sl.shape[0]), int(orig_sl.shape[1])),
        spacing,
        slice_axis=0,
        ndim=3,
    )
    with use_style("radiology"):
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(11.4, 3.9),
            constrained_layout=True,
            facecolor="white",
        )
        panels = (
            (axes[0], _window_grey(orig_sl, lo, hi), before_label, 1.0),
            (axes[1], _window_grey(pert_sl, lo, hi), after_label, 1.0),
            (axes[2], np.clip(diff_sl / d_hi, 0.0, 1.0), "Absolute difference", 1.0),
        )
        for ax, data, label, vmax in panels:
            ax.imshow(
                data,
                cmap="gray",
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(sanitize_label(label))
            ax.axis("off")
            ax.set_facecolor("white")
        fig.suptitle(sanitize_label(title))
        fig.patch.set_facecolor("white")
        _save_figure(fig, filename)


def _mask_edge_figure(
    anatomy_vol: np.ndarray,
    original_mask: np.ndarray,
    deformed_mask: np.ndarray,
    *,
    direction,
    spacing,
    filename: str,
) -> None:
    """
    ROI-cropped anatomy with original vs MONAI elastic / B-spline FFD contours.

    This is ``BSplineDeformPerturbation`` (MONAI ``Rand3DElastic``), not
    nearest-neighbour rigid resampling and not MIRP ROI grow/shrink.

    Args:
        anatomy_vol: Greyscale anatomy ``(z, y, x)``.
        original_mask: Unperturbed ROI.
        deformed_mask: ROI after the MONAI elastic / B-spline warp.
        direction: SimpleITK direction cosines.
        spacing: SimpleITK spacing ``(x, y, z)``.
        filename: Gallery basename.
    """
    from habit.viz import use_style
    from habit.viz.labels import sanitize_label
    from habit.viz.orientation import imshow_physical_extent

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    union = (original_mask > 0) | (deformed_mask > 0)
    crop = _roi_crop_slices(union.astype(np.uint8), pad=16)
    anatomy_c = anatomy_vol[crop]
    orig_c = (original_mask[crop] > 0).astype(np.uint8)
    def_c = (deformed_mask[crop] > 0).astype(np.uint8)
    index = _densest_roi_index(orig_c, axis=0)

    orig_sl = _orient(np.take(anatomy_c, index, axis=0), direction=direction)
    mask_orig = _orient(np.take(orig_c, index, axis=0), direction=direction)
    mask_def = _orient(np.take(def_c, index, axis=0), direction=direction)
    xor_def = np.abs(mask_def.astype(np.float64) - mask_orig.astype(np.float64))

    finite = orig_sl[np.isfinite(orig_sl)]
    lo, hi = np.percentile(finite, (1.0, 99.0))
    grey = _window_grey(orig_sl, lo, hi)
    extent = imshow_physical_extent(
        (int(grey.shape[0]), int(grey.shape[1])),
        spacing,
        slice_axis=0,
        ndim=3,
    )
    original_color = "#00E5FF"
    perturbed_color = "#D55E00"
    xor_color = "#F0E442"

    def _draw_panel(
        ax,
        xor_map: np.ndarray,
        perturbed_mask_slice: np.ndarray,
        panel_title: str,
    ) -> None:
        ax.imshow(
            grey,
            cmap="gray",
            interpolation="nearest",
            origin="upper",
            extent=extent,
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
        )
        if np.any(xor_map > 0):
            ax.contourf(
                xor_map,
                levels=[0.5, 1.5],
                colors=[xor_color],
                alpha=0.55,
                origin="upper",
                extent=extent,
            )
        ax.contour(
            mask_orig,
            levels=[0.5],
            colors=[original_color],
            linewidths=1.6,
            origin="upper",
            extent=extent,
        )
        ax.contour(
            perturbed_mask_slice,
            levels=[0.5],
            colors=[perturbed_color],
            linewidths=1.6,
            linestyles="--",
            origin="upper",
            extent=extent,
        )
        ax.set_title(sanitize_label(panel_title))
        ax.axis("off")
        ax.set_facecolor("white")

    with use_style("radiology"):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 4.4),
            constrained_layout=True,
            facecolor="white",
        )
        _draw_panel(
            axes[0],
            np.zeros_like(xor_def),
            mask_def,
            "MONAI B-spline FFD (contours)",
        )
        _draw_panel(
            axes[1],
            xor_def,
            mask_def,
            "Voxels that changed membership",
        )
        handles = [
            Line2D([0], [0], color=original_color, lw=2.0, label="Original ROI"),
            Line2D(
                [0],
                [0],
                color=perturbed_color,
                lw=2.0,
                linestyle="--",
                label="Perturbed ROI (MONAI Rand3DElastic)",
            ),
            Patch(facecolor=xor_color, alpha=0.55, label="Voxels that changed"),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=False,
        )
        fig.set_constrained_layout_pads(h_pad=0.12, w_pad=0.04)
        fig.suptitle(
            sanitize_label(
                "Mask edge: MONAI elastic / B-spline FFD (not Prior 2024)"
            )
        )
        fig.patch.set_facecolor("white")
        _save_figure(fig, filename)

    n_changed = int(np.sum((deformed_mask > 0) != (original_mask > 0)))
    print(f"  mask voxels changed (MONAI FFD): {n_changed}")


if __name__ == "__main__":
    from habit.viz import plot_voxel_texture_slice, use_style

    # BEGIN example
    # Change DATA / MODALITIES / ROI to your preprocessed layout
    DATA = "demo_data/preprocessed"
    MODALITIES = ("LAP",)
    ROI = "LAP"

    cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
    subject = cohort[0]
    modality = MODALITIES[0]
    anatomy = subject.image(modality).data
    mask = subject.mask(ROI).data
    rng = np.random.default_rng(7)

    noisy = GaussianNoisePerturbation()(subject, rng=rng)
    shifted = TranslationPerturbation(
        shift_voxels=(0.5, 0.5, 0.0),
        random_signs=False,
    )(subject, rng=rng)
    rotated = RotationPerturbation(angle_degrees=0.5, axis="z")(subject, rng=rng)
    retest = prior2024_retest_perturbation()(subject, rng=np.random.default_rng(7))

    # Optional follow-up (not Prior 2024): MONAI elastic / B-spline FFD.
    # The public API warps the full Subject; this LAP volume is 200x360x360,
    # so the example crops to the ROI bbox + pad (still real demo_data).
    _nz = np.argwhere(np.asarray(mask) > 0)
    _lo = np.maximum(_nz.min(axis=0) - 24, 0)
    _hi = np.minimum(_nz.max(axis=0) + 25, np.asarray(mask).shape)
    _sl = tuple(slice(int(a), int(b)) for a, b in zip(_lo, _hi))
    _src_geom = subject.image(modality).geometry
    _crop_img = np.asarray(anatomy)[_sl]
    _geom = Geometry.from_array(
        _crop_img.shape,
        spacing=_src_geom.spacing,
        direction=_src_geom.direction,
    )
    ffd_subject = type(subject)(
        subject_id=subject.subject_id,
        images={modality: ArrayImageRef(array=_crop_img, geometry=_geom)},
        masks={
            ROI: ArrayImageRef(array=np.asarray(mask)[_sl], geometry=_geom)
        },
    )
    print("Applying MONAI Rand3DElastic to the ROI crop...", flush=True)
    deformed = BSplineDeformPerturbation(
        sigma_range=(1.0, 2.0),
        magnitude_range=(10.0, 15.0),
        device="cpu",
    )(ffd_subject, rng=np.random.default_rng(7))
    print("FFD done.", flush=True)

    entropy_small = local_entropy_map(anatomy, kernel_size=3, bins=16)
    entropy_large = local_entropy_map(anatomy, kernel_size=7, bins=16)
    # END example

    volume = subject.image(modality)
    direction = volume.direction
    spacing = volume.spacing
    slice_index = _densest_roi_index(np.asarray(mask), axis=0)
    subject_id = getattr(subject, "subject_id", "demo")
    print(f"Subject {subject_id}; axial slice {slice_index}")

    _perturbation_triptych(
        np.asarray(anatomy, dtype=np.float64),
        np.asarray(noisy.image(modality).data, dtype=np.float64),
        index=slice_index,
        direction=direction,
        spacing=spacing,
        before_label="Original",
        after_label="Gaussian noise (Chang sigma)",
        title=f"Intensity noise (demo subject {subject_id})",
        filename="precise_perturb_noise.png",
    )
    _perturbation_triptych(
        np.asarray(anatomy, dtype=np.float64),
        np.asarray(shifted.image(modality).data, dtype=np.float64),
        index=slice_index,
        direction=direction,
        spacing=spacing,
        before_label="Original",
        after_label="Translation 0.5 voxel",
        title=f"Sub-voxel translation (demo subject {subject_id})",
        filename="precise_perturb_translation.png",
    )
    _perturbation_triptych(
        np.asarray(anatomy, dtype=np.float64),
        np.asarray(rotated.image(modality).data, dtype=np.float64),
        index=slice_index,
        direction=direction,
        spacing=spacing,
        before_label="Original",
        after_label="Rotation 0.5 deg (z)",
        title=f"In-plane rotation (demo subject {subject_id})",
        filename="precise_perturb_rotation.png",
    )
    _perturbation_triptych(
        np.asarray(anatomy, dtype=np.float64),
        np.asarray(retest.image(modality).data, dtype=np.float64),
        index=slice_index,
        direction=direction,
        spacing=spacing,
        before_label="Original",
        after_label="Noise + translation + rotation",
        title=f"Prior 2024 simulated retest (demo subject {subject_id})",
        filename="precise_screen_perturbation.png",
    )
    _mask_edge_figure(
        np.asarray(ffd_subject.image(modality).data, dtype=np.float64),
        np.asarray(ffd_subject.mask(ROI).data),
        np.asarray(deformed.mask(ROI).data),
        direction=direction,
        spacing=spacing,
        filename="precise_perturb_mask_edge.png",
    )

    with use_style("radiology"):
        fig_k = plot_voxel_texture_slice(
            entropy_large,
            anatomy=entropy_small,
            roi_mask=mask,
            axis=0,
            index=slice_index,
            feature_label="Entropy, kernel=7",
            title="Local entropy: kernel 3 (left) vs 7 (right)",
            direction=direction,
            spacing=spacing,
        )
        _save_figure(fig_k, "precise_screen_kernel_scale.png")
