#!/usr/bin/env python
"""
Optional extra perturbation catalogue (not the taught gallery).

Gallery figures for precise features now come from
``precise_features_demo.py`` (``perturb_image`` / ``extract_voxel_texture``
atoms). This script is kept as a private catalogue of Subject-level
perturbation components; do not copy its PNGs into the Sphinx pages.

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

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.contracts import ArrayImageRef, Geometry
from habit.precision import BSplineDeformPerturbation, GaussianNoisePerturbation, RigidPerturbation, RotationPerturbation, TranslationPerturbation, prior2024_retest_perturbation
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
    index: int | None = None,
) -> int:
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
        index: Slice index in the cropped volume. ``None`` selects the
            densest original-ROI slice (same rule as the habitat compare).

    Returns:
        Slice index used in the cropped volume.
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
    if index is None:
        index = _densest_roi_index(orig_c, axis=0)
    else:
        index = int(index)

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
    return index


if __name__ == "__main__":
    from habit.viz import plot_voxel_texture_slice, use_style

    # BEGIN example
    # fetch_demo() downloads once and prints the tree. Change DATA for your data.
    DATA = fetch_demo()
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
    rigid = RigidPerturbation(
        shift_voxels=(0.5, 0.5, 0.0),
        angle_degrees=0.5,
        axis="z",
        random_signs=False,
        random_sign=False,
    )(subject, rng=rng)
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
        target_dice=0.85,
        dice_tolerance=0.03,
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
        np.asarray(rigid.image(modality).data, dtype=np.float64),
        index=slice_index,
        direction=direction,
        spacing=spacing,
        before_label="Original",
        after_label="Rigid (one resample)",
        title=f"Rigid translation+rotation (demo subject {subject_id})",
        filename="precise_perturb_rigid.png",
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
    _ffd_mask = np.asarray(ffd_subject.mask(ROI).data)
    _def_mask = np.asarray(deformed.mask(ROI).data)
    _display_union = (_ffd_mask > 0) | (_def_mask > 0)
    _display_crop = _roi_crop_slices(_display_union.astype(np.uint8), pad=16)
    display_slice = _densest_roi_index(_ffd_mask[_display_crop], axis=0)
    _mask_edge_figure(
        np.asarray(ffd_subject.image(modality).data, dtype=np.float64),
        _ffd_mask,
        _def_mask,
        direction=direction,
        spacing=spacing,
        filename="precise_perturb_mask_edge.png",
        index=display_slice,
    )

    with use_style("radiology"):
        fig_k = plot_voxel_texture_slice(
            entropy_large,
            anatomy=subject.image(modality),
            roi_mask=subject.mask(ROI),
            axis=0,
            index=slice_index,
            feature_label="Entropy, kernel=7",
            title="Local entropy (kernel=7) on anatomy",
        )
        _save_figure(fig_k, "precise_screen_kernel_scale.png")

    # BEGIN stability
    # Paste after the Script block. Uses ffd_subject, deformed, modality,
    # ROI, MODALITIES, direction, spacing. Same crop + slice as the ROI
    # contour figure (densest original ROI after the union bbox crop).
    from habit.contracts import Cohort
    from habit.precision import align_habitat_map, habitat_stability
    from habit.recipes import one_step_habitat
    from habit.viz import plot_habitat_label_compare, use_style

    import matplotlib.pyplot as plt

    Path("out").mkdir(exist_ok=True)

    orig_mask = np.asarray(ffd_subject.mask(ROI).data)
    warped_mask = np.asarray(deformed.mask(ROI).data)
    display_union = (orig_mask > 0) | (warped_mask > 0)
    display_crop = _roi_crop_slices(display_union.astype(np.uint8), pad=16)
    display_slice = _densest_roi_index(orig_mask[display_crop], axis=0)

    crop_cohort = Cohort(subjects=(ffd_subject,))
    warped_cohort = Cohort(subjects=(deformed,))
    orig_habitats = one_step_habitat(
        modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
    ).fit_predict(crop_cohort)
    warped_habitats = one_step_habitat(
        modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
    ).fit_predict(warped_cohort)
    ref_map = orig_habitats.habitat_maps[0]
    mov_map = warped_habitats.habitat_maps[0]
    # Independent one_step fits permute integer ids. Remap the warped map
    # (panel 2 only) onto the reference by mean-intensity Hungarian
    # pairing -- the same method="centroid" habitat_stability uses -- so
    # the same colour is the same intensity-defined habitat. force=True:
    # both fits share a model_id digest. Dice is scored on the original
    # pair after that same pairing.
    orig_image = ffd_subject.image(modality)
    warp_image = deformed.image(modality)
    aligned_map = align_habitat_map(
        ref_map,
        mov_map,
        method="centroid",
        image=orig_image,
        moving_image=warp_image,
        force=True,
    )
    dice_frame = habitat_stability(
        ref_map,
        [mov_map],
        method="centroid",
        image=orig_image,
        moving_images=(warp_image,),
    )
    print("Habitat Dice after BSplineDeform (Hungarian match)")
    print(dice_frame.to_string(index=False))
    print(f"Shared display slice (cropped original ROI): {display_slice}")

    img_c = np.asarray(ffd_subject.image(modality).data)[display_crop]
    ref_c = np.asarray(ref_map.label_array)[display_crop]
    aln_c = np.asarray(aligned_map.label_array)[display_crop]
    fig_cmp = plot_habitat_label_compare(
        img_c,
        ref_c,
        aln_c,
        titles=("Original ROI habitats", "Warped ROI habitats"),
        index=display_slice,
        direction=direction,
        spacing=spacing,
        align_labels=True,
    )
    fig_cmp.savefig(
        "out/precise_habitat_stability_compare.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig_cmp)

    with use_style("radiology"):
        fig_dice, ax_d = plt.subplots(figsize=(5.4, 3.2))
        habitat_ids = dice_frame["habitat_id"].to_numpy()
        ax_d.bar(
            [f"H{int(h)}" for h in habitat_ids],
            dice_frame["dice"].to_numpy(dtype=float),
            color="#0072B2",
        )
        ax_d.set_ylim(0.0, 1.05)
        ax_d.set_ylabel("Dice")
        ax_d.set_title("Per-habitat Dice (matched)")
        fig_dice.tight_layout()
        fig_dice.savefig(
            "out/precise_habitat_dice.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig_dice)
    print(
        "Wrote out/precise_habitat_stability_compare.png, "
        "out/precise_habitat_dice.png"
    )
    # END stability

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        (
            "precise_habitat_stability_compare.png",
            "precise_habitat_dice.png",
        )
    )
