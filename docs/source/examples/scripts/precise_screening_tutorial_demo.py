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
            anatomy=subject.image(modality),
            roi_mask=subject.mask(ROI),
            axis=0,
            index=slice_index,
            feature_label="Entropy, kernel=7",
            title="Local entropy (kernel=7) on anatomy",
        )
        _save_figure(fig_k, "precise_screen_kernel_scale.png")

    # BEGIN stability
    # Paste after the Script block. Uses subject, retest, ffd_subject, deformed,
    # anatomy, mask, entropy_large, modality, ROI, MODALITIES, direction, spacing.
    from habit import (
        Cohort,
        align_habitat_map,
        extract_graph_features,
        habitat_stability,
        habitat_volume_fractions,
        ith_score,
        one_step_habitat,
        spatial_interaction_matrix,
    )
    from habit.viz import plot_habitat_label_compare
    from habit.viz.orientation import imshow_physical_extent

    Path("out").mkdir(exist_ok=True)

    # --- Prior-style (ROI unchanged): voxel entropy original vs retest ---
    entropy_retest = local_entropy_map(
        np.asarray(retest.image(modality).data), kernel_size=7, bins=16
    )
    roi = np.asarray(mask) > 0
    orig_e = np.asarray(entropy_large, dtype=np.float64)
    ret_e = np.asarray(entropy_retest, dtype=np.float64)
    abs_diff = np.abs(orig_e - ret_e)
    roi_diff = abs_diff[roi]
    thresh = float(np.nanpercentile(roi_diff, 75)) if roi_diff.size else 0.0
    voxel_state = np.zeros(orig_e.shape, dtype=np.int32)
    voxel_state[roi & (abs_diff <= thresh)] = 1
    voxel_state[roi & (abs_diff > thresh)] = 2
    print("Prior-style voxel entropy (ROI unchanged)")
    print(f"  precise/stable voxels: {int(np.sum(voxel_state == 1))}")
    print(f"  unstable voxels: {int(np.sum(voxel_state == 2))}")

    index = _densest_roi_index(np.asarray(mask), axis=0)
    orig_sl = _orient(np.take(orig_e, index, axis=0), direction=direction)
    ret_sl = _orient(np.take(ret_e, index, axis=0), direction=direction)
    state_sl = _orient(np.take(voxel_state, index, axis=0), direction=direction)
    extent = imshow_physical_extent(
        (int(orig_sl.shape[0]), int(orig_sl.shape[1])),
        spacing,
        slice_axis=0,
        ndim=3,
    )
    from matplotlib.colors import ListedColormap

    import matplotlib.pyplot as plt

    with use_style("radiology"):
        fig_voxel, axes_v = plt.subplots(
            1, 3, figsize=(11.4, 3.9), constrained_layout=True, facecolor="white"
        )
        for ax, data, title, cmap, vmin, vmax in (
            (axes_v[0], orig_sl, "Entropy, original", "cividis", None, None),
            (axes_v[1], ret_sl, "Entropy, Prior retest", "cividis", None, None),
            (
                axes_v[2],
                state_sl,
                "Stable vs unstable voxels",
                ListedColormap(["#FFFFFF", "#009E73", "#D55E00"]),
                0,
                2,
            ),
        ):
            ax.imshow(
                data,
                cmap=cmap,
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(title)
            ax.axis("off")
        fig_voxel.suptitle("Prior-style voxel stability (mask unchanged)")
        fig_voxel.savefig(
            "out/precise_voxel_stable_vs_unstable.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig_voxel)

    # --- BSpline (ROI changes): habitats before vs after + per-habitat Dice ---
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
    # (panel 2 only) onto the reference by maximal voxel overlap -- the
    # same Hungarian pairing habitat_stability uses -- so habitat 2 is the
    # same spatial region on both panels. Feature-space centroids can
    # disagree with that pairing. Dice is scored on the original pair.
    aligned_map = align_habitat_map(ref_map, mov_map, method="overlap")
    dice_frame = habitat_stability(ref_map, [mov_map])
    print("Habitat Dice after BSplineDeform (Hungarian match)")
    print(dice_frame.to_string(index=False))
    roi_labels = (np.asarray(ref_map.label_array) > 0) | (
        np.asarray(mov_map.label_array) > 0
    )
    raw_disagree = int(
        np.count_nonzero(
            (ref_map.label_array != mov_map.label_array) & roi_labels
        )
    )
    aligned_disagree = int(
        np.count_nonzero(
            (ref_map.label_array != aligned_map.label_array) & roi_labels
        )
    )
    print(
        f"Disagreement voxels (ROI): raw={raw_disagree} "
        f"after_overlap_remap={aligned_disagree}"
    )
    mov_ids = np.asarray(mov_map.label_array)
    aln_ids = np.asarray(aligned_map.label_array)
    was_h3 = mov_ids == 3
    if np.any(was_h3):
        mapped, counts = np.unique(aln_ids[was_h3], return_counts=True)
        print(
            "Raw panel-2 habitat 3 remaps to: "
            + ", ".join(f"H{int(i)} ({int(c)} vx)" for i, c in zip(mapped, counts))
        )

    fig_cmp = plot_habitat_label_compare(
        ffd_subject.image(modality),
        ref_map,
        aligned_map,
        titles=("Original ROI habitats", "Warped ROI habitats"),
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

    def _habitat_feature_row(label_array: np.ndarray) -> dict[str, float]:
        """ITH / MSI / graph / volume scalars for one habitat map."""
        labels_arr = np.asarray(label_array)
        ids = tuple(int(v) for v in np.unique(labels_arr) if int(v) > 0)
        volumes = habitat_volume_fractions(labels_arr, ids)
        n_classes = int(max(ids)) + 1 if ids else 1
        msi = spatial_interaction_matrix(labels_arr, n_classes=n_classes)
        graph = extract_graph_features(labels_arr)
        row: dict[str, float] = {
            "ith_score": float(ith_score(labels_arr)),
            "msi_mean": float(np.mean(msi)) if msi.size else 0.0,
        }
        for hid, frac in volumes.items():
            row[f"volume_h{hid}"] = float(frac)
        for key in (
            "graph_num_habitats",
            "graph_num_nodes_total",
        ):
            if key in graph:
                row[key] = float(graph[key])
        return row

    before = _habitat_feature_row(ref_map.label_array)
    after = _habitat_feature_row(mov_map.label_array)
    names = sorted(set(before) | set(after))
    rel = []
    for name in names:
        a = float(before.get(name, 0.0))
        b = float(after.get(name, 0.0))
        rel.append(abs(a - b) / (abs(a) + abs(b) + 1e-8))
    stable_cut = 0.15
    stable_feats = [n for n, r in zip(names, rel) if r <= stable_cut]
    unstable_feats = [n for n, r in zip(names, rel) if r > stable_cut]
    print("Habitat-level features after BSplineDeform (relative change <= 0.15)")
    print(f"  stable: {stable_feats}")
    print(f"  unstable: {unstable_feats}")

    with use_style("radiology"):
        fig_feat, ax_f = plt.subplots(figsize=(7.2, 3.4))
        colors_f = ["#009E73" if r <= stable_cut else "#D55E00" for r in rel]
        ax_f.bar(range(len(names)), rel, color=colors_f)
        ax_f.axhline(stable_cut, color="0.25", linestyle="--", linewidth=1.0)
        ax_f.set_xticks(range(len(names)))
        ax_f.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax_f.set_ylabel("Relative change")
        ax_f.set_title("ITH / MSI / graph / volume: stable vs unstable")
        fig_feat.tight_layout()
        fig_feat.savefig(
            "out/precise_habitat_feature_stability.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_feat)
    print(
        "Wrote out/precise_voxel_stable_vs_unstable.png, "
        "out/precise_habitat_stability_compare.png, "
        "out/precise_habitat_dice.png, "
        "out/precise_habitat_feature_stability.png"
    )
    # END stability

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        (
            "precise_voxel_stable_vs_unstable.png",
            "precise_habitat_stability_compare.png",
            "precise_habitat_dice.png",
            "precise_habitat_feature_stability.png",
        )
    )
