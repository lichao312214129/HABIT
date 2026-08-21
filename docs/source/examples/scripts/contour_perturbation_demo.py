#!/usr/bin/env python
"""Contour perturbations: one copyable call per operator, plus figures.

Teaches the three mask-only operators that simulate inter-rater contour
variability. They change the ROI, never the image intensities:

* morphological -- uniform grow / shrink (systematic volume bias)
* gradient_weighted -- flip more voxels on fuzzy (low-gradient) edges
* slice_extent -- add or drop whole axial slices at the z ends

The L0 helper ``boundary_band_mask`` draws the strip a mouse actually
traverses. Domain calls go through ``ImagePerturbationRegistry.create``.
``perturb_image`` returns only the intensity volume, so it is the wrong
entry point for these mask-only methods.

Change DATA / MODALITIES / ROI to your preprocessed tree. Accompanies
``docs/source/examples/precise_features.rst``.

Run from the repository root::

    python docs/source/examples/scripts/contour_perturbation_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import numpy as np

from habit import ImagePerturbationRegistry, Subject, cohort_from_directory
from habit.contracts import ArrayImageRef, Geometry
from habit.kernels import boundary_band_mask

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
full_subject = cohort[0]


def _crop_to_roi(item: Subject, modality: str, roi: str, pad: int = 8) -> Subject:
    """Crop one subject to the ROI bounding box plus pad (demo speed)."""
    mask_arr = np.asarray(item.mask(roi).data)
    image_arr = np.asarray(item.image(modality).data)
    nz = np.argwhere(mask_arr > 0)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad + 1, mask_arr.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    # Keep image and mask on one geometry so SimpleITK operators stay aligned.
    # Display figures use ``full_subject`` (mask direction) instead of this crop.
    src = item.image(modality).geometry
    geom = Geometry.from_array(
        image_arr[sl].shape, spacing=src.spacing, direction=src.direction
    )
    return Subject(
        subject_id=item.subject_id,
        images={modality: ArrayImageRef(array=image_arr[sl], geometry=geom)},
        masks={roi: ArrayImageRef(array=mask_arr[sl], geometry=geom)},
    )


subject = _crop_to_roi(full_subject, MODALITIES[0], ROI)
image = subject.image(MODALITIES[0])
mask = subject.mask(ROI)
orig_mask = np.asarray(mask.data)
orig_n = int((orig_mask > 0).sum())
print(f"Subject {subject.subject_id} ROI voxels={orig_n}")
# END example

# BEGIN morphological_grow
# P1 grow: one observer traces a systematically larger contour.
grow = ImagePerturbationRegistry.create(
    "morphological", grow_mm=4.0, roi=ROI, connectivity=1
)
grown = grow(subject, rng=np.random.default_rng(0))
grown_mask = np.asarray(grown.mask(ROI).data)
print(
    f"morphological grow +4 mm: voxels {orig_n} -> {int((grown_mask > 0).sum())}"
)
# END morphological_grow

# BEGIN morphological_shrink
# P1 shrink: one observer traces a systematically smaller contour.
shrink = ImagePerturbationRegistry.create(
    "morphological", grow_mm=-4.0, roi=ROI, connectivity=1
)
shrunk = shrink(subject, rng=np.random.default_rng(0))
shrunk_mask = np.asarray(shrunk.mask(ROI).data)
print(
    f"morphological shrink -4 mm: voxels {orig_n} -> {int((shrunk_mask > 0).sum())}"
)
# END morphological_shrink

# BEGIN boundary_band
# L0 helper: voxels within band_mm of the foreground boundary.
spacing_xyz = tuple(float(v) for v in mask.geometry.spacing)
band = boundary_band_mask(orig_mask, band_mm=4.0, spacing_xyz=spacing_xyz)
print(f"boundary_band_mask 4 mm: band voxels={int(band.sum())}")
# END boundary_band

# BEGIN gradient_weighted
# P3: flip probability scales with (1 - normalised gradient). Fuzzy
# (low-gradient) edges move more than sharp edges. Image is unchanged.
fuzzy = ImagePerturbationRegistry.create(
    "gradient_weighted",
    modality=MODALITIES[0],
    roi=ROI,
    max_radius_voxels=2,
    probability=0.35,
)
fuzzy_subject = fuzzy(subject, rng=np.random.default_rng(7))
fuzzy_mask = np.asarray(fuzzy_subject.mask(ROI).data)
print(
    f"gradient_weighted: voxels {orig_n} -> {int((fuzzy_mask > 0).sum())}"
)
# END gradient_weighted

# BEGIN slice_extent
# P4: copy or clear whole axial slices at the superior / inferior ends.
extent = ImagePerturbationRegistry.create(
    "slice_extent", grow_slices=2, roi=ROI
)
extended = extent(subject, rng=np.random.default_rng(0))
extended_mask = np.asarray(extended.mask(ROI).data)
orig_z = np.flatnonzero(orig_mask.any(axis=(1, 2)))
new_z = np.flatnonzero(extended_mask.any(axis=(1, 2)))
print(
    f"slice_extent grow 2: z [{int(orig_z[0])}, {int(orig_z[-1])}] -> "
    f"[{int(new_z[0])}, {int(new_z[-1])}]"
)
# END slice_extent

# BEGIN figures
# Paste after the Script blocks. Uses full_subject plus the cropped operator
# outputs. Writes out/contour_*.png. Display follows plot_intensity_slice:
# ImageVolume + MaskVolume geometry, radiological convention, ITK-SNAP z.
from scipy import ndimage as ndi

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from habit.viz import use_style
from habit.viz.labels import sanitize_label
from habit.viz.orientation import (
    direction_matrix,
    imshow_physical_extent,
    orient_slice_for_display,
    resolve_display_geometry,
)

Path("out").mkdir(exist_ok=True)
written: list[str] = []
ORIGINAL_COLOR = "#00E5FF"
PERTURBED_COLOR = "#D55E00"
XOR_COLOR = "#F0E442"
BAND_COLOR = "#0072B2"
DISPLAY_PAD = 36
SLICE_AXIS = 0
CONVENTION = "radiological"

# Uncropped volumes keep the file / ITK-SNAP axial index and the mask
# direction (demo LAP: image +z Superior, mask +z Inferior).
full_image = full_subject.image(MODALITIES[0])
full_mask = full_subject.mask(ROI)
full_grey = np.asarray(full_image.data, dtype=np.float64)
full_shape = tuple(int(v) for v in full_grey.shape)
resolved_direction, resolved_spacing = resolve_display_geometry(
    full_image, full_mask
)
direction = direction_matrix(resolved_direction, ndim=3)
spacing_xyz = tuple(float(v) for v in resolved_spacing)

full_roi = np.asarray(full_mask.data) > 0
nz = np.argwhere(full_roi)
crop_lo = np.maximum(nz.min(axis=0) - 8, 0)


def _embed_crop(cropped: np.ndarray) -> np.ndarray:
    """Place a cropped (z, y, x) array back on the full-volume grid."""
    out = np.zeros(full_shape, dtype=cropped.dtype)
    sl = tuple(
        slice(int(start), int(start) + int(length))
        for start, length in zip(crop_lo, cropped.shape)
    )
    out[sl] = cropped
    return out


def _orient_axial(volume: np.ndarray, z_index: int) -> np.ndarray:
    """Take axis-0 and apply the same transform as plot_intensity_slice."""
    plane = np.take(np.asarray(volume), int(z_index), axis=SLICE_AXIS)
    return orient_slice_for_display(
        plane,
        slice_axis=SLICE_AXIS,
        direction=direction,
        convention=CONVENTION,
    )


def _zoom_box(mask_2d: np.ndarray, pad: int = DISPLAY_PAD) -> tuple[slice, slice]:
    """Return (row, col) slices around the oriented ROI, clipped to the plane."""
    rows, cols = np.where(np.asarray(mask_2d) > 0)
    if rows.size == 0:
        return slice(None), slice(None)
    r0 = max(0, int(rows.min()) - pad)
    r1 = min(int(mask_2d.shape[0]), int(rows.max()) + pad + 1)
    c0 = max(0, int(cols.min()) - pad)
    c1 = min(int(mask_2d.shape[1]), int(cols.max()) + pad + 1)
    return slice(r0, r1), slice(c0, c1)


def _apply_zoom(panel: np.ndarray, box: tuple[slice, slice]) -> np.ndarray:
    """Crop one oriented 2D panel to ``box``."""
    return np.asarray(panel)[box]


def _tissue_window(slice_2d: np.ndarray) -> tuple[float, float]:
    """Window like plot_intensity_slice: drop air, then 2nd-90th percentiles."""
    finite = np.asarray(slice_2d, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 16:
        return 0.0, 1.0
    lo_v = float(np.min(finite))
    hi_v = float(np.max(finite))
    if hi_v <= lo_v:
        return lo_v, hi_v
    hist, edges = np.histogram(finite, bins=min(256, int(finite.size)), range=(lo_v, hi_v))
    search_n = max(1, int(np.ceil(hist.size * 0.25)))
    peak = int(np.argmax(hist[:search_n]))
    tissue = finite
    if float(hist[peak]) > 0.10 * float(finite.size):
        kept = finite[finite > float(edges[peak + 1])]
        if kept.size >= 16:
            tissue = kept
    vmin = float(np.percentile(tissue, 2.0))
    vmax = float(np.percentile(tissue, 90.0))
    if vmax <= vmin:
        vmin, vmax = float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.0))
    return vmin, vmax


def _extent_mm(shape_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    """Physical imshow extent so aspect='equal' matches voxel spacing."""
    return imshow_physical_extent(
        shape_hw,
        spacing_xyz,
        slice_axis=SLICE_AXIS,
        ndim=3,
        direction=direction,
        convention=CONVENTION,
    )


def _slice_caption(z_index: int) -> str:
    """File / ITK 0-based z and the common 1-based ITK-SNAP slider."""
    return f"z={int(z_index)} (ITK-SNAP {int(z_index) + 1})"


def _annotate_rl(ax: object) -> None:
    """Radiological markers: patient right on the viewer's left."""
    stroke = [patheffects.withStroke(linewidth=2.4, foreground="black")]
    for x, letter in ((0.07, "R"), (0.93, "L")):
        ax.text(
            x,
            0.50,
            letter,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            path_effects=stroke,
        )


def _draw_anatomy(
    ax: object,
    slice_2d: np.ndarray,
    *,
    extent: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    annotate_rl: bool = True,
) -> None:
    """Draw one oriented greyscale slice with physical aspect."""
    ax.imshow(
        slice_2d,
        cmap="gray",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor("white")
    if annotate_rl:
        _annotate_rl(ax)


def _coherent_mask(mask_2d: np.ndarray) -> np.ndarray:
    """Largest component, fill holes, close 1-voxel gaps (display only).

    Voxel-wise flips otherwise contour as many tiny islands and look like
    a sparse dashed line. Closing absorbs nearby speckles into one outline
    so local displacement (hug vs deviate) is readable.
    """
    binary = np.asarray(mask_2d) > 0
    if not np.any(binary):
        return binary.astype(np.float64)
    labeled, n_lab = ndi.label(binary)
    if n_lab == 0:
        return binary.astype(np.float64)
    sizes = ndi.sum(binary, labeled, index=np.arange(1, n_lab + 1))
    keep = labeled == (int(np.argmax(sizes)) + 1)
    keep = ndi.binary_fill_holes(keep)
    keep = ndi.binary_closing(keep, iterations=1)
    return keep.astype(np.float64)


def _draw_contour(
    ax: object,
    mask_2d: np.ndarray,
    color: str,
    extent: tuple[float, float, float, float],
    *,
    linewidth: float = 2.0,
    coherent: bool = False,
) -> None:
    """Outline a binary mask with a solid line (same extent as anatomy)."""
    binary = _coherent_mask(mask_2d) if coherent else (np.asarray(mask_2d) > 0).astype(
        np.float64
    )
    if not np.any(binary):
        return
    ax.contour(
        binary,
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        linestyles="-",
        origin="upper",
        extent=extent,
    )


def _draw_fill(
    ax: object,
    mask_2d: np.ndarray,
    color: str,
    extent: tuple[float, float, float, float],
    *,
    alpha: float,
) -> None:
    """Fill a binary region (XOR / band) in the same display frame."""
    binary = (np.asarray(mask_2d) > 0).astype(np.float64)
    if not np.any(binary):
        return
    ax.contourf(
        binary,
        levels=[0.5, 1.5],
        colors=[color],
        alpha=alpha,
        origin="upper",
        extent=extent,
    )


def _save_fig(fig: object, name: str) -> None:
    """Write one taught figure under out/ and close it."""
    fig.savefig(f"out/{name}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(name)


orig_full = _embed_crop((orig_mask > 0).astype(np.uint8))
grown_full = _embed_crop((grown_mask > 0).astype(np.uint8))
shrunk_full = _embed_crop((shrunk_mask > 0).astype(np.uint8))
band_full = _embed_crop(np.asarray(band, dtype=np.uint8))
fuzzy_full = _embed_crop((fuzzy_mask > 0).astype(np.uint8))
extended_full = _embed_crop((extended_mask > 0).astype(np.uint8))

counts = np.sum(orig_full > 0, axis=(1, 2))
axial_index = (
    int(np.argmax(counts)) if int(np.max(counts)) > 0 else int(full_shape[0] // 2)
)
itk_snap_slice = axial_index + 1
print(
    f"Display axial numpy/ITK z={axial_index} "
    f"(ITK-SNAP 1-based {itk_snap_slice})"
)

grey_ax = _orient_axial(full_grey, axial_index)
mask_o = _orient_axial(orig_full, axial_index)
zoom = _zoom_box(mask_o)
grey_ax = _apply_zoom(grey_ax, zoom)
mask_o = _apply_zoom(mask_o, zoom)
vmin, vmax = _tissue_window(grey_ax)
extent = _extent_mm((int(grey_ax.shape[0]), int(grey_ax.shape[1])))
plane_hw = (int(grey_ax.shape[0]), int(grey_ax.shape[1]))


def _box_extent(
    parent_extent: tuple[float, float, float, float],
    parent_shape: tuple[int, int],
    box: tuple[slice, slice],
) -> tuple[float, float, float, float]:
    """Physical extent of a row/col crop (origin-upper imshow)."""
    left, right, bottom, top = parent_extent
    nrows, ncols = (int(parent_shape[0]), int(parent_shape[1]))
    r0 = int(box[0].start if box[0].start is not None else 0)
    r1 = int(box[0].stop if box[0].stop is not None else nrows)
    c0 = int(box[1].start if box[1].start is not None else 0)
    c1 = int(box[1].stop if box[1].stop is not None else ncols)
    width = float(right - left)
    height = float(top - bottom)
    x0 = left + (c0 / ncols) * width
    x1 = left + (c1 / ncols) * width
    y_top = top - (r0 / nrows) * height
    y_bot = top - (r1 / nrows) * height
    return (x0, x1, y_bot, y_top)


def _inset_window(
    center: tuple[int, int],
    shape_hw: tuple[int, int],
    half: int = 16,
) -> tuple[slice, slice]:
    """Return (row, col) slices around ``center``, clipped to the plane."""
    cr, cc = int(center[0]), int(center[1])
    r0 = max(0, cr - half)
    r1 = min(int(shape_hw[0]), cr + half)
    c0 = max(0, cc - half)
    c1 = min(int(shape_hw[1]), cc + half)
    return slice(r0, r1), slice(c0, c1)


def _boundary_foci(
    mask_orig: np.ndarray,
    mask_new: np.ndarray,
    weights: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pick a sharp (high-w) and fuzzy (low-w) boundary focus for insets.

    Sharp: high gradient, small distance to the perturbed contour (hugs).
    Fuzzy: low gradient, larger distance (deviates). Foci are kept apart
    so the two insets show different parts of the same slice.
    """
    orig_bin = np.asarray(mask_orig) > 0
    new_bin = np.asarray(mask_new) > 0
    boundary = orig_bin & ~ndi.binary_erosion(orig_bin)
    ys, xs = np.where(boundary)
    if ys.size == 0:
        mid = (int(orig_bin.shape[0] // 2), int(orig_bin.shape[1] // 2))
        return mid, mid
    ww = np.asarray(weights, dtype=np.float64)[ys, xs]
    new_edge = new_bin & ~ndi.binary_erosion(new_bin)
    if np.any(new_edge):
        dist = ndi.distance_transform_edt(~new_edge)
        dd = dist[ys, xs]
    else:
        dd = np.zeros(ys.shape, dtype=np.float64)
    sharp_i = int(np.argmax(ww - 0.20 * dd))
    sharp = (int(ys[sharp_i]), int(xs[sharp_i]))
    sep = np.hypot(ys.astype(np.float64) - sharp[0], xs.astype(np.float64) - sharp[1])
    fuzzy_score = (1.0 - ww) + 0.35 * dd
    fuzzy_score[sep < 14.0] = -np.inf
    fuzzy_i = int(np.argmax(fuzzy_score))
    fuzzy = (int(ys[fuzzy_i]), int(xs[fuzzy_i]))
    return sharp, fuzzy


def _mark_inset(
    ax: object,
    box_extent: tuple[float, float, float, float],
    color: str,
    label: str,
) -> None:
    """Draw a labelled rectangle on the parent panel."""
    x0, x1, y0, y1 = box_extent
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            zorder=5,
        )
    )
    stroke = [patheffects.withStroke(linewidth=2.2, foreground="black")]
    ax.text(
        x0 + 0.04 * (x1 - x0),
        y1 - 0.06 * (y1 - y0),
        label,
        color="white",
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
        path_effects=stroke,
        zorder=6,
    )


def _save_contour_xor(
    mask_new_full: np.ndarray,
    filename: str,
    left_title: str,
    right_title: str,
    new_label: str,
) -> None:
    """Two-panel axial figure: original vs perturbed contours, plus XOR."""
    mask_n = _apply_zoom(_orient_axial(mask_new_full, axial_index), zoom)
    xor_map = np.abs(mask_n.astype(np.float64) - mask_o.astype(np.float64))
    z_txt = _slice_caption(axial_index)
    with use_style("radiology"):
        fig, axes = plt.subplots(
            1, 2, figsize=(9.6, 5.0), constrained_layout=True, facecolor="white"
        )
        for ax, show_xor, title in (
            (axes[0], False, f"{left_title}  {z_txt}"),
            (axes[1], True, f"{right_title}  {z_txt}"),
        ):
            _draw_anatomy(ax, grey_ax, extent=extent, vmin=vmin, vmax=vmax)
            if show_xor:
                _draw_fill(ax, xor_map, XOR_COLOR, extent, alpha=0.50)
            _draw_contour(ax, mask_o, ORIGINAL_COLOR, extent)
            _draw_contour(ax, mask_n, PERTURBED_COLOR, extent, coherent=True)
            ax.set_title(sanitize_label(title), fontsize=10)
        fig.legend(
            handles=[
                Line2D([0], [0], color=ORIGINAL_COLOR, lw=2.0, label="Original ROI"),
                Line2D([0], [0], color=PERTURBED_COLOR, lw=2.0, label=new_label),
                Patch(facecolor=XOR_COLOR, edgecolor="none", alpha=0.50, label="XOR"),
            ],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.patch.set_facecolor("white")
    _save_fig(fig, filename)


_save_contour_xor(
    grown_full,
    "contour_morphological_grow.png",
    "Original vs grown ROI",
    "Membership change (XOR)",
    "Grown ROI (+4 mm)",
)
_save_contour_xor(
    shrunk_full,
    "contour_morphological_shrink.png",
    "Original vs shrunk ROI",
    "Membership change (XOR)",
    "Shrunk ROI (-4 mm)",
)

# Boundary band: the strip a radiologist's mouse actually traverses.
band_sl = _apply_zoom(_orient_axial(band_full, axial_index), zoom)
with use_style("radiology"):
    fig_band, ax_band = plt.subplots(
        figsize=(5.2, 5.4), constrained_layout=True, facecolor="white"
    )
    _draw_anatomy(ax_band, grey_ax, extent=extent, vmin=vmin, vmax=vmax)
    _draw_fill(ax_band, band_sl, BAND_COLOR, extent, alpha=0.40)
    _draw_contour(ax_band, mask_o, ORIGINAL_COLOR, extent)
    ax_band.set_title(
        sanitize_label(f"Boundary band (4 mm)  {_slice_caption(axial_index)}"),
        fontsize=10,
    )
    fig_band.legend(
        handles=[
            Line2D([0], [0], color=ORIGINAL_COLOR, lw=2.0, label="Original ROI"),
            Patch(facecolor=BAND_COLOR, edgecolor="none", alpha=0.40, label="Band"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig_band.patch.set_facecolor("white")
_save_fig(fig_band, "contour_boundary_band.png")

# Gradient-weighted: anatomy, gradient, both solid contours, plus
# sharp/fuzzy insets. XOR is not the teaching panel: the operator
# flips inside a fixed-radius band, so a filled XOR ring looks
# uniformly wide even when sharp edges hug and fuzzy edges deviate.
grad = ndi.gaussian_gradient_magnitude(full_grey, sigma=1.0)
peak = float(grad.max())
weights = grad / peak if peak > 0.0 else np.zeros_like(grad)
w_sl = _apply_zoom(_orient_axial(weights, axial_index), zoom)
mask_f = _apply_zoom(_orient_axial(fuzzy_full, axial_index), zoom)
xor_f = np.abs(mask_f.astype(np.float64) - mask_o.astype(np.float64))
op_band = ndi.binary_dilation(orig_full > 0, iterations=2) ^ ndi.binary_erosion(
    orig_full > 0, iterations=2
)
band_sl = _apply_zoom(_orient_axial(op_band.astype(np.uint8), axial_index), zoom)
changed = xor_f > 0
if np.any(band_sl):
    w_band = w_sl[band_sl.astype(bool)]
    w_changed = w_sl[changed] if np.any(changed) else np.array([])
    print(
        f"gradient_weighted {_slice_caption(axial_index)}: "
        f"band mean w={float(w_band.mean()):.3f}; "
        f"changed voxels={int(changed.sum())} "
        f"mean w={float(w_changed.mean()) if w_changed.size else float('nan'):.3f}"
    )
mask_f_draw = _coherent_mask(mask_f)
sharp_rc, fuzzy_rc = _boundary_foci(mask_o, mask_f_draw, w_sl)
print(
    f"gradient_weighted insets: sharp={sharp_rc} w={float(w_sl[sharp_rc]):.3f}; "
    f"fuzzy={fuzzy_rc} w={float(w_sl[fuzzy_rc]):.3f}"
)
sharp_box = _inset_window(sharp_rc, plane_hw, half=16)
fuzzy_box = _inset_window(fuzzy_rc, plane_hw, half=16)
sharp_ext = _box_extent(extent, plane_hw, sharp_box)
fuzzy_ext = _box_extent(extent, plane_hw, fuzzy_box)
SHARP_BOX_COLOR = "#F0E442"
FUZZY_BOX_COLOR = "#CC79A7"
z_txt = _slice_caption(axial_index)
with use_style("radiology"):
    fig_gw = plt.figure(figsize=(10.4, 9.8), constrained_layout=True, facecolor="white")
    gs_gw = fig_gw.add_gridspec(2, 2)
    ax_an = fig_gw.add_subplot(gs_gw[0, 0])
    ax_w = fig_gw.add_subplot(gs_gw[0, 1])
    ax_ov = fig_gw.add_subplot(gs_gw[1, 0])
    gs_in = gs_gw[1, 1].subgridspec(1, 2, wspace=0.08)
    ax_sh = fig_gw.add_subplot(gs_in[0, 0])
    ax_fz = fig_gw.add_subplot(gs_in[0, 1])

    _draw_anatomy(ax_an, grey_ax, extent=extent, vmin=vmin, vmax=vmax)
    _draw_contour(ax_an, mask_o, ORIGINAL_COLOR, extent)
    ax_an.set_title(sanitize_label(f"1. Anatomy + original ROI  {z_txt}"), fontsize=10)

    im_w = ax_w.imshow(
        w_sl,
        cmap="magma",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
    )
    ax_w.set_aspect("equal", adjustable="box")
    ax_w.axis("off")
    ax_w.set_facecolor("white")
    _annotate_rl(ax_w)
    _draw_contour(ax_w, mask_o, ORIGINAL_COLOR, extent)
    ax_w.set_title(sanitize_label(f"2. Gradient (bright = sharp)  {z_txt}"), fontsize=10)
    fig_gw.colorbar(im_w, ax=ax_w, fraction=0.046, pad=0.03, label="w")

    _draw_anatomy(ax_ov, grey_ax, extent=extent, vmin=vmin, vmax=vmax)
    _draw_contour(ax_ov, mask_o, ORIGINAL_COLOR, extent)
    _draw_contour(ax_ov, mask_f_draw, PERTURBED_COLOR, extent)
    _mark_inset(ax_ov, sharp_ext, SHARP_BOX_COLOR, "S")
    _mark_inset(ax_ov, fuzzy_ext, FUZZY_BOX_COLOR, "F")
    ax_ov.set_title(sanitize_label(f"3. Original vs perturbed  {z_txt}"), fontsize=10)

    for ax_in, box, box_ext, title in (
        (ax_sh, sharp_box, sharp_ext, "S  Sharp edge (hugs)"),
        (ax_fz, fuzzy_box, fuzzy_ext, "F  Fuzzy edge (deviates)"),
    ):
        _draw_anatomy(
            ax_in,
            _apply_zoom(grey_ax, box),
            extent=box_ext,
            vmin=vmin,
            vmax=vmax,
            annotate_rl=False,
        )
        _draw_contour(ax_in, _apply_zoom(mask_o, box), ORIGINAL_COLOR, box_ext, linewidth=2.4)
        _draw_contour(
            ax_in, _apply_zoom(mask_f_draw, box), PERTURBED_COLOR, box_ext, linewidth=2.4
        )
        ax_in.set_title(sanitize_label(title), fontsize=9)
    fig_gw.legend(
        handles=[
            Line2D([0], [0], color=ORIGINAL_COLOR, lw=2.0, label="Original ROI"),
            Line2D(
                [0],
                [0],
                color=PERTURBED_COLOR,
                lw=2.0,
                label="Gradient-weighted ROI",
            ),
            Rectangle(
                (0, 0), 1, 1, fill=False, edgecolor=SHARP_BOX_COLOR, lw=1.5, label="Sharp inset"
            ),
            Rectangle(
                (0, 0), 1, 1, fill=False, edgecolor=FUZZY_BOX_COLOR, lw=1.5, label="Fuzzy inset"
            ),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig_gw.patch.set_facecolor("white")
_save_fig(fig_gw, "contour_gradient_weighted.png")

# Slice-extent: first / mid / last occupied axial slices, original vs grown.
orig_occupied = np.flatnonzero(orig_full.any(axis=(1, 2)))
new_occupied = np.flatnonzero(extended_full.any(axis=(1, 2)))
orig_idx = (
    int(orig_occupied[0]),
    int(orig_occupied[len(orig_occupied) // 2]),
    int(orig_occupied[-1]),
)
new_idx = (
    int(new_occupied[0]),
    int(new_occupied[len(new_occupied) // 2]),
    int(new_occupied[-1]),
)
with use_style("radiology"):
    fig_z, axes_z = plt.subplots(
        2, 3, figsize=(11.2, 8.0), constrained_layout=True, facecolor="white"
    )
    for col, (zi, title) in enumerate(
        zip(orig_idx, ("First occupied", "Mid occupied", "Last occupied"))
    ):
        sl = _orient_axial(full_grey, zi)
        mk = _orient_axial(orig_full, zi)
        box = _zoom_box(mk)
        sl = _apply_zoom(sl, box)
        mk = _apply_zoom(mk, box)
        sl_ext = _extent_mm((int(sl.shape[0]), int(sl.shape[1])))
        sl_vmin, sl_vmax = _tissue_window(sl)
        _draw_anatomy(axes_z[0, col], sl, extent=sl_ext, vmin=sl_vmin, vmax=sl_vmax)
        _draw_contour(axes_z[0, col], mk, ORIGINAL_COLOR, sl_ext)
        axes_z[0, col].set_title(
            sanitize_label(f"Original {title}  {_slice_caption(zi)}"),
            fontsize=9,
        )
    for col, (zi, title) in enumerate(
        zip(new_idx, ("First occupied", "Mid occupied", "Last occupied"))
    ):
        sl = _orient_axial(full_grey, zi)
        mk = _orient_axial(extended_full, zi)
        box = _zoom_box(mk)
        sl = _apply_zoom(sl, box)
        mk = _apply_zoom(mk, box)
        sl_ext = _extent_mm((int(sl.shape[0]), int(sl.shape[1])))
        sl_vmin, sl_vmax = _tissue_window(sl)
        _draw_anatomy(axes_z[1, col], sl, extent=sl_ext, vmin=sl_vmin, vmax=sl_vmax)
        _draw_contour(axes_z[1, col], mk, PERTURBED_COLOR, sl_ext, coherent=True)
        axes_z[1, col].set_title(
            sanitize_label(f"Grown {title}  {_slice_caption(zi)}"),
            fontsize=9,
        )
    fig_z.legend(
        handles=[
            Line2D([0], [0], color=ORIGINAL_COLOR, lw=2.0, label="Original ROI"),
            Line2D(
                [0],
                [0],
                color=PERTURBED_COLOR,
                lw=2.0,
                label="slice_extent +2 slices",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig_z.patch.set_facecolor("white")
_save_fig(fig_z, "contour_slice_extent.png")
print("Wrote " + ", ".join(f"out/{name}" for name in written))
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        (
            "contour_morphological_grow.png",
            "contour_morphological_shrink.png",
            "contour_boundary_band.png",
            "contour_gradient_weighted.png",
            "contour_slice_extent.png",
        )
    )
