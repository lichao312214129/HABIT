# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""L0 kernels: image perturbation for simulated test-retest analysis.

Pure functions that turn one image into a perturbed copy of itself:
Gaussian noise addition, sub-voxel translation, small-angle rotation, a
composed rigid (translation + rotation) resample, and noise-level
estimation.

This is the perturbation family Prior et al. used to probe voxel-radiomics
repeatability:

    Prior O, Macarro C, Navarro V, et al. Identification of Precise 3D CT
    Radiomics for Habitat Computation by Machine Learning in Cancer.
    Radiol Artif Intell. 2024;6(2):e230118. doi:10.1148/ryai.230118

Appendix S2 of that paper (and the matching MIRP 1.2.0 chain they ran)
applies, in order:

1. Additive Gaussian noise whose sigma is estimated from the image
   (Chang's wavelet estimator when no level is configured; alternatively
   the ROI standard deviation). Noise is added to the *whole* image.
2. Sub-voxel translation: a fraction ``η`` of the voxel spacing along
   x, y and z (MIRP ``perturbation_translation_fraction``, typically 0.5).
   HABIT expresses the same shift in voxel units.
3. In-plane rotation of 0.5 degrees about the z (axial) axis.

Intensity images are resampled with B-spline interpolation; label masks
use nearest neighbour. Geometric transforms are resampled back onto the
ORIGINAL grid so perturbed maps stay voxel-wise comparable.

MIRP 1.2.0 (the paper) applied translation and rotation as two resamples.
MIRP ≥ 2 composes them into one affine. :func:`rigid_transform_image` is
that single-resample path. ROI morphological variation (``perturbation_roi_adapt_size``) is *not*
part of the Prior 2024 protocol. MONAI elastic / B-spline free-form
deformation of the image and ROI is a separate optional domain
component (``BSplineDeformPerturbation``), not these L0 kernels.

Implemented natively so HABIT does not depend on MIRP (EUPL-1.2 license,
Python >= 3.11 -- both incompatible with HABIT).

Conventions
-----------
* Voxel arrays are numpy arrays in ``(z, y, x)`` order, the SimpleITK
  convention; geometric kernels take and return ``sitk.Image`` objects so
  spacing, origin and direction are honoured.
* Shifts are specified in VOXEL units in SimpleITK ``(x, y, z)`` axis order
  and converted to physical offsets with the image's spacing and direction.
* Geometric transforms are resampled back onto the ORIGINAL grid, so a
  perturbed image stays directly comparable to its source voxel-by-voxel.
* Randomness always enters through an explicit ``numpy.random.Generator``;
  these kernels never touch global random state.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    # Typing-only: SimpleITK is an imaging backend and must stay out of
    # ``import habit.kernels`` (see tests/test_import_lightweight.py), so the
    # geometric kernels import it lazily inside the function body.
    import SimpleITK as sitk

__all__ = [
    "estimate_noise_sigma",
    "add_gaussian_noise",
    "translate_image",
    "rotate_image",
    "rigid_transform_image",
    "binary_mask_dice",
    "morphological_grow_shrink",
    "boundary_band_mask",
    "boundary_weighted_perturbation",
    "slice_extent_perturbation",
]

#: Interpolator names accepted by the geometric kernels, mapped to the SimpleITK
#: enum ATTRIBUTE names (resolved lazily). ``bspline`` is the paper's choice
#: for intensity images; ``nearest`` is the only valid choice for label masks.
_INTERPOLATORS = {
    "nearest": "sitkNearestNeighbor",
    "linear": "sitkLinear",
    "bspline": "sitkBSpline",
}


def _interpolator_code(name: str) -> int:
    """
    Resolve an interpolator name to its SimpleITK enum value.

    Args:
        name: One of ``"nearest"``, ``"linear"``, ``"bspline"``.

    Returns:
        The ``sitk`` interpolator constant.

    Raises:
        ValueError: For an unknown interpolator name.
    """
    import SimpleITK as sitk

    try:
        return int(getattr(sitk, _INTERPOLATORS[name]))
    except KeyError:
        raise ValueError(
            f"Unknown interpolator {name!r}; expected one of "
            f"{sorted(_INTERPOLATORS)}."
        ) from None


def estimate_noise_sigma(
    array: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "chang",
) -> float:
    """
    Estimate the Gaussian noise level of an image.

    Args:
        array: Image voxel array in ``(z, y, x)`` order.
        mask: Optional ROI selector; only used by ``method="roi_std"``.
        method: ``"chang"`` -- wavelet estimator (median absolute coefficient
            of the finest high-high subband of a coif1 transform, divided by
            0.6754, the normal-consistency constant), applied on the last
            two axes (in-plane for axial acquisitions). This is the
            estimator MIRP uses when no noise level is given. ``"roi_std"``
            -- standard deviation of the intensities inside ``mask`` (or of
            the whole array when no mask is given), the alternative named in
            the paper.

    Returns:
        The estimated noise standard deviation in intensity units; ``0.0``
        for a constant image.

    Raises:
        ValueError: For an unknown method, arrays with fewer than two axes
            (``chang``), or an empty ``roi_std`` mask.
        ImportError: If ``method="chang"`` and PyWavelets is not installed.
    """
    values = np.asarray(array, dtype=np.float64)
    if method == "roi_std":
        if mask is not None:
            selector = np.asarray(mask) > 0
            if not selector.any():
                raise ValueError(
                    "estimate_noise_sigma: roi_std mask selects no voxels."
                )
            values = values[selector]
        return float(np.std(values))
    if method != "chang":
        raise ValueError(
            f"estimate_noise_sigma: unknown method {method!r}; "
            "expected 'chang' or 'roi_std'."
        )
    if values.ndim < 2:
        raise ValueError(
            "estimate_noise_sigma: the chang method needs at least two axes; "
            f"got ndim={values.ndim}."
        )
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "estimate_noise_sigma(method='chang') requires PyWavelets "
            "(pip install PyWavelets)."
        ) from exc
    # wavedecn requires even-sized axes; pad by edge replication (MIRP does
    # the same) so the decomposition is well defined for odd-sized images.
    pad_width = [(0, 0)] * values.ndim
    for axis in (values.ndim - 2, values.ndim - 1):
        if values.shape[axis] % 2:
            pad_width[axis] = (0, 1)
    if any(width != (0, 0) for width in pad_width):
        values = np.pad(values, pad_width, mode="edge")
    _, details = pywt.wavedecn(values, "coif1", level=1, axes=(-2, -1))
    high_high = details["dd"]
    return float(np.median(np.abs(high_high)) / 0.6754)


def add_gaussian_noise(
    array: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    mask: Optional[np.ndarray] = None,
    round_to_int: bool = False,
) -> np.ndarray:
    """
    Return a copy of ``array`` with zero-mean Gaussian noise added.

    Following MIRP, the noise is added to the WHOLE image by default (the
    ROI only determines ``sigma``); pass ``mask`` to restrict the addition
    to the region of interest.

    Args:
        array: Image voxel array in ``(z, y, x)`` order.
        sigma: Noise standard deviation in intensity units; ``0`` returns an
            unmodified copy.
        rng: Random generator supplying the noise field.
        mask: Optional ROI selector; noise is added only where ``mask > 0``.
        round_to_int: Round the result to whole numbers, mirroring MIRP's
            handling of integer-valued CT (HU) data.

    Returns:
        The perturbed array as ``float64``, same shape as ``array``.
    """
    result = np.asarray(array, dtype=np.float64).copy()
    if sigma <= 0.0:
        return result
    noise = rng.normal(0.0, float(sigma), size=result.shape)
    if mask is None:
        result += noise
    else:
        selector = np.asarray(mask) > 0
        result[selector] += noise[selector]
    if round_to_int:
        result = np.rint(result)
    return result


def translate_image(
    image: sitk.Image,
    shift_voxels: Sequence[float],
    interpolator: str = "bspline",
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Translate image content by a (sub-)voxel shift, resampled on the same grid.

    Args:
        image: Source image.
        shift_voxels: Shift in VOXEL units, SimpleITK ``(x, y, z)`` axis
            order; fractions of a voxel are the intended use. The physical
            offset is ``direction @ (shift * spacing)``.
        interpolator: ``"bspline"`` (paper default), ``"linear"`` or
            ``"nearest"`` (required for label masks).
        default_value: Intensity for voxels mapped outside the source.

    Returns:
        The translated image on ``image``'s grid.

    Raises:
        ValueError: If ``shift_voxels`` does not have exactly 3 components.
    """
    import SimpleITK as sitk

    shift = np.asarray(shift_voxels, dtype=np.float64)
    if shift.shape != (3,):
        raise ValueError(
            "translate_image: shift_voxels must have 3 components "
            f"(x, y, z); got shape {shift.shape}."
        )
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    offset = direction @ (shift * spacing)
    # sitk.Resample reads the transform as output-point -> input-point, so a
    # content shift of +offset is requested with the negated transform.
    transform = sitk.TranslationTransform(3, (-offset).tolist())
    return sitk.Resample(
        image,
        image,
        transform,
        _interpolator_code(interpolator),
        float(default_value),
    )


def rotate_image(
    image: sitk.Image,
    angle_degrees: float,
    axis: str = "z",
    interpolator: str = "bspline",
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Rotate image content about the image centre, resampled on the same grid.

    Args:
        image: Source image.
        angle_degrees: Rotation angle in degrees; the paper uses 0.5. The
            positive sense is counterclockwise looking down the positive
            axis towards the origin (right-hand rule about ``axis``).
        axis: Axis to rotate around: ``"x"``, ``"y"`` or ``"z"`` (``"z"``
            is the axial in-plane axis, the paper's choice).
        interpolator: ``"bspline"`` (paper default), ``"linear"`` or
            ``"nearest"`` (required for label masks).
        default_value: Intensity for voxels mapped outside the source.

    Returns:
        The rotated image on ``image``'s grid.

    Raises:
        ValueError: For an unknown ``axis``.
    """
    import SimpleITK as sitk

    axes = {"x": 0, "y": 1, "z": 2}
    if axis not in axes:
        raise ValueError(
            f"rotate_image: axis must be one of {sorted(axes)}; got {axis!r}."
        )
    centre = image.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2.0 for size in image.GetSize()]
    )
    angles = [0.0, 0.0, 0.0]
    # Negative angle: sitk.Resample reads the transform as output -> input,
    # so this yields a content rotation of +angle_degrees.
    angles[axes[axis]] = -math.radians(float(angle_degrees))
    transform = sitk.Euler3DTransform(centre, *angles)
    return sitk.Resample(
        image,
        image,
        transform,
        _interpolator_code(interpolator),
        float(default_value),
    )


def rigid_transform_image(
    image: sitk.Image,
    shift_voxels: Sequence[float],
    angle_degrees: float,
    axis: str = "z",
    interpolator: str = "bspline",
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Translate then rotate in ONE resample (MIRP ≥ 2 affine composition).

    Prior et al. 2024 used MIRP 1.2.0, which applied translation and
    rotation as two successive interpolations. Composing them avoids a
    second B-spline pass. Content mapping is translate-then-rotate about
    the image centre, matching HABIT's default chain order.

    Args:
        image: Source image.
        shift_voxels: Translation in VOXEL units, SimpleITK ``(x, y, z)``.
        angle_degrees: Rotation angle in degrees (paper default 0.5).
        axis: Axis to rotate around: ``"x"``, ``"y"`` or ``"z"``.
        interpolator: ``"bspline"`` (paper default), ``"linear"`` or
            ``"nearest"`` (required for label masks).
        default_value: Intensity for voxels mapped outside the source.

    Returns:
        The rigidly perturbed image on ``image``'s grid.

    Raises:
        ValueError: If ``shift_voxels`` is not length 3 or ``axis`` is unknown.
    """
    import SimpleITK as sitk

    shift = np.asarray(shift_voxels, dtype=np.float64)
    if shift.shape != (3,):
        raise ValueError(
            "rigid_transform_image: shift_voxels must have 3 components "
            f"(x, y, z); got shape {shift.shape}."
        )
    axes = {"x": 0, "y": 1, "z": 2}
    if axis not in axes:
        raise ValueError(
            f"rigid_transform_image: axis must be one of {sorted(axes)}; "
            f"got {axis!r}."
        )
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    offset = direction @ (shift * spacing)
    centre = image.TransformContinuousIndexToPhysicalPoint(
        [(size - 1) / 2.0 for size in image.GetSize()]
    )
    # sitk.Resample maps output -> input. Content is translate then rotate,
    # so the inverse is rotate^{-1} then translate^{-1}. CompositeTransform
    # applies the last-added transform first.
    translation_inv = sitk.TranslationTransform(3, (-offset).tolist())
    angles = [0.0, 0.0, 0.0]
    angles[axes[axis]] = -math.radians(float(angle_degrees))
    rotation_inv = sitk.Euler3DTransform(centre, *angles)
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(translation_inv)
    composite.AddTransform(rotation_inv)
    return sitk.Resample(
        image,
        image,
        composite,
        _interpolator_code(interpolator),
        float(default_value),
    )


def binary_mask_dice(reference: np.ndarray, moved: np.ndarray) -> float:
    """
    Dice coefficient of two binary masks on the same voxel grid.

    Foreground is every voxel with a non-zero label. Empty-vs-empty is
    defined as 1.0 so a missing ROI does not look like a total mismatch.

    Args:
        reference: Reference mask (any integer / boolean array).
        moved: Compared mask; must have the same shape.

    Returns:
        float: Dice in ``[0, 1]``.

    Raises:
        ValueError: When the arrays have different shapes.
    """
    ref = np.asarray(reference) != 0
    mov = np.asarray(moved) != 0
    if ref.shape != mov.shape:
        raise ValueError(
            "binary_mask_dice: shapes differ: "
            f"{tuple(ref.shape)} vs {tuple(mov.shape)}."
        )
    intersection = int(np.count_nonzero(ref & mov))
    n_ref = int(np.count_nonzero(ref))
    n_moved = int(np.count_nonzero(mov))
    if n_ref + n_moved == 0:
        return 1.0
    return float(2.0 * intersection / (n_ref + n_moved))


def _iterations_for_mm(
    distance_mm: float, spacing_xyz: Sequence[float], voxel_scale: Sequence[float]
) -> int:
    """
    Convert a physical morphological radius to a whole iteration count.

    Morphology runs on voxel grids with an isotropic structuring element, so
    a physical distance becomes an iteration count via the smallest in-plane
    spacing (the limiting axis). At least one iteration is applied whenever
    the requested distance is positive.

    Args:
        distance_mm: Requested grow/shrink distance in millimetres.
        spacing_xyz: Voxel spacing in SimpleITK ``(x, y, z)`` order.
        voxel_scale: Alias kept for call-site clarity; same as
            ``spacing_xyz`` (unused placeholder for anisotropic handling).

    Returns:
        A non-negative integer iteration count.
    """
    if distance_mm <= 0.0:
        return 0
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    positive = spacing[spacing > 0]
    base = float(np.min(positive)) if positive.size else 1.0
    return max(1, int(round(distance_mm / base)))


def morphological_grow_shrink(
    mask: np.ndarray,
    grow_mm: float,
    spacing_xyz: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = 1,
) -> np.ndarray:
    """
    Uniformly dilate (``grow_mm > 0``) or erode (``grow_mm < 0``) a mask.

    This is MIRP ``perturbation_roi_adapt_size``: the systematic component
    of inter-rater contour variability, where one observer consistently
    traces slightly larger or smaller than another. Only the mask changes;
    image intensities are untouched. Applied per foreground label so a
    multi-label ROI grows each region instead of merging them.

    Args:
        mask: Integer / boolean label array in ``(z, y, x)`` order; ``0``
            is background.
        grow_mm: Physical radius in millimetres. Positive dilates, negative
            erodes, zero returns an unchanged copy.
        spacing_xyz: Voxel spacing in SimpleITK ``(x, y, z)`` order, used to
            convert millimetres to an iteration count.
        connectivity: Structuring-element connectivity in ``{1, 2, 3}``;
            ``1`` is the 6-connected face neighbourhood (the conservative,
            MIRP-like default).

    Returns:
        A new label array, same shape and dtype as ``mask``.

    Raises:
        ValueError: If erosion removes every foreground voxel (the ROI
            would vanish), or ``connectivity`` is outside ``{1, 2, 3}``.
    """
    from scipy import ndimage as _ndi

    labels = np.asarray(mask)
    if connectivity not in (1, 2, 3):
        raise ValueError(
            f"morphological_grow_shrink: connectivity must be in {{1,2,3}}; "
            f"got {connectivity}."
        )
    iterations = _iterations_for_mm(abs(float(grow_mm)), spacing_xyz, spacing_xyz)
    result = np.array(labels, copy=True)
    if iterations == 0:
        return result
    structure = _ndi.generate_binary_structure(labels.ndim, connectivity)
    grow = float(grow_mm) > 0.0
    foreground_labels = np.unique(labels)
    foreground_labels = foreground_labels[foreground_labels != 0]
    if foreground_labels.size == 0:
        return result
    if grow:
        # Grow the UNION of foreground so regions do not overwrite each
        # other, then restrict to the original foreground label set.
        union = labels != 0
        dilated = _ndi.binary_dilation(
            union, structure=structure, iterations=iterations
        )
        # Fill newly claimed background voxels by nearest original label.
        added = dilated & ~union
        if added.any():
            _, nearest = _ndi.distance_transform_edt(
                ~union, return_distances=True, return_indices=True
            )
            result[added] = labels[tuple(nearest[:, added])]
        return result
    # Erode each label independently to preserve label identity.
    for label in foreground_labels:
        region = labels == label
        eroded = _ndi.binary_erosion(region, structure=structure, iterations=iterations)
        if not eroded.any():
            raise ValueError(
                "morphological_grow_shrink: erosion of "
                f"{abs(float(grow_mm))} mm removes label {int(label)} entirely; "
                "reduce the magnitude."
            )
        result[~eroded & region] = 0
    return result


def boundary_band_mask(
    mask: np.ndarray,
    band_mm: float,
    spacing_xyz: Sequence[float] = (1.0, 1.0, 1.0),
    connectivity: int = 1,
) -> np.ndarray:
    """
    Return the voxels within ``band_mm`` of the foreground boundary.

    The band is the union of the outer dilation shell and the inner erosion
    shell of the foreground, i.e. the strip a radiologist's mouse actually
    traverses. Used to weight boundary perturbations.

    Args:
        mask: Integer / boolean label array; ``0`` is background.
        band_mm: Half-width of the band in millimetres.
        spacing_xyz: Voxel spacing in SimpleITK ``(x, y, z)`` order.
        connectivity: Structuring-element connectivity in ``{1, 2, 3}``.

    Returns:
        A boolean array, ``True`` inside the boundary band.
    """
    from scipy import ndimage as _ndi

    labels = np.asarray(mask)
    union = labels != 0
    if not union.any():
        return np.zeros(labels.shape, dtype=bool)
    iterations = _iterations_for_mm(abs(float(band_mm)), spacing_xyz, spacing_xyz)
    structure = _ndi.generate_binary_structure(labels.ndim, connectivity)
    dilated = _ndi.binary_dilation(union, structure=structure, iterations=iterations)
    eroded = _ndi.binary_erosion(union, structure=structure, iterations=iterations)
    return (dilated ^ eroded)


def boundary_weighted_perturbation(
    mask: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    max_radius_voxels: int = 2,
    probability: float = 0.5,
) -> np.ndarray:
    """
    Locally grow or shrink a mask where ``weights`` is high (gradient-weighted).

    Models the fact that inter-rater disagreement concentrates where image
    contrast is poor: boundary voxels at high-gradient (sharp) edges are
    drawn consistently, whereas low-gradient (fuzzy) edges vary. ``weights``
    is typically a normalised gradient-magnitude map; the local perturbation
    probability scales with ``1 - weight`` so fuzzy edges move more.

    A random subset of boundary voxels is flipped (foreground -> background
    shrinks, background -> foreground grows) within a local radius, biased
    toward the low-weight side.

    Args:
        mask: Integer / boolean label array; ``0`` is background.
        weights: Per-voxel weight in ``[0, 1]``, same shape as ``mask``;
            high means a confident (sharp) edge. Typically a normalised
            gradient magnitude of the driving image.
        rng: Random generator supplying the flip decisions.
        max_radius_voxels: Neighbourhood radius bounding each local flip.
        probability: Base flip probability at zero weight; the effective
            probability is ``probability * (1 - weight)``.

    Returns:
        A new label array, same shape and dtype as ``mask``.

    Raises:
        ValueError: If ``weights`` shape differs from ``mask``.
    """
    from scipy import ndimage as _ndi

    labels = np.asarray(mask)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != labels.shape:
        raise ValueError(
            "boundary_weighted_perturbation: weights shape "
            f"{tuple(weights.shape)} != mask shape {tuple(labels.shape)}."
        )
    union = labels != 0
    if not union.any():
        return np.array(labels, copy=True)
    band = _ndi.binary_dilation(union, iterations=int(max(1, max_radius_voxels))) ^ (
        _ndi.binary_erosion(union, iterations=int(max(1, max_radius_voxels)))
    )
    result = np.array(labels, copy=True)
    w = np.clip(weights, 0.0, 1.0)
    flip_prob = float(probability) * (1.0 - w)
    draw = rng.random(labels.shape)
    # Grow: background voxels in the band whose draw falls under the
    # local (low-gradient) probability become foreground.
    grow_sel = band & ~union & (draw < flip_prob)
    if grow_sel.any():
        _, nearest = _ndi.distance_transform_edt(
            ~union, return_distances=True, return_indices=True
        )
        result[grow_sel] = labels[tuple(nearest[:, grow_sel])]
    # Shrink: foreground voxels in the band whose draw falls under the
    # local probability become background.
    shrink_sel = band & union & (draw < flip_prob)
    result[shrink_sel] = 0
    return result


def slice_extent_perturbation(
    mask: np.ndarray,
    grow_slices: int = 0,
    shrink_slices: int = 0,
    rng: Optional[np.random.Generator] = None,
    max_slices: int = 0,
) -> np.ndarray:
    """
    Add or remove whole axial slices at the superior/inferior ROI ends.

    Models z-axis delineation variability: observers often agree in-plane
    but differ on the first and last slice they call tumour (slice-extent
    or "end-slice" disagreement). Operates on the ``z`` (first) axis only.

    Provide either fixed ``grow_slices`` / ``shrink_slices`` or a random
    ``max_slices`` with ``rng`` (each end independently draws a count in
    ``[-max_slices, +max_slices]``; positive grows, negative shrinks).

    Args:
        mask: Integer / boolean label array in ``(z, y, x)`` order.
        grow_slices: Number of slices to append at each occupied end by
            copying the nearest occupied slice's labels.
        shrink_slices: Number of occupied slices to remove at each end.
        rng: Random generator enabling random per-end counts; requires
            ``max_slices > 0``.
        max_slices: Bound for the random per-end slice count.

    Returns:
        A new label array, same shape and dtype as ``mask``.

    Raises:
        ValueError: If shrinking removes every occupied slice, or if both
            fixed and random modes are mixed inconsistently.
    """
    labels = np.asarray(mask)
    result = np.array(labels, copy=True)
    union = labels != 0
    if not union.any():
        return result
    occupied = np.flatnonzero(union.any(axis=(1, 2)))
    first, last = int(occupied[0]), int(occupied[-1])

    if rng is not None:
        if max_slices <= 0:
            raise ValueError(
                "slice_extent_perturbation: random mode needs max_slices > 0."
            )
        grow_slices = 0
        shrink_slices = 0
        start_delta = int(rng.integers(-max_slices, max_slices + 1))
        end_delta = int(rng.integers(-max_slices, max_slices + 1))
    else:
        start_delta = int(grow_slices) - int(shrink_slices)
        end_delta = int(grow_slices) - int(shrink_slices)

    # Shrink (negative delta removes occupied slices from that end).
    if start_delta < 0:
        remove = min(-start_delta, last - first + 1)
        result[first : first + remove] = 0
    if end_delta < 0:
        remove = min(-end_delta, last - first + 1)
        result[last - remove + 1 : last + 1] = 0
    if not (result != 0).any():
        raise ValueError(
            "slice_extent_perturbation: shrinking removes every occupied slice; "
            "reduce the slice count."
        )
    # Grow (positive delta copies the nearest occupied slice outward).
    if start_delta > 0 and first > 0:
        source = result[first]
        for offset in range(1, start_delta + 1):
            target_index = first - offset
            if target_index < 0:
                break
            result[target_index] = source
    if end_delta > 0 and last < labels.shape[0] - 1:
        source = result[last]
        for offset in range(1, end_delta + 1):
            target_index = last + offset
            if target_index >= labels.shape[0]:
                break
            result[target_index] = source
    return result
