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
Gaussian noise addition, sub-voxel translation, small-angle rotation, and
noise-level estimation. This is the perturbation family Prior et al. used
to probe voxel-radiomics repeatability (Radiol Artif Intell 2024;6(2):
e230118, Appendix E2), implemented natively so HABIT does not depend on
MIRP (EUPL-1.2 license, Python >= 3.11 -- both incompatible with HABIT).

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
