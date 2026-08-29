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
"""Built-in image perturbations: the simulated test-retest family.

The three components mirror the perturbation chain of Prior et al. (Radiol
Artif Intell 2024;6(2):e230118, Appendix S2) -- Gaussian noise, sub-voxel
translation, small-angle rotation -- implemented on the L0 kernels of
:mod:`habit.kernels.image_perturbation`. Each one maps a
:class:`~habit.contracts.subject.Subject` to a perturbed copy on the SAME
voxel grid, so perturbed feature maps stay comparable to the original
voxel-by-voxel.

Prior's published extractor (radiomicsgroup/precise-habitats
``compute_features_parallel_perturbed.py``) pairs the perturbed CT with
the original ROI file. Geometric steps therefore keep the original
masks unless ``warp_masks=True`` (Zwanenburg / MIRP "patient moved"
style). :func:`prior2024_retest_perturbation` uses the paper default.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from habit.contracts.image import ArrayImageRef, ImageRef
from habit.contracts.subject import Subject
from habit.domain.precision.registry import ImagePerturbationRegistry
from habit.exceptions import HABITAPIError
from habit.kernels.image_perturbation import (
    add_gaussian_noise,
    boundary_weighted_perturbation,
    estimate_noise_sigma,
    morphological_grow_shrink,
    rigid_transform_image,
    rotate_image,
    slice_extent_perturbation,
    translate_image,
)
from habit.spec.specs import Spec

__all__ = [
    "BSplineDeformPerturbation",
    "BSplineDeformPerturbationParams",
    "GaussianNoisePerturbation",
    "GaussianNoisePerturbationParams",
    "TranslationPerturbation",
    "TranslationPerturbationParams",
    "RotationPerturbation",
    "RotationPerturbationParams",
    "RigidPerturbation",
    "RigidPerturbationParams",
    "MorphologicalPerturbation",
    "MorphologicalPerturbationParams",
    "GradientWeightedPerturbation",
    "GradientWeightedPerturbationParams",
    "SliceExtentPerturbation",
    "SliceExtentPerturbationParams",
    "prior2024_retest_perturbation",
]

#: Noise estimation methods accepted by :class:`GaussianNoisePerturbation`.
_NOISE_METHODS = ("chang", "roi_std")


def _sitk_image(array: np.ndarray, geometry: Any) -> Any:
    """
    Convert a contract array plus geometry to a float64 SimpleITK image.

    Interpolation during resampling must not quantise back to integer
    intensities, so images are always promoted to float64 first.

    Args:
        array: Voxel values, NumPy axis order ``(z, y, x)``.
        geometry: Spatial definition of ``array``.

    Returns:
        A ``SimpleITK.Image`` carrying the geometry metadata.
    """
    from habit.domain.habitat_features._radiomics import sitk_image_from_contract

    return sitk_image_from_contract(np.asarray(array, dtype=np.float64), geometry)


def _replace_images(
    subject: Subject,
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
) -> Subject:
    """
    Rebuild a subject with perturbed arrays wrapped in memory references.

    Args:
        subject: Source subject; its metadata is carried over unchanged.
        images: Modality key to perturbed array; empty keeps the originals.
        masks: ROI key to perturbed label array; empty keeps the originals.

    Returns:
        The perturbed subject copy.
    """
    new_images: Dict[str, ImageRef] = dict(subject.images)
    for modality, array in images.items():
        new_images[modality] = ArrayImageRef(
            array=array, geometry=subject.images[modality].geometry
        )
    new_masks: Dict[str, ImageRef] = dict(subject.masks)
    for roi, array in masks.items():
        new_masks[roi] = ArrayImageRef(
            array=array, geometry=subject.masks[roi].geometry
        )
    return dataclasses.replace(subject, images=new_images, masks=new_masks)


def _geometric_transform(
    subject: Subject,
    transform,
    interpolator: str,
    *,
    warp_masks: bool = True,
) -> Subject:
    """
    Apply one geometric kernel to every image, and optionally every mask.

    Args:
        subject: Source subject.
        transform: Kernel (``translate_image`` / ``rotate_image``) called as
            ``transform(sitk_image, interpolator=..., default_value=...)``.
        interpolator: Interpolator for the intensity images; masks always
            use nearest neighbour when ``warp_masks`` is True.
        warp_masks: When True, apply the same rigid move to every ROI
            (nearest neighbour). When False, keep the original masks so
            the voxel list stays the Prior 2024 GitHub pairing (perturbed
            image, original ROI).

    Returns:
        The perturbed subject copy.
    """
    import SimpleITK as sitk

    images: Dict[str, np.ndarray] = {}
    for modality in subject.images:
        volume = subject.image(modality)
        moved = transform(
            _sitk_image(volume.data, volume.geometry),
            interpolator=interpolator,
            default_value=0.0,
        )
        images[modality] = sitk.GetArrayFromImage(moved)
    if not warp_masks:
        # Empty mask map: _replace_images keeps subject.masks unchanged.
        return _replace_images(subject, images, {})
    masks: Dict[str, np.ndarray] = {}
    for roi in subject.masks:
        mask = subject.mask(roi)
        mask_array = np.asarray(mask.data)
        moved = transform(
            _sitk_image(mask_array, mask.geometry),
            interpolator="nearest",
            default_value=0.0,
        )
        # rint absorbs nearest-neighbour float dust; the label set is
        # unchanged by construction.
        masks[roi] = np.rint(sitk.GetArrayFromImage(moved)).astype(mask_array.dtype)
    return _replace_images(subject, images, masks)


def _as_numpy_volume(value: Any) -> np.ndarray:
    """
    Convert a MONAI transform output to a ``(z, y, x)`` NumPy volume.

    Args:
        value: Tensor or ndarray, either ``(z, y, x)`` or channel-first
            ``(1, z, y, x)``.

    Returns:
        A 3-D NumPy array.

    Raises:
        HABITAPIError: When the rank is not a single 3-D volume.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 3:
        raise HABITAPIError(
            "bspline_deform: expected a 3-D volume or a single-channel "
            f"(1, z, y, x) array; got shape {tuple(array.shape)}."
        )
    return array


def _pair_range(name: str, values: Sequence[float]) -> Tuple[float, float]:
    """
    Validate a two-element numeric range ``(low, high)`` with ``low <= high``.

    Args:
        name: Parameter name for the error message.
        values: Candidate pair.

    Returns:
        The pair as floats.

    Raises:
        HABITAPIError: When the length is not 2, a value is negative, or
            ``low > high``.
    """
    pair = tuple(float(v) for v in values)
    if len(pair) != 2:
        raise HABITAPIError(
            f"bspline_deform: {name} must be a (low, high) pair; got {values}."
        )
    if pair[0] < 0.0 or pair[1] < 0.0:
        raise HABITAPIError(
            f"bspline_deform: {name} values must be >= 0; got {pair}."
        )
    if pair[0] > pair[1]:
        raise HABITAPIError(
            f"bspline_deform: {name} low must be <= high; got {pair}."
        )
    return (pair[0], pair[1])


def _resample_mode(name: str, mode: Union[str, int]) -> Union[str, int]:
    """
    Validate a MONAI ``Rand3DElastic`` interpolator.

    Args:
        name: Parameter name for the error message.
        mode: ``"bilinear"`` / ``"nearest"`` or spline order 0–5.

    Returns:
        The validated mode.

    Raises:
        HABITAPIError: When ``mode`` is not an allowed interpolator.
    """
    if isinstance(mode, bool):
        raise HABITAPIError(
            f"bspline_deform: {name} must be 'bilinear', 'nearest', or "
            f"an integer spline order 0..5; got {mode!r}."
        )
    if isinstance(mode, int):
        if mode not in range(6):
            raise HABITAPIError(
                f"bspline_deform: {name} spline order must be in 0..5; "
                f"got {mode}."
            )
        return int(mode)
    text = str(mode)
    if text not in {"bilinear", "nearest"}:
        raise HABITAPIError(
            f"bspline_deform: {name} must be 'bilinear', 'nearest', or "
            f"an integer spline order 0..5; got {mode!r}."
        )
    return text


class GaussianNoisePerturbationParams(BaseModel):
    """Constructor parameters for :class:`GaussianNoisePerturbation`."""

    model_config = ConfigDict(extra="forbid")
    sigma: Optional[float] = Field(default=None, ge=0.0)
    noise_method: str = "chang"
    roi: Optional[str] = None
    round_to_int: bool = False


@ImagePerturbationRegistry.register("gaussian_noise")
class GaussianNoisePerturbation:
    """
    Add zero-mean Gaussian noise to every image of a subject.

    Args:
        sigma: Noise standard deviation in intensity units; ``None``
            estimates it per subject with ``noise_method`` (the paper's
            choice, MIRP's behaviour when no level is configured).
        noise_method: ``"chang"`` (wavelet estimator) or ``"roi_std"``
            (standard deviation inside the ROI).
        roi: Mask key for ``roi_std`` estimation; ``None`` uses the
            subject's single mask.
        round_to_int: Round the noisy image to whole numbers, mirroring
            MIRP's handling of integer-valued CT (HU) data.
    """

    def __init__(
        self,
        sigma: Optional[float] = None,
        noise_method: str = "chang",
        roi: Optional[str] = None,
        round_to_int: bool = False,
    ) -> None:
        if noise_method not in _NOISE_METHODS:
            raise HABITAPIError(
                f"gaussian_noise: noise_method must be one of {_NOISE_METHODS}; "
                f"got {noise_method!r}."
            )
        if sigma is not None and sigma < 0.0:
            raise HABITAPIError(f"gaussian_noise: sigma must be >= 0; got {sigma}.")
        self.sigma = None if sigma is None else float(sigma)
        self.noise_method = noise_method
        self.roi = roi
        self.round_to_int = bool(round_to_int)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="gaussian_noise",
            params={
                "sigma": self.sigma,
                "noise_method": self.noise_method,
                "roi": self.roi,
                "round_to_int": self.round_to_int,
            },
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` with Gaussian noise added to all images.

        Args:
            subject: Subject providing the images (and the ROI when the
                noise level is estimated with ``roi_std``).
            rng: Random generator supplying the noise field.

        Returns:
            The perturbed subject copy.
        """
        mask_array: Optional[np.ndarray] = None
        if self.sigma is None and self.noise_method == "roi_std":
            mask_array = np.asarray(subject.mask(self.roi).data) > 0
        images: Dict[str, np.ndarray] = {}
        for modality in subject.images:
            volume = subject.image(modality)
            array = np.asarray(volume.data)
            sigma = self.sigma
            if sigma is None:
                sigma = estimate_noise_sigma(array, mask_array, self.noise_method)
            images[modality] = add_gaussian_noise(
                array, sigma, rng, round_to_int=self.round_to_int
            )
        return _replace_images(subject, images, {})


class TranslationPerturbationParams(BaseModel):
    """Constructor parameters for :class:`TranslationPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    shift_voxels: Optional[Tuple[float, float, float]] = None
    shift_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_shift_voxels: float = Field(default=1.0, ge=0.0)
    random_signs: bool = True
    interpolator: str = "bspline"
    warp_masks: bool = True


@ImagePerturbationRegistry.register("translation")
class TranslationPerturbation:
    """
    Translate image content by a (random) sub-voxel shift.

    Matches MIRP ``perturbation_translation_fraction`` when
    ``shift_fraction`` is set: a fraction of one voxel along x, y and z
    (Prior et al., Radiol Artif Intell 2024;6(2):e230118, Appendix S2).
    ``shift_voxels`` is the explicit SimpleITK ``(x, y, z)`` alternative.
    When both are unset, each axis is drawn from
    ``Uniform(-max_shift_voxels, +max_shift_voxels)``.

    Args:
        shift_voxels: Fixed shift in voxel units, SimpleITK ``(x, y, z)``
            order; ``None`` defers to ``shift_fraction`` or random sampling.
        shift_fraction: MIRP-style fraction of a voxel in ``[0, 1]``. When
            set (and ``shift_voxels`` is unset), the shift is
            ``±fraction`` on each axis (signs random if ``random_signs``).
        max_shift_voxels: Sampling bound when neither fixed shift is set.
        random_signs: When using ``shift_fraction``, randomize the sign of
            each axis (MIRP interpolates at a shifted grid; the direction
            of the shift is not anatomically privileged).
        interpolator: Interpolator for the intensity images (``"bspline"``
            is the paper's choice); masks use nearest neighbour only when
            ``warp_masks`` is True.
        warp_masks: When True, apply the same translation to every ROI.
            Prior 2024 extraction keeps the original mask (False).
    """

    def __init__(
        self,
        shift_voxels: Optional[Sequence[float]] = None,
        max_shift_voxels: float = 1.0,
        interpolator: str = "bspline",
        shift_fraction: Optional[float] = None,
        random_signs: bool = True,
        warp_masks: bool = True,
    ) -> None:
        if shift_voxels is not None and len(tuple(shift_voxels)) != 3:
            raise HABITAPIError(
                "translation: shift_voxels must have 3 components (x, y, z); "
                f"got {tuple(shift_voxels)}."
            )
        if shift_voxels is not None and shift_fraction is not None:
            raise HABITAPIError(
                "translation: pass shift_voxels or shift_fraction, not both."
            )
        if shift_fraction is not None and not (0.0 <= float(shift_fraction) <= 1.0):
            raise HABITAPIError(
                "translation: shift_fraction must be in [0, 1] "
                f"(MIRP perturbation_translation_fraction); got {shift_fraction}."
            )
        if max_shift_voxels < 0.0:
            raise HABITAPIError(
                f"translation: max_shift_voxels must be >= 0; got {max_shift_voxels}."
            )
        self.shift_voxels = (
            None if shift_voxels is None else tuple(float(v) for v in shift_voxels)
        )
        self.shift_fraction = (
            None if shift_fraction is None else float(shift_fraction)
        )
        self.max_shift_voxels = float(max_shift_voxels)
        self.random_signs = bool(random_signs)
        self.interpolator = str(interpolator)
        self.warp_masks = bool(warp_masks)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="translation",
            params={
                "shift_voxels": self.shift_voxels,
                "shift_fraction": self.shift_fraction,
                "max_shift_voxels": self.max_shift_voxels,
                "random_signs": self.random_signs,
                "interpolator": self.interpolator,
                "warp_masks": self.warp_masks,
            },
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` translated by the (sampled) shift.

        Args:
            subject: Subject providing images and masks.
            rng: Random generator sampling the shift when ``shift_voxels``
                is unset.

        Returns:
            The perturbed subject copy.
        """
        if self.shift_voxels is not None:
            shift = self.shift_voxels
        elif self.shift_fraction is not None:
            fraction = self.shift_fraction
            if self.random_signs:
                signs = rng.choice(np.array([-1.0, 1.0]), size=3)
            else:
                signs = np.ones(3, dtype=np.float64)
            shift = tuple(float(fraction * s) for s in signs)
        else:
            shift = tuple(
                rng.uniform(-self.max_shift_voxels, self.max_shift_voxels, size=3)
            )
        return _geometric_transform(
            subject,
            lambda image, interpolator, default_value: translate_image(
                image, shift, interpolator=interpolator, default_value=default_value
            ),
            self.interpolator,
            warp_masks=self.warp_masks,
        )


class RotationPerturbationParams(BaseModel):
    """Constructor parameters for :class:`RotationPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    angle_degrees: float = 0.5
    axis: str = "z"
    interpolator: str = "bspline"
    random_sign: bool = False
    warp_masks: bool = True


@ImagePerturbationRegistry.register("rotation")
class RotationPerturbation:
    """
    Rotate image content by a small fixed angle about the image centre.

    Args:
        angle_degrees: Rotation angle in degrees; the paper uses ``0.5``.
            Deterministic by design -- pass the sign you want.
        axis: Axis to rotate around (``"x"``, ``"y"`` or ``"z"``; ``"z"``
            is the axial in-plane axis, the paper's choice).
        interpolator: Interpolator for the intensity images (``"bspline"``
            is the paper's choice); masks use nearest neighbour only when
            ``warp_masks`` is True.
        random_sign: When ``True``, the sign of ``angle_degrees`` is drawn
            as ``±1`` per call (some MIRP configs randomize the sense of
            the 0.5° in-plane rotation). The paper's default is a fixed
            ``+0.5`` degrees, so this stays ``False``.
        warp_masks: When True, apply the same rotation to every ROI.
            Prior 2024 extraction keeps the original mask (False).
    """

    def __init__(
        self,
        angle_degrees: float = 0.5,
        axis: str = "z",
        interpolator: str = "bspline",
        random_sign: bool = False,
        warp_masks: bool = True,
    ) -> None:
        if axis not in ("x", "y", "z"):
            raise HABITAPIError(
                f"rotation: axis must be one of ('x', 'y', 'z'); got {axis!r}."
            )
        self.angle_degrees = float(angle_degrees)
        self.axis = axis
        self.interpolator = str(interpolator)
        self.random_sign = bool(random_sign)
        self.warp_masks = bool(warp_masks)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="rotation",
            params={
                "angle_degrees": self.angle_degrees,
                "axis": self.axis,
                "interpolator": self.interpolator,
                "random_sign": self.random_sign,
                "warp_masks": self.warp_masks,
            },
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` rotated by ``angle_degrees``.

        Args:
            subject: Subject providing images and masks.
            rng: Random generator; consumed only when ``random_sign`` is True.

        Returns:
            The perturbed subject copy.
        """
        angle = self.angle_degrees
        if self.random_sign:
            angle = float(angle * rng.choice(np.array([-1.0, 1.0])))
        return _geometric_transform(
            subject,
            lambda image, interpolator, default_value: rotate_image(
                image,
                angle,
                axis=self.axis,
                interpolator=interpolator,
                default_value=default_value,
            ),
            self.interpolator,
            warp_masks=self.warp_masks,
        )


class RigidPerturbationParams(BaseModel):
    """Constructor parameters for :class:`RigidPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    shift_voxels: Optional[Tuple[float, float, float]] = None
    shift_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    random_signs: bool = True
    angle_degrees: float = 0.5
    axis: str = "z"
    interpolator: str = "bspline"
    random_sign: bool = False
    warp_masks: bool = True


@ImagePerturbationRegistry.register("rigid")
class RigidPerturbation:
    """
    Sub-voxel translation and small-angle rotation in ONE resample.

    MIRP ≥ 2 composes the geometric pair into a single affine so the
    image is not B-spline interpolated twice. Prior et al. 2024 used
    MIRP 1.2.0 (two resamples); use :func:`prior2024_retest_perturbation`
    with ``single_resample=False`` (default) to match that paper, or
    ``single_resample=True`` for this component.

    Args:
        shift_voxels: Fixed voxel shift; ``None`` uses ``shift_fraction``.
        shift_fraction: MIRP ``perturbation_translation_fraction`` in
            ``[0, 1]`` (paper-style default 0.5).
        random_signs: Randomize the sign of each translation axis.
        angle_degrees: In-plane rotation in degrees (paper: 0.5).
        axis: Rotation axis (paper: ``"z"``).
        interpolator: Intensity interpolator (paper: ``"bspline"``).
        random_sign: Randomize the rotation sense.
        warp_masks: When True, apply the same rigid move to every ROI.
            Prior 2024 extraction keeps the original mask (False).
    """

    def __init__(
        self,
        shift_voxels: Optional[Sequence[float]] = None,
        shift_fraction: float = 0.5,
        random_signs: bool = True,
        angle_degrees: float = 0.5,
        axis: str = "z",
        interpolator: str = "bspline",
        random_sign: bool = False,
        warp_masks: bool = True,
    ) -> None:
        if shift_voxels is not None and len(tuple(shift_voxels)) != 3:
            raise HABITAPIError(
                "rigid: shift_voxels must have 3 components (x, y, z); "
                f"got {tuple(shift_voxels)}."
            )
        if not (0.0 <= float(shift_fraction) <= 1.0):
            raise HABITAPIError(
                "rigid: shift_fraction must be in [0, 1]; "
                f"got {shift_fraction}."
            )
        if axis not in ("x", "y", "z"):
            raise HABITAPIError(
                f"rigid: axis must be one of ('x', 'y', 'z'); got {axis!r}."
            )
        self.shift_voxels = (
            None if shift_voxels is None else tuple(float(v) for v in shift_voxels)
        )
        self.shift_fraction = float(shift_fraction)
        self.random_signs = bool(random_signs)
        self.angle_degrees = float(angle_degrees)
        self.axis = axis
        self.interpolator = str(interpolator)
        self.random_sign = bool(random_sign)
        self.warp_masks = bool(warp_masks)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="rigid",
            params={
                "shift_voxels": self.shift_voxels,
                "shift_fraction": self.shift_fraction,
                "random_signs": self.random_signs,
                "angle_degrees": self.angle_degrees,
                "axis": self.axis,
                "interpolator": self.interpolator,
                "random_sign": self.random_sign,
                "warp_masks": self.warp_masks,
            },
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` after one rigid resample.

        Args:
            subject: Subject providing images and masks.
            rng: Random generator for optional sign draws.

        Returns:
            The perturbed subject copy.
        """
        if self.shift_voxels is not None:
            shift = self.shift_voxels
        else:
            if self.random_signs:
                signs = rng.choice(np.array([-1.0, 1.0]), size=3)
            else:
                signs = np.ones(3, dtype=np.float64)
            shift = tuple(float(self.shift_fraction * s) for s in signs)
        angle = self.angle_degrees
        if self.random_sign:
            angle = float(angle * rng.choice(np.array([-1.0, 1.0])))
        return _geometric_transform(
            subject,
            lambda image, interpolator, default_value: rigid_transform_image(
                image,
                shift,
                angle,
                axis=self.axis,
                interpolator=interpolator,
                default_value=default_value,
            ),
            self.interpolator,
            warp_masks=self.warp_masks,
        )


def _perturb_masks(
    subject: Subject,
    fn,
) -> Subject:
    """
    Apply a mask-array kernel to every ROI of a subject, images untouched.

    Contour-variability perturbations change where the boundary lies, not
    the underlying intensities, so only the masks are transformed.

    Args:
        subject: Source subject.
        fn: Callable ``fn(mask_array, spacing_xyz) -> new_mask_array``.

    Returns:
        The perturbed subject copy with masks replaced.
    """
    masks: Dict[str, np.ndarray] = {}
    for roi in subject.masks:
        mask = subject.mask(roi)
        mask_array = np.asarray(mask.data)
        spacing = tuple(float(v) for v in mask.geometry.spacing)
        masks[roi] = fn(mask_array, spacing).astype(mask_array.dtype, copy=False)
    return _replace_images(subject, {}, masks)


class MorphologicalPerturbationParams(BaseModel):
    """Constructor parameters for :class:`MorphologicalPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    grow_mm: Optional[float] = None
    max_grow_mm: float = Field(default=1.0, ge=0.0)
    roi: Optional[str] = None
    connectivity: int = Field(default=1, ge=1, le=3)


@ImagePerturbationRegistry.register("morphological")
class MorphologicalPerturbation:
    """
    Uniformly grow or shrink every ROI (MIRP ``perturbation_roi_adapt_size``).

    This is the systematic component of inter-rater contour variability:
    one observer consistently traces slightly larger or smaller than
    another. It complements the Prior 2024 simulated-retest chain (which
    perturbs the *image*, not the contour). Only masks change; image
    intensities are untouched. Applied per foreground label so multi-label
    ROIs grow each region instead of merging them.

    Args:
        grow_mm: Fixed physical radius in millimetres; positive dilates,
            negative erodes, zero is a no-op. ``None`` samples a signed
            radius from ``Uniform(-max_grow_mm, +max_grow_mm)`` per call.
        max_grow_mm: Sampling bound when ``grow_mm`` is unset.
        roi: Restrict the perturbation to one mask key; ``None`` perturbs
            all masks.
        connectivity: Structuring-element connectivity in ``{1, 2, 3}``;
            ``1`` (6-connected) is the MIRP-like default.
    """

    def __init__(
        self,
        grow_mm: Optional[float] = None,
        max_grow_mm: float = 1.0,
        roi: Optional[str] = None,
        connectivity: int = 1,
    ) -> None:
        if max_grow_mm < 0.0:
            raise HABITAPIError(
                f"morphological: max_grow_mm must be >= 0; got {max_grow_mm}."
            )
        if connectivity not in (1, 2, 3):
            raise HABITAPIError(
                f"morphological: connectivity must be in {{1,2,3}}; got {connectivity}."
            )
        self.grow_mm = None if grow_mm is None else float(grow_mm)
        self.max_grow_mm = float(max_grow_mm)
        self.roi = roi
        self.connectivity = int(connectivity)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="morphological",
            params={
                "grow_mm": self.grow_mm,
                "max_grow_mm": self.max_grow_mm,
                "roi": self.roi,
                "connectivity": self.connectivity,
            },
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` with each ROI grown or shrunk.

        Args:
            subject: Subject providing the masks.
            rng: Random generator sampling the radius when ``grow_mm`` is
                unset.

        Returns:
            The perturbed subject copy.
        """
        grow_mm = self.grow_mm
        if grow_mm is None:
            grow_mm = float(
                rng.uniform(-self.max_grow_mm, self.max_grow_mm)
            )
        radius = grow_mm

        def _fn(mask_array: np.ndarray, spacing) -> np.ndarray:
            return morphological_grow_shrink(
                mask_array, radius, spacing_xyz=spacing,
                connectivity=self.connectivity,
            )

        if self.roi is not None:
            masks = {
                self.roi: _fn(
                    np.asarray(subject.mask(self.roi).data),
                    tuple(float(v) for v in subject.mask(self.roi).geometry.spacing),
                )
            }
            return _replace_images(subject, {}, masks)
        return _perturb_masks(subject, _fn)


class GradientWeightedPerturbationParams(BaseModel):
    """Constructor parameters for :class:`GradientWeightedPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    modality: Optional[str] = None
    roi: Optional[str] = None
    max_radius_voxels: int = Field(default=2, ge=1)
    probability: float = Field(default=0.5, ge=0.0, le=1.0)


@ImagePerturbationRegistry.register("gradient_weighted")
class GradientWeightedPerturbation:
    """
    Locally grow/shrink ROI boundaries where image gradient is low.

    Inter-rater disagreement concentrates where contrast is poor: sharp
    (high-gradient) edges are drawn consistently, fuzzy (low-gradient)
    edges vary. This operator flips boundary voxels with a probability that
    scales with ``1 - normalised_gradient`` of a reference image, so the
    fuzzy parts of the contour move more than the sharp parts. Only masks
    change.

    Args:
        modality: Image modality supplying the gradient-magnitude map;
            ``None`` uses the subject's first image. The map is normalised
            to ``[0, 1]`` over the ROI bounding region.
        roi: Restrict the perturbation to one mask key; ``None`` perturbs
            all masks.
        max_radius_voxels: Neighbourhood radius bounding each local flip.
        probability: Base flip probability at zero gradient; effective
            probability is ``probability * (1 - gradient)``.
    """

    def __init__(
        self,
        modality: Optional[str] = None,
        roi: Optional[str] = None,
        max_radius_voxels: int = 2,
        probability: float = 0.5,
    ) -> None:
        if max_radius_voxels < 1:
            raise HABITAPIError(
                "gradient_weighted: max_radius_voxels must be >= 1; "
                f"got {max_radius_voxels}."
            )
        if not (0.0 <= float(probability) <= 1.0):
            raise HABITAPIError(
                f"gradient_weighted: probability must be in [0, 1]; got {probability}."
            )
        self.modality = modality
        self.roi = roi
        self.max_radius_voxels = int(max_radius_voxels)
        self.probability = float(probability)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="gradient_weighted",
            params={
                "modality": self.modality,
                "roi": self.roi,
                "max_radius_voxels": self.max_radius_voxels,
                "probability": self.probability,
            },
        )

    def _gradient_weights(self, subject: Subject) -> np.ndarray:
        """
        Return the normalised gradient magnitude of the reference image.

        Args:
            subject: Subject providing the reference image.

        Returns:
            A ``float64`` map in ``[0, 1]``; high at sharp edges.

        Raises:
            HABITAPIError: When the subject has no images.
        """
        from scipy import ndimage as _ndi

        if not list(subject.images):
            raise HABITAPIError(
                "gradient_weighted: subject has no images to derive a "
                "gradient map from."
            )
        modality = self.modality or next(iter(subject.images))
        image = np.asarray(subject.image(modality).data, dtype=np.float64)
        gradient = _ndi.gaussian_gradient_magnitude(image, sigma=1.0)
        peak = float(gradient.max())
        if peak <= 0.0:
            return np.zeros_like(gradient)
        return gradient / peak

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` with ROI boundaries locally perturbed.

        Args:
            subject: Subject providing images (for the gradient) and masks.
            rng: Random generator supplying the flip decisions.

        Returns:
            The perturbed subject copy.
        """
        weights = self._gradient_weights(subject)

        def _fn(mask_array: np.ndarray, spacing) -> np.ndarray:
            return boundary_weighted_perturbation(
                mask_array,
                weights,
                rng,
                max_radius_voxels=self.max_radius_voxels,
                probability=self.probability,
            )

        if self.roi is not None:
            masks = {
                self.roi: _fn(
                    np.asarray(subject.mask(self.roi).data),
                    tuple(float(v) for v in subject.mask(self.roi).geometry.spacing),
                )
            }
            return _replace_images(subject, {}, masks)
        return _perturb_masks(subject, _fn)


class SliceExtentPerturbationParams(BaseModel):
    """Constructor parameters for :class:`SliceExtentPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    grow_slices: int = Field(default=0, ge=0)
    shrink_slices: int = Field(default=0, ge=0)
    max_slices: int = Field(default=0, ge=0)
    roi: Optional[str] = None


@ImagePerturbationRegistry.register("slice_extent")
class SliceExtentPerturbation:
    """
    Add or remove whole axial slices at the superior/inferior ROI ends.

    Models z-axis delineation variability: observers often agree in-plane
    but differ on the first and last slice they call tumour. Only the ``z``
    (first) axis is touched. Only masks change.

    Provide fixed ``grow_slices`` / ``shrink_slices`` (applied to each end),
    or set ``max_slices > 0`` to draw a random per-end count in
    ``[-max_slices, +max_slices]`` (positive grows, negative shrinks). When
    ``max_slices`` is set the fixed counts are ignored.

    Args:
        grow_slices: Slices to append at each occupied end (copy of the
            nearest occupied slice's labels).
        shrink_slices: Occupied slices to remove at each end.
        max_slices: Bound for random per-end counts; ``0`` uses the fixed
            counts.
        roi: Restrict the perturbation to one mask key; ``None`` perturbs
            all masks.
    """

    def __init__(
        self,
        grow_slices: int = 0,
        shrink_slices: int = 0,
        max_slices: int = 0,
        roi: Optional[str] = None,
    ) -> None:
        for name, value in (
            ("grow_slices", grow_slices),
            ("shrink_slices", shrink_slices),
            ("max_slices", max_slices),
        ):
            if int(value) < 0:
                raise HABITAPIError(
                    f"slice_extent: {name} must be >= 0; got {value}."
                )
        self.grow_slices = int(grow_slices)
        self.shrink_slices = int(shrink_slices)
        self.max_slices = int(max_slices)
        self.roi = roi

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="slice_extent",
            params={
                "grow_slices": self.grow_slices,
                "shrink_slices": self.shrink_slices,
                "max_slices": self.max_slices,
                "roi": self.roi,
            },
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` with ROI slice extents perturbed.

        Args:
            subject: Subject providing the masks.
            rng: Random generator for random mode (``max_slices > 0``).

        Returns:
            The perturbed subject copy.
        """
        use_random = self.max_slices > 0

        def _fn(mask_array: np.ndarray, spacing) -> np.ndarray:
            if use_random:
                return slice_extent_perturbation(
                    mask_array, rng=rng, max_slices=self.max_slices
                )
            return slice_extent_perturbation(
                mask_array,
                grow_slices=self.grow_slices,
                shrink_slices=self.shrink_slices,
            )

        if self.roi is not None:
            masks = {
                self.roi: _fn(
                    np.asarray(subject.mask(self.roi).data),
                    tuple(float(v) for v in subject.mask(self.roi).geometry.spacing),
                )
            }
            return _replace_images(subject, {}, masks)
        return _perturb_masks(subject, _fn)


def prior2024_retest_perturbation(    *,
    shift_fraction: float = 0.5,
    angle_degrees: float = 0.5,
    interpolator: str = "bspline",
    single_resample: bool = False,
    warp_masks: bool = False,
) -> "PerturbationChain":
    """
    Simulated-retest chain of Prior et al. 2024 / MIRP 1.2.0 Appendix S2.

    Paper: Prior O, et al. Identification of Precise 3D CT Radiomics for
    Habitat Computation by Machine Learning in Cancer. Radiol Artif Intell.
    2024;6(2):e230118. doi:10.1148/ryai.230118

    Order: Gaussian noise (Chang wavelet sigma) → sub-voxel translation
    (fraction ``η`` of voxel spacing, default 0.5, random axis signs) →
    0.5° in-plane (z) rotation. Images use B-spline.
    ``single_resample=True`` composes translation+rotation (MIRP ≥ 2);
    the paper used two geometric resamples.

    Default ``warp_masks=False`` matches their GitHub extractor: the
    perturbed CT is paired with the original ROI
    (``compute_features_parallel_perturbed.py``). Set ``warp_masks=True``
    for the Zwanenburg / MIRP "image and mask move together" variant.

    ROI morphological grow/shrink (MIRP ``perturbation_roi_adapt_size``)
    is not in this protocol. MONAI elastic / B-spline free-form warps
    are a separate optional component
    (:class:`BSplineDeformPerturbation`), not part of this chain.

    Args:
        shift_fraction: MIRP translation fraction in ``[0, 1]``.
        angle_degrees: In-plane rotation in degrees.
        interpolator: Intensity interpolator.
        single_resample: Compose translation and rotation into one affine.
        warp_masks: When True, nearest-neighbour warp every ROI with the
            image. Paper / GitHub default is False.

    Returns:
        A :class:`~habit.domain.precision.PerturbationChain`.
    """
    from habit.domain.precision.chain import PerturbationChain

    noise = GaussianNoisePerturbation()
    if single_resample:
        return PerturbationChain(
            [
                noise,
                RigidPerturbation(
                    shift_fraction=shift_fraction,
                    angle_degrees=angle_degrees,
                    interpolator=interpolator,
                    warp_masks=warp_masks,
                ),
            ]
        )
    return PerturbationChain(
        [
            noise,
            TranslationPerturbation(
                shift_fraction=shift_fraction,
                interpolator=interpolator,
                warp_masks=warp_masks,
            ),
            RotationPerturbation(
                angle_degrees=angle_degrees,
                interpolator=interpolator,
                warp_masks=warp_masks,
            ),
        ]
    )


def _scipy_spline_order(mode: Union[str, int]) -> int:
    """Map a MONAI-style interpolator name to a scipy spline order."""
    if isinstance(mode, int):
        return int(mode)
    mapping = {
        "nearest": 0,
        "bilinear": 1,
        "linear": 1,
        "bspline": 3,
        "bicubic": 3,
    }
    return int(mapping.get(str(mode), 1))


def _fit_displacement_shape(
    field: np.ndarray, shape: Tuple[int, int, int]
) -> np.ndarray:
    """Crop or pad a ``(3, ...)`` displacement so spatial axes match ``shape``."""
    out = np.zeros((3,) + shape, dtype=np.float64)
    sl = tuple(slice(0, min(int(field.shape[i + 1]), int(shape[i]))) for i in range(3))
    out[(slice(None),) + sl] = field[(slice(None),) + sl]
    return out


def _ffd_displacement(
    shape: Tuple[int, int, int],
    *,
    control_spacing: float,
    magnitude: float,
    seed: int,
) -> np.ndarray:
    """
    Build a Rueckert-style cubic B-spline FFD displacement.

    Random offsets live on a coarse control lattice (about one knot every
    ``control_spacing`` voxels). Cubic zoom to the full grid is what makes
    the contour a slow bulge instead of 1-voxel teeth.

    Args:
        shape: Full ``(z, y, x)`` grid.
        control_spacing: Voxels between neighbouring control points.
        magnitude: Peak displacement at a control point, in voxels.
        seed: Frozen RNG seed so image and mask share one field.

    Returns:
        ``(3, z, y, x)`` float64 displacement in voxel units.
    """
    from scipy.ndimage import zoom

    spacing = float(control_spacing)
    coarse = tuple(max(4, int(np.ceil(float(s) / spacing)) + 1) for s in shape)
    rng = np.random.default_rng(int(seed))
    coarse_field = rng.uniform(-1.0, 1.0, size=(3,) + coarse).astype(np.float64)
    factors = tuple(float(shape[i]) / float(coarse[i]) for i in range(3))
    field = np.stack(
        [zoom(coarse_field[c], factors, order=3, mode="nearest") for c in range(3)],
        axis=0,
    )
    return _fit_displacement_shape(field, shape) * float(magnitude)


def _warp_with_displacement(
    volume: np.ndarray,
    field: np.ndarray,
    *,
    order: int,
    padding_mode: str,
) -> np.ndarray:
    """Resample ``volume`` by ``field`` with scipy ``map_coordinates``."""
    from scipy.ndimage import map_coordinates

    mode_map = {
        "reflection": "reflect",
        "border": "nearest",
        "zeros": "constant",
    }
    coords = np.indices(volume.shape, dtype=np.float64) + field
    return map_coordinates(
        np.asarray(volume, dtype=np.float64),
        coords,
        order=int(order),
        mode=mode_map[padding_mode],
        cval=0.0,
        prefilter=True,
    )


class BSplineDeformPerturbationParams(BaseModel):
    """Constructor parameters for :class:`BSplineDeformPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    sigma_range: Tuple[float, float] = (1.5, 3.0)
    magnitude_range: Tuple[float, float] = (8.0, 12.0)
    image_mode: Union[str, int] = "bilinear"
    mask_mode: Union[str, int] = "nearest"
    padding_mode: str = "reflection"
    device: str = "cpu"
    target_dice: Optional[float] = None
    dice_tolerance: float = 0.02
    control_spacing: Optional[float] = None


@ImagePerturbationRegistry.register("bspline_deform")
class BSplineDeformPerturbation:
    """
    MONAI elastic / B-spline free-form warp of every image and ROI.

    This is **not** the Prior 2024 / MIRP 1.2.0 simulated-retest chain
    (noise → translation → rotation). It is the optional follow-up that
    actually changes ROI *shape*: one random 3-D displacement field is
    drawn and applied to every modality and mask of the subject so the
    contour and the anatomy stay paired.

    Implementation (MONAI ``Rand3DElasticd``, optional extra ``monai``):

    * A random offset grid is sampled, Gaussian-smoothed with a sigma
      drawn from ``sigma_range``, and scaled by a magnitude drawn from
      ``magnitude_range`` (voxel units). Optional extra affine terms are
      left off so this component is a pure elastic warp.
    * Intensities are resampled with ``image_mode`` (default
      ``"bilinear"``, MONAI's torch grid-sample path). Labels use
      ``mask_mode`` (default ``"nearest"``) so the ROI stays a discrete
      label. Integer 0–5 selects scipy spline order instead (order 3 is
      cubic B-spline resampling; it is much slower on full clinical
      volumes).
    * The output stays on the original grid (same shape and geometry).

    Default path is MONAI's documented 3-D elastic deformation (full-
    resolution random offsets + Gaussian), not Rueckert cubic-B-spline
    FFD. Pass ``control_spacing`` to switch to an explicit coarse
    control lattice with cubic interpolation — that is the smooth
    teaching warp (slow bulge, not 1-voxel teeth). Neither path is
    MIRP ``perturbation_roi_adapt_size`` (morphological grow/shrink).

    Args:
        sigma_range: Gaussian smoothing ``(low, high)`` of the offset
            grid, in voxels. Wider sigma gives a smoother warp. Ignored
            when ``control_spacing`` is set.
        magnitude_range: Displacement magnitude ``(low, high)`` in
            voxels. Larger values wrinkle the ROI more.
        image_mode: Intensity interpolator (``"bilinear"`` / ``"nearest"``
            or spline order 0–5).
        mask_mode: ROI interpolator (``"nearest"`` recommended for the
            MONAI path; ``"bilinear"`` then ``rint`` is a smoother
            iso-contour on the FFD path).
        padding_mode: Out-of-grid padding (``reflection``, ``border``,
            or ``zeros``).
        device: Torch device (``"cpu"`` / ``"cuda"``). Default ``"cpu"``
            is the portable path; pass ``"cuda"`` when a GPU is available
            and the volume fits in memory. Unused on the FFD path.
        target_dice: When set, scale one frozen offset field so the
            Dice between the original and warped ROI is within
            ``dice_tolerance`` of this value. ``None`` keeps the random
            magnitude from ``magnitude_range``.
        dice_tolerance: Allowed absolute error on ``target_dice``.
        control_spacing: When set, voxels between neighbouring FFD
            control points (must be ``> 1``). ``None`` keeps the MONAI
            ``Rand3DElasticd`` path.
    """

    def __init__(
        self,
        sigma_range: Sequence[float] = (1.5, 3.0),
        magnitude_range: Sequence[float] = (8.0, 12.0),
        image_mode: Union[str, int] = "bilinear",
        mask_mode: Union[str, int] = "nearest",
        padding_mode: str = "reflection",
        device: str = "cpu",
        target_dice: Optional[float] = None,
        dice_tolerance: float = 0.02,
        control_spacing: Optional[float] = None,
    ) -> None:
        self.sigma_range = _pair_range("sigma_range", sigma_range)
        self.magnitude_range = _pair_range("magnitude_range", magnitude_range)
        self.image_mode = _resample_mode("image_mode", image_mode)
        self.mask_mode = _resample_mode("mask_mode", mask_mode)
        allowed_pad = ("reflection", "border", "zeros")
        if padding_mode not in allowed_pad:
            raise HABITAPIError(
                "bspline_deform: padding_mode must be one of "
                f"{allowed_pad}; got {padding_mode!r}."
            )
        self.padding_mode = str(padding_mode)
        if str(device) not in {"cpu", "cuda"}:
            raise HABITAPIError(
                "bspline_deform: device must be 'cpu' or 'cuda'; "
                f"got {device!r}."
            )
        self.device = str(device)
        if target_dice is not None:
            target = float(target_dice)
            if not 0.0 < target <= 1.0:
                raise HABITAPIError(
                    "bspline_deform: target_dice must be in (0, 1]; "
                    f"got {target_dice!r}."
                )
            self.target_dice: Optional[float] = target
        else:
            self.target_dice = None
        tolerance = float(dice_tolerance)
        if not 0.0 < tolerance < 1.0:
            raise HABITAPIError(
                "bspline_deform: dice_tolerance must be in (0, 1); "
                f"got {dice_tolerance!r}."
            )
        self.dice_tolerance = tolerance
        if control_spacing is None:
            self.control_spacing: Optional[float] = None
        else:
            spacing = float(control_spacing)
            if not np.isfinite(spacing) or spacing <= 1.0:
                raise HABITAPIError(
                    "bspline_deform: control_spacing must be > 1 voxel "
                    f"when set; got {control_spacing!r}."
                )
            self.control_spacing = spacing

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="bspline_deform",
            params={
                "sigma_range": self.sigma_range,
                "magnitude_range": self.magnitude_range,
                "image_mode": self.image_mode,
                "mask_mode": self.mask_mode,
                "padding_mode": self.padding_mode,
                "device": self.device,
                "target_dice": self.target_dice,
                "dice_tolerance": self.dice_tolerance,
                "control_spacing": self.control_spacing,
            },
        )

    def _warp(
        self,
        subject: Subject,
        *,
        seed: int,
        magnitude_range: Tuple[float, float],
        mask_keys_only: bool = False,
    ) -> Subject:
        """
        Apply one MONAI elastic field with a frozen seed and magnitude.

        Args:
            subject: Subject providing images and/or masks on one grid.
            seed: Integer seed for the shared offset grid.
            magnitude_range: Displacement magnitude ``(low, high)``.
            mask_keys_only: When ``True``, warp masks only (used while
                searching for a target Dice so images are not resampled
                on every binary-search step).

        Returns:
            The warped subject copy (same keys, same geometry).
        """
        if self.control_spacing is not None:
            return self._warp_ffd(
                subject,
                seed=seed,
                magnitude_range=magnitude_range,
                mask_keys_only=mask_keys_only,
            )
        from habit.utils.optional_deps import require

        require(
            "monai.transforms",
            extra="monai",
            purpose=(
                "MONAI Rand3DElastic free-form / elastic deformation of "
                "images and ROI contours"
            ),
        )
        from monai.transforms import Rand3DElasticd

        image_keys = [] if mask_keys_only else list(subject.images)
        mask_keys = list(subject.masks)
        if not image_keys and not mask_keys:
            raise HABITAPIError(
                "bspline_deform: subject has no images or masks to warp."
            )

        data: Dict[str, np.ndarray] = {}
        modes: list[Union[str, int]] = []
        keys: list[str] = []
        reference_shape: Optional[Tuple[int, int, int]] = None

        def _add(key: str, array: np.ndarray, order: Union[str, int]) -> None:
            nonlocal reference_shape
            volume = np.asarray(array)
            if volume.ndim != 3:
                raise HABITAPIError(
                    f"bspline_deform: {key!r} must be 3-D (z, y, x); "
                    f"got shape {tuple(volume.shape)}."
                )
            shape = tuple(int(s) for s in volume.shape)
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise HABITAPIError(
                    "bspline_deform: all images and masks must share one "
                    f"grid; {key!r} is {shape}, expected {reference_shape}."
                )
            # Channel-first float32 is what Rand3DElasticd resamples.
            data[key] = np.asarray(volume, dtype=np.float32)[np.newaxis, ...]
            keys.append(key)
            modes.append(order)

        for modality in image_keys:
            _add(f"image:{modality}", subject.image(modality).data, self.image_mode)
        for roi in mask_keys:
            _add(f"mask:{roi}", subject.mask(roi).data, self.mask_mode)

        assert reference_shape is not None
        transform = Rand3DElasticd(
            keys=keys,
            sigma_range=self.sigma_range,
            magnitude_range=magnitude_range,
            spatial_size=reference_shape,
            prob=1.0,
            mode=tuple(modes),
            padding_mode=self.padding_mode,
            device=self.device,
        )
        transform.set_random_state(int(seed))
        warped = transform(data)

        images: Dict[str, np.ndarray] = {}
        for modality in list(subject.images):
            if mask_keys_only:
                images[modality] = np.asarray(subject.image(modality).data)
            else:
                images[modality] = _as_numpy_volume(
                    warped[f"image:{modality}"]
                ).astype(np.float64, copy=False)
        masks: Dict[str, np.ndarray] = {}
        for roi in mask_keys:
            mask_array = np.asarray(subject.mask(roi).data)
            warped_mask = _as_numpy_volume(warped[f"mask:{roi}"])
            # Nearest-neighbour (order 0) is already discrete; rint absorbs
            # residual float dust from the MONAI / scipy backend.
            masks[roi] = np.rint(warped_mask).astype(mask_array.dtype)
        return _replace_images(subject, images, masks)

    def _warp_ffd(
        self,
        subject: Subject,
        *,
        seed: int,
        magnitude_range: Tuple[float, float],
        mask_keys_only: bool,
    ) -> Subject:
        """
        Apply one coarse-lattice cubic B-spline FFD (no MONAI).

        Args:
            subject: Subject providing images and/or masks on one grid.
            seed: Frozen seed for the control-point offsets.
            magnitude_range: Displacement magnitude ``(low, high)``.
                When both ends match (target-Dice search), that value is
                used; otherwise one draw is taken from the range.
            mask_keys_only: When ``True``, warp masks only.

        Returns:
            The warped subject copy (same keys, same geometry).
        """
        assert self.control_spacing is not None
        image_keys = [] if mask_keys_only else list(subject.images)
        mask_keys = list(subject.masks)
        if not image_keys and not mask_keys:
            raise HABITAPIError(
                "bspline_deform: subject has no images or masks to warp."
            )
        reference_shape: Optional[Tuple[int, int, int]] = None

        def _shape_of(array: np.ndarray, key: str) -> Tuple[int, int, int]:
            nonlocal reference_shape
            volume = np.asarray(array)
            if volume.ndim != 3:
                raise HABITAPIError(
                    f"bspline_deform: {key!r} must be 3-D (z, y, x); "
                    f"got shape {tuple(volume.shape)}."
                )
            shape = tuple(int(s) for s in volume.shape)
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise HABITAPIError(
                    "bspline_deform: all images and masks must share one "
                    f"grid; {key!r} is {shape}, expected {reference_shape}."
                )
            return shape

        for modality in image_keys:
            _shape_of(subject.image(modality).data, f"image:{modality}")
        for roi in mask_keys:
            _shape_of(subject.mask(roi).data, f"mask:{roi}")
        assert reference_shape is not None
        if float(magnitude_range[0]) == float(magnitude_range[1]):
            magnitude = float(magnitude_range[0])
        else:
            magnitude = float(
                np.random.default_rng(int(seed)).uniform(
                    float(magnitude_range[0]), float(magnitude_range[1])
                )
            )
        field = _ffd_displacement(
            reference_shape,
            control_spacing=self.control_spacing,
            magnitude=magnitude,
            seed=seed,
        )
        images: Dict[str, np.ndarray] = {}
        for modality in list(subject.images):
            if mask_keys_only:
                images[modality] = np.asarray(subject.image(modality).data)
            else:
                images[modality] = _warp_with_displacement(
                    np.asarray(subject.image(modality).data),
                    field,
                    order=_scipy_spline_order(self.image_mode),
                    padding_mode=self.padding_mode,
                )
        masks: Dict[str, np.ndarray] = {}
        for roi in mask_keys:
            mask_array = np.asarray(subject.mask(roi).data)
            warped_mask = _warp_with_displacement(
                mask_array,
                field,
                order=_scipy_spline_order(self.mask_mode),
                padding_mode=self.padding_mode,
            )
            masks[roi] = np.rint(warped_mask).astype(mask_array.dtype)
        return _replace_images(subject, images, masks)

    def _mean_roi_dice(self, reference: Subject, warped: Subject) -> float:
        """Return the mean binary Dice across every mask key."""
        from habit.kernels.image_perturbation import binary_mask_dice

        scores = [
            binary_mask_dice(
                np.asarray(reference.mask(roi).data),
                np.asarray(warped.mask(roi).data),
            )
            for roi in reference.masks
        ]
        return float(np.mean(np.asarray(scores, dtype=float)))

    def _magnitude_for_target_dice(self, subject: Subject, seed: int) -> float:
        """
        Binary-search a magnitude so ROI Dice matches ``target_dice``.

        The MONAI offset-grid direction is frozen by ``seed``; only the
        scale changes. Larger magnitude usually lowers Dice.

        Args:
            subject: Subject whose masks define the Dice.
            seed: Frozen seed for the offset field.

        Returns:
            float: Magnitude that lands closest to ``target_dice``.
        """
        assert self.target_dice is not None
        target = float(self.target_dice)
        lo = 0.0
        hi = max(float(self.magnitude_range[1]) * 2.0, 8.0)
        best_magnitude = hi
        best_error = 1.0
        # Expand the upper bound if even a large warp stays too similar.
        for _ in range(4):
            probe = self._warp(
                subject,
                seed=seed,
                magnitude_range=(hi, hi),
                mask_keys_only=True,
            )
            dice = self._mean_roi_dice(subject, probe)
            error = abs(dice - target)
            if error < best_error:
                best_error = error
                best_magnitude = hi
            if dice <= target + self.dice_tolerance:
                break
            hi = min(hi * 2.0, 64.0)
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            probe = self._warp(
                subject,
                seed=seed,
                magnitude_range=(mid, mid),
                mask_keys_only=True,
            )
            dice = self._mean_roi_dice(subject, probe)
            error = abs(dice - target)
            if error < best_error:
                best_error = error
                best_magnitude = mid
            if error <= self.dice_tolerance:
                return mid
            if dice > target:
                lo = mid
            else:
                hi = mid
        return best_magnitude

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` warped by one MONAI elastic field.

        Args:
            subject: Subject providing images and/or masks on one grid.
            rng: Random generator; one integer seed is drawn so every
                volume of this subject shares the same displacement.

        Returns:
            The warped subject copy (same keys, same geometry).

        Raises:
            OptionalDependencyError: When the ``monai`` extra is missing.
            HABITAPIError: When the subject has no volumes, or images
                and masks do not share one 3-D shape. ``target_dice``
                also requires at least one mask.
        """
        seed = int(rng.integers(0, 2**31 - 1))
        if self.target_dice is None:
            return self._warp(
                subject, seed=seed, magnitude_range=self.magnitude_range
            )
        if not list(subject.masks):
            raise HABITAPIError(
                "bspline_deform: target_dice requires at least one ROI mask."
            )
        magnitude = self._magnitude_for_target_dice(subject, seed)
        return self._warp(
            subject, seed=seed, magnitude_range=(magnitude, magnitude)
        )


ImagePerturbationRegistry.register_params_model(
    "gaussian_noise", GaussianNoisePerturbationParams
)
ImagePerturbationRegistry.register_params_model(
    "translation", TranslationPerturbationParams
)
ImagePerturbationRegistry.register_params_model("rotation", RotationPerturbationParams)
ImagePerturbationRegistry.register_params_model("rigid", RigidPerturbationParams)
ImagePerturbationRegistry.register_params_model(
    "bspline_deform", BSplineDeformPerturbationParams
)
ImagePerturbationRegistry.register_params_model(
    "morphological", MorphologicalPerturbationParams
)
ImagePerturbationRegistry.register_params_model(
    "gradient_weighted", GradientWeightedPerturbationParams
)
ImagePerturbationRegistry.register_params_model(
    "slice_extent", SliceExtentPerturbationParams
)
