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
Artif Intell 2024;6(2):e230118, Appendix E2) -- Gaussian noise, sub-voxel
translation, small-angle rotation -- implemented on the L0 kernels of
:mod:`habit.kernels.image_perturbation`. Each one maps a
:class:`~habit.contracts.subject.Subject` to a perturbed copy on the SAME
voxel grid, so perturbed feature maps stay comparable to the original
voxel-by-voxel.

Geometric perturbations transform the masks as well (nearest-neighbour),
because a shifted acquisition images a shifted patient; the precision
analysis then aligns the original and perturbed feature fields on their
common ROI voxels.
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
    estimate_noise_sigma,
    rigid_transform_image,
    rotate_image,
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
) -> Subject:
    """
    Apply one geometric kernel to every image and mask of a subject.

    Args:
        subject: Source subject.
        transform: Kernel (``translate_image`` / ``rotate_image``) called as
            ``transform(sitk_image, interpolator=..., default_value=...)``.
        interpolator: Interpolator for the intensity images; masks always
            use nearest neighbour.

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
            is the paper's choice); masks always use nearest neighbour.
    """

    def __init__(
        self,
        shift_voxels: Optional[Sequence[float]] = None,
        max_shift_voxels: float = 1.0,
        interpolator: str = "bspline",
        shift_fraction: Optional[float] = None,
        random_signs: bool = True,
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
        )


class RotationPerturbationParams(BaseModel):
    """Constructor parameters for :class:`RotationPerturbation`."""

    model_config = ConfigDict(extra="forbid")
    angle_degrees: float = 0.5
    axis: str = "z"
    interpolator: str = "bspline"
    random_sign: bool = False


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
            is the paper's choice); masks always use nearest neighbour.
        random_sign: When ``True``, the sign of ``angle_degrees`` is drawn
            as ``±1`` per call (some MIRP configs randomize the sense of
            the 0.5° in-plane rotation). The paper's default is a fixed
            ``+0.5`` degrees, so this stays ``False``.
    """

    def __init__(
        self,
        angle_degrees: float = 0.5,
        axis: str = "z",
        interpolator: str = "bspline",
        random_sign: bool = False,
    ) -> None:
        if axis not in ("x", "y", "z"):
            raise HABITAPIError(
                f"rotation: axis must be one of ('x', 'y', 'z'); got {axis!r}."
            )
        self.angle_degrees = float(angle_degrees)
        self.axis = axis
        self.interpolator = str(interpolator)
        self.random_sign = bool(random_sign)

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
        )


def prior2024_retest_perturbation(
    *,
    shift_fraction: float = 0.5,
    angle_degrees: float = 0.5,
    interpolator: str = "bspline",
    single_resample: bool = False,
) -> "PerturbationChain":
    """
    Simulated-retest chain of Prior et al. 2024 / MIRP 1.2.0 Appendix S2.

    Paper: Prior O, et al. Identification of Precise 3D CT Radiomics for
    Habitat Computation by Machine Learning in Cancer. Radiol Artif Intell.
    2024;6(2):e230118. doi:10.1148/ryai.230118

    Order: Gaussian noise (Chang wavelet sigma) → sub-voxel translation
    (fraction ``η`` of voxel spacing, default 0.5, random axis signs) →
    0.5° in-plane (z) rotation. Images use B-spline; masks nearest
    neighbour. ``single_resample=True`` composes translation+rotation
    (MIRP ≥ 2); the paper used two geometric resamples.

    ROI morphological grow/shrink (MIRP ``perturbation_roi_adapt_size``)
    is not in this protocol. MONAI elastic / B-spline free-form warps
    are a separate optional component
    (:class:`BSplineDeformPerturbation`), not part of this chain.

    Args:
        shift_fraction: MIRP translation fraction in ``[0, 1]``.
        angle_degrees: In-plane rotation in degrees.
        interpolator: Intensity interpolator.
        single_resample: Compose translation and rotation into one affine.

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
                ),
            ]
        )
    return PerturbationChain(
        [
            noise,
            TranslationPerturbation(
                shift_fraction=shift_fraction, interpolator=interpolator
            ),
            RotationPerturbation(
                angle_degrees=angle_degrees, interpolator=interpolator
            ),
        ]
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

    This is MONAI's documented 3-D elastic deformation, not Rueckert
    cubic-B-spline FFD with an explicit control-point mesh, and not MIRP
    ``perturbation_roi_adapt_size`` (morphological grow/shrink).

    Args:
        sigma_range: Gaussian smoothing ``(low, high)`` of the offset
            grid, in voxels. Wider sigma gives a smoother warp.
        magnitude_range: Displacement magnitude ``(low, high)`` in
            voxels. Larger values wrinkle the ROI more.
        image_mode: Intensity interpolator (``"bilinear"`` / ``"nearest"``
            or spline order 0–5).
        mask_mode: ROI interpolator (``"nearest"`` recommended).
        padding_mode: Out-of-grid padding (``reflection``, ``border``,
            or ``zeros``).
        device: Torch device (``"cpu"`` / ``"cuda"``). Default ``"cpu"``
            is the portable path; pass ``"cuda"`` when a GPU is available
            and the volume fits in memory.
    """

    def __init__(
        self,
        sigma_range: Sequence[float] = (1.5, 3.0),
        magnitude_range: Sequence[float] = (8.0, 12.0),
        image_mode: Union[str, int] = "bilinear",
        mask_mode: Union[str, int] = "nearest",
        padding_mode: str = "reflection",
        device: str = "cpu",
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
            },
        )

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
                and masks do not share one 3-D shape.
        """
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

        image_keys = list(subject.images)
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
            magnitude_range=self.magnitude_range,
            spatial_size=reference_shape,
            prob=1.0,
            mode=tuple(modes),
            padding_mode=self.padding_mode,
            device=self.device,
        )
        # One seed drives the shared offset grid for every key.
        transform.set_random_state(int(rng.integers(0, 2**31 - 1)))
        warped = transform(data)

        images: Dict[str, np.ndarray] = {}
        for modality in image_keys:
            images[modality] = _as_numpy_volume(warped[f"image:{modality}"]).astype(
                np.float64, copy=False
            )
        masks: Dict[str, np.ndarray] = {}
        for roi in mask_keys:
            mask_array = np.asarray(subject.mask(roi).data)
            warped_mask = _as_numpy_volume(warped[f"mask:{roi}"])
            # Nearest-neighbour (order 0) is already discrete; rint absorbs
            # residual float dust from the MONAI / scipy backend.
            masks[roi] = np.rint(warped_mask).astype(mask_array.dtype)
        return _replace_images(subject, images, masks)


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
