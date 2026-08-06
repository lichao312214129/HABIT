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
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from habit.contracts.image import ArrayImageRef, ImageRef
from habit.contracts.subject import Subject
from habit.domain.precision.registry import ImagePerturbationRegistry
from habit.exceptions import HABITAPIError
from habit.kernels.image_perturbation import (
    add_gaussian_noise,
    estimate_noise_sigma,
    rotate_image,
    translate_image,
)
from habit.spec.specs import Spec

__all__ = [
    "GaussianNoisePerturbation",
    "GaussianNoisePerturbationParams",
    "TranslationPerturbation",
    "TranslationPerturbationParams",
    "RotationPerturbation",
    "RotationPerturbationParams",
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
    max_shift_voxels: float = Field(default=1.0, ge=0.0)
    interpolator: str = "bspline"


@ImagePerturbationRegistry.register("translation")
class TranslationPerturbation:
    """
    Translate image content by a (random) sub-voxel shift.

    Args:
        shift_voxels: Fixed shift in voxel units, SimpleITK ``(x, y, z)``
            order; ``None`` samples a uniform random shift per axis from
            ``[-max_shift_voxels, +max_shift_voxels]`` (the paper's
            sub-voxel translation).
        max_shift_voxels: Sampling bound for the random shift; the paper's
            sub-voxel regime corresponds to ``1.0``.
        interpolator: Interpolator for the intensity images (``"bspline"``
            is the paper's choice); masks always use nearest neighbour.
    """

    def __init__(
        self,
        shift_voxels: Optional[Sequence[float]] = None,
        max_shift_voxels: float = 1.0,
        interpolator: str = "bspline",
    ) -> None:
        if shift_voxels is not None and len(tuple(shift_voxels)) != 3:
            raise HABITAPIError(
                "translation: shift_voxels must have 3 components (x, y, z); "
                f"got {tuple(shift_voxels)}."
            )
        if max_shift_voxels < 0.0:
            raise HABITAPIError(
                f"translation: max_shift_voxels must be >= 0; got {max_shift_voxels}."
            )
        self.shift_voxels = (
            None if shift_voxels is None else tuple(float(v) for v in shift_voxels)
        )
        self.max_shift_voxels = float(max_shift_voxels)
        self.interpolator = str(interpolator)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="translation",
            params={
                "shift_voxels": self.shift_voxels,
                "max_shift_voxels": self.max_shift_voxels,
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
    """

    def __init__(
        self,
        angle_degrees: float = 0.5,
        axis: str = "z",
        interpolator: str = "bspline",
    ) -> None:
        if axis not in ("x", "y", "z"):
            raise HABITAPIError(
                f"rotation: axis must be one of ('x', 'y', 'z'); got {axis!r}."
            )
        self.angle_degrees = float(angle_degrees)
        self.axis = axis
        self.interpolator = str(interpolator)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="rotation",
            params={
                "angle_degrees": self.angle_degrees,
                "axis": self.axis,
                "interpolator": self.interpolator,
            },
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` rotated by ``angle_degrees``.

        Args:
            subject: Subject providing images and masks.
            rng: Accepted for protocol conformance; the rotation is
                deterministic and the generator is not consumed.

        Returns:
            The perturbed subject copy.
        """
        return _geometric_transform(
            subject,
            lambda image, interpolator, default_value: rotate_image(
                image,
                self.angle_degrees,
                axis=self.axis,
                interpolator=interpolator,
                default_value=default_value,
            ),
            self.interpolator,
        )


ImagePerturbationRegistry.register_params_model(
    "gaussian_noise", GaussianNoisePerturbationParams
)
ImagePerturbationRegistry.register_params_model(
    "translation", TranslationPerturbationParams
)
ImagePerturbationRegistry.register_params_model("rotation", RotationPerturbationParams)
