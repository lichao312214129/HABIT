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
"""Volume-level image perturbation atom (L3).

:func:`perturb_image` is the single-volume entry point for the registered
``image_perturbation`` family. Callers pass one :class:`ImageVolume`, a
method name, and that method's constructor parameters -- no
:class:`~habit.contracts.subject.Cohort`, YAML, or precision recipe.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from habit.api.image import ImageVolume, MaskVolume
from habit.contracts.geometry import Geometry
from habit.contracts.image import ArrayImageRef
from habit.contracts.subject import Subject
from habit.precision.registry import ImagePerturbationRegistry
from habit.exceptions import HABITAPIError

# Importing the implementations registers the built-in method names on
# ``ImagePerturbationRegistry`` (gaussian_noise, translation, rotation,
# rigid, bspline_deform).
from habit.precision import perturbations as _perturbations  # noqa: F401

__all__ = ["perturb_image"]

#: Synthetic keys used when wrapping a lone volume into a Subject.
_DEFAULT_MODALITY = "image"
_DEFAULT_ROI = "roi"


def _geometry_of(volume: ImageVolume) -> Geometry:
    """
    Build a :class:`Geometry` from a public or contracts volume.

    Args:
        volume: Intensity or mask volume with spacing / origin / direction.

    Returns:
        Geometry describing ``volume.data``.
    """
    existing = getattr(volume, "geometry", None)
    if isinstance(existing, Geometry):
        return existing
    return Geometry(
        shape=tuple(int(v) for v in volume.data.shape),
        spacing=tuple(float(v) for v in volume.spacing),
        origin=tuple(float(v) for v in volume.origin),
        direction=tuple(float(v) for v in volume.direction),
    )


def _subject_from_volumes(
    image: ImageVolume,
    mask: Optional[MaskVolume] = None,
    *,
    modality: str = _DEFAULT_MODALITY,
    roi: str = _DEFAULT_ROI,
) -> Subject:
    """
    Wrap one image (and optional mask) as a single-modality Subject.

    Args:
        image: Intensity volume.
        mask: Optional ROI sharing the image grid (required by methods
            that estimate noise from the ROI or warp labels).
        modality: Synthetic image key inside the subject.
        roi: Synthetic mask key inside the subject.

    Returns:
        An in-memory subject the registered perturbation can call.
    """
    geometry = _geometry_of(image)
    images = {modality: ArrayImageRef(array=np.asarray(image.data), geometry=geometry)}
    masks = {}
    if mask is not None:
        masks[roi] = ArrayImageRef(
            array=np.asarray(mask.data), geometry=_geometry_of(mask)
        )
    return Subject(
        subject_id=str(image.subject_id or "image"),
        images=images,
        masks=masks,
    )


def perturb_image(
    image: ImageVolume,
    method: str,
    *,
    mask: Optional[MaskVolume] = None,
    seed: int = 0,
    rng: Optional[np.random.Generator] = None,
    **params: Any,
) -> ImageVolume:
    """
    Return a same-grid perturbed copy of one image.

    This is the atomic teaching / experiment call: one volume, one
    registered method, and that method's parameters. Intensity methods
    are ``gaussian_noise``, ``translation``, ``rotation``, ``rigid``,
    and ``bspline_deform`` (the last needs the optional ``monai`` extra).
    Mask-only contour methods (``morphological``, ``gradient_weighted``,
    ``slice_extent``) leave the image unchanged; call the registered
    component on a :class:`~habit.contracts.subject.Subject` when you
    need the edited mask.
    Geometric methods resample back onto the original voxel grid so the
    result stays voxel-wise comparable to ``image``.

    ``mask`` is optional. When given, methods that need an ROI (for
    example ``gaussian_noise`` with ``noise_method="roi_std"``, or
    ``bspline_deform`` with ``target_dice``) see it, and geometric
    methods warp the mask internally. This function still returns only
    the perturbed intensity volume. Use the registered component on a
    :class:`~habit.contracts.subject.Subject` when you also need the
    warped mask.

    Args:
        image: Source intensity volume.
        method: Registered ``image_perturbation`` name.
        mask: Optional ROI used by methods that consult or warp a mask.
        seed: Used when ``rng`` is omitted; each call with the same
            seed and parameters is reproducible.
        rng: Random generator; overrides ``seed`` when set.
        **params: Constructor kwargs of the named method (validated by
            that method's params model). Examples: ``sigma`` /
            ``noise_method`` for ``gaussian_noise``; ``shift_voxels`` or
            ``shift_fraction`` for ``translation``; ``angle_degrees``
            for ``rotation``.

    Returns:
        Perturbed intensity volume on the same grid as ``image``.
        ``metadata["perturbation"]`` records the component :class:`Spec`.

    Raises:
        HABITAPIError: When ``method`` is unknown or ``params`` fail
            validation.
        OptionalDependencyError: When the method's extra is missing
            (``bspline_deform`` without MONAI).
    """
    name = str(method).strip()
    if not name:
        raise HABITAPIError("perturb_image: method must be a non-empty string.")
    component = ImagePerturbationRegistry.create(name, **params)
    generator = rng if rng is not None else np.random.default_rng(int(seed))
    modality = str(image.modality or _DEFAULT_MODALITY)
    roi = str(getattr(mask, "modality", None) or _DEFAULT_ROI) if mask is not None else _DEFAULT_ROI
    subject = _subject_from_volumes(image, mask, modality=modality, roi=roi)
    perturbed = component(subject, rng=generator)
    volume = perturbed.image(modality)
    metadata = dict(getattr(image, "metadata", {}) or {})
    metadata["perturbation"] = component.spec.to_dict()
    return type(volume)(
        data=np.asarray(volume.data),
        spacing=tuple(volume.spacing),
        origin=tuple(volume.origin),
        direction=tuple(volume.direction),
        modality=image.modality,
        subject_id=image.subject_id,
        timepoint=getattr(image, "timepoint", None),
        metadata=metadata,
    )
