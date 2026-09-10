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
"""Filesystem I/O and geometry alignment for :class:`~habit.contracts.image.ImageVolume`.

Types live in :mod:`habit.contracts.image`. This adapter is the only layer
that reads image files or resamples a mask onto an image grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union
import warnings

import numpy as np

from habit.contracts.image import (
    GeometryPolicy,
    GeometryReport,
    ImageMaskPair,
    ImageVolume,
    MaskVolume,
)
from habit.exceptions import GeometryError, HABITAPIError, OptionalDependencyError

__all__ = [
    "ImageInput",
    "MaskInput",
    "read_image",
    "read_mask",
    "validate_geometry",
    "align_image_mask",
]

ImageInput = Union[ImageVolume, np.ndarray, str, Path, Any]
MaskInput = Union[MaskVolume, np.ndarray, str, Path, Any]


def _require_simpleitk() -> Any:
    """Import SimpleITK only for operations that require it."""
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "SimpleITK is required for image file I/O and resampling. "
            "Install the radiomics or full HABIT dependency set."
        ) from exc
    return sitk


def read_image(
    path: Union[str, Path],
    *,
    modality: Optional[str] = None,
    subject_id: Optional[str] = None,
    timepoint: Optional[str] = None,
) -> ImageVolume:
    """Read an image file into a geometry-aware :class:`ImageVolume`."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Image file not found: {source}")
    sitk = _require_simpleitk()
    return ImageVolume.from_sitk(
        sitk.ReadImage(str(source)),
        modality=modality,
        subject_id=subject_id,
        timepoint=timepoint,
        source=source,
    )


def read_mask(
    path: Union[str, Path],
    *,
    labels: Tuple[int, ...] = (),
    label_names: Optional[Mapping[int, str]] = None,
    subject_id: Optional[str] = None,
) -> MaskVolume:
    """Read a mask file into a geometry-aware :class:`MaskVolume`."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Mask file not found: {source}")
    sitk = _require_simpleitk()
    return MaskVolume.from_sitk(
        sitk.ReadImage(str(source)),
        labels=labels,
        label_names=label_names,
        subject_id=subject_id,
        source=source,
    )


def coerce_image(image: ImageInput) -> ImageVolume:
    """Convert supported public image inputs to :class:`ImageVolume`."""
    if isinstance(image, ImageVolume) and not isinstance(image, MaskVolume):
        return image
    if isinstance(image, np.ndarray):
        return ImageVolume.from_array(image)
    if isinstance(image, (str, Path)):
        return read_image(image)
    sitk = _require_simpleitk()
    if isinstance(image, sitk.Image):
        return ImageVolume.from_sitk(image)
    raise HABITAPIError(
        "image must be an ImageVolume, numpy array, SimpleITK.Image, or file path."
    )


def coerce_mask(mask: MaskInput) -> MaskVolume:
    """Convert supported public mask inputs to :class:`MaskVolume`."""
    if isinstance(mask, MaskVolume):
        return mask
    if isinstance(mask, np.ndarray):
        return MaskVolume.from_array(mask)
    if isinstance(mask, (str, Path)):
        return read_mask(mask)
    sitk = _require_simpleitk()
    if isinstance(mask, sitk.Image):
        return MaskVolume.from_sitk(mask)
    raise HABITAPIError(
        "mask must be a MaskVolume, numpy array, SimpleITK.Image, or file path."
    )


def validate_geometry(
    image: ImageInput,
    mask: MaskInput,
    *,
    tolerance: float = 1e-6,
) -> GeometryReport:
    """Return an explicit geometry comparison without changing either input."""
    if tolerance < 0.0:
        raise HABITAPIError("tolerance must be greater than or equal to zero.")
    image_volume = coerce_image(image)
    mask_volume = coerce_mask(mask)
    mismatches = []
    if image_volume.data.shape != mask_volume.data.shape:
        mismatches.append("shape")
    if not np.allclose(image_volume.spacing, mask_volume.spacing, atol=tolerance):
        mismatches.append("spacing")
    if not np.allclose(image_volume.origin, mask_volume.origin, atol=tolerance):
        mismatches.append("origin")
    if not np.allclose(image_volume.direction, mask_volume.direction, atol=tolerance):
        mismatches.append("direction")
    return GeometryReport(
        compatible=not mismatches,
        mismatches=tuple(mismatches),
        tolerance=tolerance,
    )


def align_image_mask(
    pair: ImageMaskPair,
    *,
    policy: GeometryPolicy = GeometryPolicy.STRICT,
    tolerance: float = 1e-6,
) -> ImageMaskPair:
    """Validate or explicitly correct image/mask geometry according to ``policy``."""
    report = validate_geometry(pair.image, pair.mask, tolerance=tolerance)
    if report.compatible:
        return ImageMaskPair(pair.image, pair.mask, report)
    message = (
        "Image and mask geometry are incompatible: " f"{', '.join(report.mismatches)}."
    )
    if policy is GeometryPolicy.STRICT:
        raise GeometryError(message)
    if policy is GeometryPolicy.WARN:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return ImageMaskPair(
            pair.image,
            pair.mask,
            GeometryReport(
                compatible=False,
                mismatches=report.mismatches,
                action=GeometryPolicy.WARN.value,
                tolerance=tolerance,
            ),
        )

    sitk = _require_simpleitk()
    if policy is GeometryPolicy.HARMONIZE:
        if pair.image.data.shape == pair.mask.data.shape:
            corrected_mask = MaskVolume(
                pair.mask.data,
                spacing=pair.image.spacing,
                origin=pair.image.origin,
                direction=pair.image.direction,
                labels=pair.mask.labels,
                label_names=pair.mask.label_names,
                subject_id=pair.mask.subject_id,
                source=pair.mask.source,
                metadata=pair.mask.metadata,
            )
            return ImageMaskPair(
                pair.image,
                corrected_mask,
                GeometryReport(
                    compatible=True,
                    mismatches=report.mismatches,
                    action=GeometryPolicy.HARMONIZE.value,
                    tolerance=tolerance,
                ),
            )
        policy = GeometryPolicy.RESAMPLE_MASK

    if policy is GeometryPolicy.RESAMPLE_MASK:
        resampled = sitk.Resample(
            pair.mask.to_sitk(),
            pair.image.to_sitk(),
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            pair.mask.to_sitk().GetPixelID(),
        )
        corrected_mask = MaskVolume.from_sitk(
            resampled,
            labels=pair.mask.labels,
            label_names=pair.mask.label_names,
            subject_id=pair.mask.subject_id,
            source=pair.mask.source,
            metadata=pair.mask.metadata,
        )
        return ImageMaskPair(
            pair.image,
            corrected_mask,
            GeometryReport(
                compatible=True,
                mismatches=report.mismatches,
                action=GeometryPolicy.RESAMPLE_MASK.value,
                tolerance=tolerance,
            ),
        )
    if policy is GeometryPolicy.RESAMPLE_IMAGE:
        resampled = sitk.Resample(
            pair.image.to_sitk(),
            pair.mask.to_sitk(),
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            pair.image.to_sitk().GetPixelID(),
        )
        corrected_image = ImageVolume.from_sitk(
            resampled,
            modality=pair.image.modality,
            subject_id=pair.image.subject_id,
            timepoint=pair.image.timepoint,
            source=pair.image.source,
            metadata=pair.image.metadata,
        )
        return ImageMaskPair(
            corrected_image,
            pair.mask,
            GeometryReport(
                compatible=True,
                mismatches=report.mismatches,
                action=GeometryPolicy.RESAMPLE_IMAGE.value,
                tolerance=tolerance,
            ),
        )
    raise HABITAPIError(f"Unsupported geometry policy: {policy!r}.")
