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
"""Stable image, mask, and spatial-geometry contracts for HABIT APIs.

The workflow implementation historically passed file paths, ``SimpleITK`` images,
and ``numpy`` arrays through different code paths.  This module provides one
explicit public representation so callers can validate image/mask geometry before
running preprocessing, radiomics, or habitat analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union
import warnings

import numpy as np

from habit.api.exceptions import GeometryError, HABITAPIError, OptionalDependencyError

__all__ = [
    "GeometryPolicy",
    "GeometryReport",
    "ImageVolume",
    "MaskVolume",
    "ImageMaskPair",
    "ImageInput",
    "MaskInput",
    "read_image",
    "read_mask",
    "validate_geometry",
    "align_image_mask",
]

ImageInput = Union["ImageVolume", np.ndarray, str, Path, Any]
MaskInput = Union["MaskVolume", np.ndarray, str, Path, Any]


class GeometryPolicy(str, Enum):
    """Define how an image/mask geometry mismatch is handled."""

    STRICT = "strict"
    RESAMPLE_MASK = "resample_mask"
    RESAMPLE_IMAGE = "resample_image"
    WARN = "warn"


@dataclass(frozen=True)
class GeometryReport:
    """Describe image/mask geometry compatibility and any correction applied."""

    compatible: bool
    mismatches: Tuple[str, ...] = ()
    action: str = "none"
    tolerance: float = 1e-6


def _normalized_geometry(
    *,
    ndim: int,
    spacing: Tuple[float, ...],
    origin: Tuple[float, ...],
    direction: Tuple[float, ...],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Validate and normalize physical metadata for a volume."""
    if ndim not in (2, 3, 4):
        raise HABITAPIError(
            f"Only 2D, 3D, and 4D volumes are supported; received {ndim}D."
        )
    normalized_spacing = tuple(float(value) for value in spacing)
    normalized_origin = tuple(float(value) for value in origin)
    normalized_direction = tuple(float(value) for value in direction)
    if len(normalized_spacing) != ndim:
        raise HABITAPIError(
            f"spacing must contain {ndim} values; received {len(normalized_spacing)}."
        )
    if len(normalized_origin) != ndim:
        raise HABITAPIError(
            f"origin must contain {ndim} values; received {len(normalized_origin)}."
        )
    if len(normalized_direction) != ndim * ndim:
        raise HABITAPIError(
            "direction must contain a square direction matrix flattened to "
            f"{ndim * ndim} values; received {len(normalized_direction)}."
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized_spacing):
        raise HABITAPIError("spacing must contain finite values greater than zero.")
    if any(
        not np.isfinite(value) for value in normalized_origin + normalized_direction
    ):
        raise HABITAPIError("origin and direction must contain finite values.")
    return normalized_spacing, normalized_origin, normalized_direction


def _default_direction(ndim: int) -> Tuple[float, ...]:
    """Return a flattened identity direction matrix for ``ndim`` dimensions."""
    return tuple(float(value) for value in np.eye(ndim, dtype=float).ravel())


@dataclass(frozen=True)
class ImageVolume:
    """An image array with explicit physical-space metadata.

    ``data`` follows the NumPy convention used by ``SimpleITK.GetArrayFromImage``.
    Physical metadata remains in SimpleITK axis order.  The distinction is
    intentional: callers should use this object rather than infer geometry from
    array axes.
    """

    data: np.ndarray
    spacing: Tuple[float, ...]
    origin: Tuple[float, ...]
    direction: Tuple[float, ...]
    modality: Optional[str] = None
    subject_id: Optional[str] = None
    timepoint: Optional[str] = None
    source: Optional[Path] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize arrays and immutable metadata at the public boundary."""
        array = np.asarray(self.data)
        if array.ndim not in (2, 3, 4):
            raise HABITAPIError(
                f"ImageVolume.data must be 2D, 3D, or 4D; received {array.ndim}D."
            )
        if array.size == 0:
            raise HABITAPIError("ImageVolume.data must not be empty.")
        spacing, origin, direction = _normalized_geometry(
            ndim=array.ndim,
            spacing=self.spacing,
            origin=self.origin,
            direction=self.direction,
        )
        object.__setattr__(self, "data", array)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(
            self, "source", Path(self.source) if self.source is not None else None
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        *,
        spacing: Optional[Tuple[float, ...]] = None,
        origin: Optional[Tuple[float, ...]] = None,
        direction: Optional[Tuple[float, ...]] = None,
        modality: Optional[str] = None,
        subject_id: Optional[str] = None,
        timepoint: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ImageVolume":
        """Create a volume from an array using identity physical metadata by default."""
        array = np.asarray(data)
        ndim = array.ndim
        return cls(
            data=array,
            spacing=spacing or tuple(1.0 for _ in range(ndim)),
            origin=origin or tuple(0.0 for _ in range(ndim)),
            direction=direction or _default_direction(ndim),
            modality=modality,
            subject_id=subject_id,
            timepoint=timepoint,
            metadata=metadata or {},
        )

    @classmethod
    def from_sitk(
        cls,
        image: Any,
        *,
        modality: Optional[str] = None,
        subject_id: Optional[str] = None,
        timepoint: Optional[str] = None,
        source: Optional[Union[str, Path]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ImageVolume":
        """Convert a ``SimpleITK.Image`` without discarding its physical metadata."""
        sitk = _require_simpleitk()
        if not isinstance(image, sitk.Image):
            raise HABITAPIError("image must be a SimpleITK.Image.")
        return cls(
            data=sitk.GetArrayFromImage(image),
            spacing=tuple(image.GetSpacing()),
            origin=tuple(image.GetOrigin()),
            direction=tuple(image.GetDirection()),
            modality=modality,
            subject_id=subject_id,
            timepoint=timepoint,
            source=Path(source) if source is not None else None,
            metadata=metadata or {},
        )

    def to_sitk(self) -> Any:
        """Convert this public volume into a ``SimpleITK.Image`` lazily."""
        sitk = _require_simpleitk()
        image = sitk.GetImageFromArray(self.data)
        image.SetSpacing(self.spacing)
        image.SetOrigin(self.origin)
        image.SetDirection(self.direction)
        return image


@dataclass(frozen=True)
class MaskVolume(ImageVolume):
    """An image-space segmentation mask with explicit label semantics."""

    labels: Tuple[int, ...] = ()
    label_names: Mapping[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate mask labels after the shared image-volume normalization."""
        super().__post_init__()
        if not np.issubdtype(self.data.dtype, np.number):
            raise HABITAPIError("MaskVolume.data must use a numeric dtype.")
        inferred_labels = tuple(
            int(value) for value in np.unique(self.data) if int(value) != 0
        )
        normalized_labels = (
            tuple(sorted({int(value) for value in self.labels}))
            if self.labels
            else inferred_labels
        )
        if any(value == 0 for value in normalized_labels):
            raise HABITAPIError(
                "MaskVolume.labels must not contain background label 0."
            )
        object.__setattr__(self, "labels", normalized_labels)
        object.__setattr__(
            self,
            "label_names",
            {int(label): str(name) for label, name in self.label_names.items()},
        )

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        *,
        spacing: Optional[Tuple[float, ...]] = None,
        origin: Optional[Tuple[float, ...]] = None,
        direction: Optional[Tuple[float, ...]] = None,
        modality: Optional[str] = None,
        labels: Tuple[int, ...] = (),
        label_names: Optional[Mapping[int, str]] = None,
        subject_id: Optional[str] = None,
        timepoint: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MaskVolume":
        """Create a mask from an array with explicit or inferred nonzero labels."""
        array = np.asarray(data)
        ndim = array.ndim
        return cls(
            data=array,
            spacing=spacing or tuple(1.0 for _ in range(ndim)),
            origin=origin or tuple(0.0 for _ in range(ndim)),
            direction=direction or _default_direction(ndim),
            modality=modality,
            labels=labels,
            label_names=label_names or {},
            subject_id=subject_id,
            timepoint=timepoint,
            metadata=metadata or {},
        )

    @classmethod
    def from_sitk(
        cls,
        image: Any,
        *,
        modality: Optional[str] = None,
        labels: Tuple[int, ...] = (),
        label_names: Optional[Mapping[int, str]] = None,
        subject_id: Optional[str] = None,
        timepoint: Optional[str] = None,
        source: Optional[Union[str, Path]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MaskVolume":
        """Convert a ``SimpleITK.Image`` into a mask while retaining geometry."""
        sitk = _require_simpleitk()
        if not isinstance(image, sitk.Image):
            raise HABITAPIError("image must be a SimpleITK.Image.")
        return cls(
            data=sitk.GetArrayFromImage(image),
            spacing=tuple(image.GetSpacing()),
            origin=tuple(image.GetOrigin()),
            direction=tuple(image.GetDirection()),
            modality=modality,
            labels=labels,
            label_names=label_names or {},
            subject_id=subject_id,
            timepoint=timepoint,
            source=Path(source) if source is not None else None,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ImageMaskPair:
    """Pair one image and mask with the geometry result used by downstream code."""

    image: ImageVolume
    mask: MaskVolume
    geometry_report: Optional[GeometryReport] = None


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


def _coerce_image(image: ImageInput) -> ImageVolume:
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


def _coerce_mask(mask: MaskInput) -> MaskVolume:
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
    image_volume = _coerce_image(image)
    mask_volume = _coerce_mask(mask)
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
