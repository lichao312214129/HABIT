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
"""Lazy image references and materialised volume contracts.

This module is the single home of :class:`ImageVolume` / :class:`MaskVolume`.
Operators receive an :class:`ImageRef` and decide when to materialise it, so
small cohorts can stay in memory while large cohorts pass lightweight handles
across process boundaries. File I/O and resampling live in
:mod:`habit.adapters.volume_io`; they return these same classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np

from habit.exceptions import HABITAPIError, OptionalDependencyError
from habit.contracts.geometry import Geometry

__all__ = [
    "GeometryPolicy",
    "GeometryReport",
    "ImageRef",
    "ImageVolume",
    "MaskVolume",
    "ImageMaskPair",
    "ArrayImageRef",
]


class GeometryPolicy(str, Enum):
    """Define how an image/mask geometry mismatch is handled."""

    STRICT = "strict"
    RESAMPLE_MASK = "resample_mask"
    RESAMPLE_IMAGE = "resample_image"
    WARN = "warn"
    HARMONIZE = "harmonize"


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


def _check_geometry_matches_array(array: np.ndarray, geometry: Geometry) -> None:
    """
    Reject a geometry that does not describe the grid of ``array``.

    Catching the mismatch here turns the most common axis-order mistake
    (passing ``shape`` in SimpleITK ``(x, y, z)`` order instead of the NumPy
    ``(z, y, x)`` order) into an explicit message at construction time,
    rather than a wrong-but-silent volume flowing down the pipeline.

    Args:
        array: Voxel array in NumPy axis order ``(z, y, x)``.
        geometry: Geometry expected to describe ``array``.

    Raises:
        HABITAPIError: If ``geometry.shape`` differs from ``array.shape``.
    """
    array_shape = tuple(int(v) for v in array.shape)
    geometry_shape = tuple(int(v) for v in geometry.shape)
    if array_shape != geometry_shape:
        raise HABITAPIError(
            f"geometry.shape {geometry_shape} does not match the array shape "
            f"{array_shape}. Note that shape uses NumPy axis order (z, y, x) "
            "while spacing/origin/direction use SimpleITK axis order (x, y, z)."
        )


@runtime_checkable
class ImageRef(Protocol):
    """
    Lazy handle to volumetric data.

    Operators always receive an ``ImageRef`` and decide when to materialise
    it, which means:

    - small cohorts can stay fully in memory and compose freely;
    - large cohorts pass lightweight handles across process boundaries;
    - third parties can back a subject with PACS, zarr, a torch tensor, or an
      in-memory array by implementing this protocol alone.

    ``ImageVolume`` / ``MaskVolume`` below are the already-materialised
    counterparts and satisfy this protocol structurally (``load()`` returning
    their own array).
    """

    @property
    def geometry(self) -> Geometry:
        """Return grid definition without materialising voxel data."""

    def load(self) -> np.ndarray:
        """Materialise and return the voxel array."""


@dataclass(frozen=True)
class ImageVolume:
    """An image array with explicit physical-space metadata.

    ``data`` follows the NumPy convention used by ``SimpleITK.GetArrayFromImage``.
    Physical metadata remains in SimpleITK axis order. Callers should use this
    object rather than infer geometry from array axes.

    The volume also satisfies :class:`ImageRef`: ``geometry`` returns a
    :class:`~habit.contracts.geometry.Geometry` and ``load()`` returns the
    already-resident array.
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
    def from_geometry(
        cls,
        array: np.ndarray,
        geometry: Geometry,
        *,
        modality: Optional[str] = None,
        subject_id: Optional[str] = None,
        timepoint: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ImageVolume":
        """Build a volume from an array plus a :class:`Geometry` value."""
        values = np.asarray(array)
        _check_geometry_matches_array(values, geometry)
        return cls(
            data=values,
            spacing=tuple(geometry.spacing),
            origin=tuple(geometry.origin),
            direction=tuple(geometry.direction),
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

    @property
    def geometry(self) -> Geometry:
        """Return the spatial definition of this volume without copying data."""
        return Geometry(
            shape=tuple(int(v) for v in self.data.shape),
            spacing=tuple(self.spacing),
            origin=tuple(self.origin),
            direction=tuple(self.direction),
        )

    def load(self) -> np.ndarray:
        """Return the already-resident voxel array (ImageRef conformance)."""
        return cast(np.ndarray, self.data)


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
    def from_geometry(
        cls,
        array: np.ndarray,
        geometry: Geometry,
        *,
        roi_name: Optional[str] = None,
        labels: Tuple[int, ...] = (),
        label_names: Optional[Mapping[int, str]] = None,
        subject_id: Optional[str] = None,
        timepoint: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MaskVolume":
        """Build a mask from a label array plus a :class:`Geometry` value."""
        values = np.asarray(array)
        _check_geometry_matches_array(values, geometry)
        return cls(
            data=values,
            spacing=tuple(geometry.spacing),
            origin=tuple(geometry.origin),
            direction=tuple(geometry.direction),
            modality=roi_name,
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

    def load(self) -> np.ndarray:
        """Return the already-resident label array (ImageRef conformance)."""
        return cast(np.ndarray, self.data)

    @property
    def roi_name(self) -> Optional[str]:
        """Return the ROI name for this mask, mapped from ``modality``."""
        return cast(Optional[str], self.modality)


@dataclass(frozen=True)
class ImageMaskPair:
    """Pair one image and mask with the geometry result used by downstream code."""

    image: ImageVolume
    mask: MaskVolume
    geometry_report: Optional[GeometryReport] = None


@dataclass(frozen=True)
class ArrayImageRef:
    """
    In-memory :class:`ImageRef` backed by a NumPy array.

    This is the reference implementation for custom lazy references: it holds
    the array plus its geometry and materialises trivially.

    Attributes:
        array: Voxel values, NumPy axis order ``(z, y, x)``.
        geometry: Spatial definition of ``array``.
    """

    array: np.ndarray
    geometry: Geometry

    def load(self) -> np.ndarray:
        """Return the held array."""
        return self.array

    def load_volume(self, *, modality: Optional[str] = None) -> ImageVolume:
        """Materialise directly as an :class:`ImageVolume`."""
        return ImageVolume.from_geometry(self.array, self.geometry, modality=modality)
