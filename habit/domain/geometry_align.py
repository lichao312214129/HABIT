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
"""Align ROI masks onto image voxel grids inside the domain layer.

HABIT treats geometry as an explicit contract: an image and a mask may only be
combined when they share a voxel grid. Real cohorts often leave Size/Spacing
matched while Origin or Direction differ (for example Direction z = +1 vs -1
after incomplete preprocessing). Raising immediately made users abandon the
tool; the default ``resample_mask`` policy therefore aligns the mask onto the
reference image grid:

* When array shapes already match, voxels are kept and the image's geometry
  metadata is adopted. The labels were almost always painted in the image's
  *index* space; a physical ``Resample`` through a flipped Direction empties
  the ROI even though Size matches.
* When shapes differ, SimpleITK nearest-neighbour resampling regrids the
  labels onto the image FOV.

Registration (ANTs, etc.) is out of scope -- this is grid alignment only.

This module lives in L3 ``habit.domain`` and must not import ``habit.api``.
SimpleITK is imported lazily inside the physical-resample path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from habit.contracts.geometry import Geometry
from habit.contracts.image import ImageVolume, MaskVolume
from habit.contracts.subject import Subject
from habit.exceptions import GeometryError, HABITAPIError
from habit.utils.log_utils import get_module_logger

__all__ = [
    "ON_GEOMETRY_MISMATCH_DEFAULT",
    "ON_GEOMETRY_MISMATCH_VALUES",
    "geometry_mismatch_fields",
    "describe_geometry_mismatch",
    "resample_mask_to_reference",
    "align_mask_to_reference",
    "align_subject_masks",
    "coerce_on_geometry_mismatch",
    "anatomy_aware_field_geometry",
]

_logger = get_module_logger(__name__)

#: Default habitat policy: nearest-neighbour resample mask onto the image grid.
#: Applies to metadata-only drift (spacing/origin/direction); a shape mismatch
#: always raises GeometryError because regridding across physical extents
#: would silently drop or invent mask voxels.
ON_GEOMETRY_MISMATCH_DEFAULT: str = "resample_mask"

#: Allowed ``on_geometry_mismatch`` values (HabitatSpec / YAML / helpers).
ON_GEOMETRY_MISMATCH_VALUES: Tuple[str, ...] = ("resample_mask", "strict")

#: Metadata key written onto a resampled :class:`MaskVolume`.
GEOMETRY_ALIGN_METADATA_KEY: str = "geometry_align"


def coerce_on_geometry_mismatch(value: Optional[str]) -> str:
    """
    Validate and normalise an ``on_geometry_mismatch`` policy string.

    Args:
        value: Policy name, or ``None`` for the default.

    Returns:
        One of :data:`ON_GEOMETRY_MISMATCH_VALUES`.

    Raises:
        HABITAPIError: If ``value`` is not a known policy.
    """
    if value is None:
        return ON_GEOMETRY_MISMATCH_DEFAULT
    normalized = str(value).strip().lower()
    if normalized not in ON_GEOMETRY_MISMATCH_VALUES:
        raise HABITAPIError(
            f"Unknown on_geometry_mismatch {value!r}; expected one of "
            f"{list(ON_GEOMETRY_MISMATCH_VALUES)}."
        )
    return normalized


def geometry_mismatch_fields(
    left: Geometry,
    right: Geometry,
    *,
    tolerance: float = 1e-5,
    direction_tolerance: float = 1e-4,
) -> Tuple[str, ...]:
    """
    List which geometry attributes disagree between two grids.

    Args:
        left: First geometry (typically the image).
        right: Second geometry (typically the mask).
        tolerance: Absolute tolerance for spacing and origin.
        direction_tolerance: Absolute tolerance for direction cosines.

    Returns:
        Ordered field names among ``shape``, ``spacing``, ``origin``,
        ``direction``, and ``frame_of_reference`` that differ.
    """
    mismatches: List[str] = []
    if tuple(left.shape) != tuple(right.shape):
        mismatches.append("shape")
    if not np.allclose(left.spacing, right.spacing, rtol=0.0, atol=tolerance):
        mismatches.append("spacing")
    if not np.allclose(left.origin, right.origin, rtol=0.0, atol=tolerance):
        mismatches.append("origin")
    if not np.allclose(
        left.direction, right.direction, rtol=0.0, atol=direction_tolerance
    ):
        mismatches.append("direction")
    if (
        left.frame_of_reference
        and right.frame_of_reference
        and left.frame_of_reference != right.frame_of_reference
    ):
        mismatches.append("frame_of_reference")
    return tuple(mismatches)


def describe_geometry_mismatch(
    left: Geometry,
    right: Geometry,
    *,
    left_label: str = "image",
    right_label: str = "mask",
    tolerance: float = 1e-5,
    direction_tolerance: float = 1e-4,
) -> str:
    """
    Build a human-readable summary of geometry differences.

    Args:
        left: Reference geometry (image grid).
        right: Mask geometry.
        left_label: Name used for ``left`` in the message.
        right_label: Name used for ``right`` in the message.
        tolerance: Absolute tolerance for spacing and origin.
        direction_tolerance: Absolute tolerance for direction cosines.

    Returns:
        Multi-field summary suitable for logs and error messages.
    """
    fields = geometry_mismatch_fields(
        left,
        right,
        tolerance=tolerance,
        direction_tolerance=direction_tolerance,
    )
    if not fields:
        return f"{left_label} and {right_label} share a compatible voxel grid."
    parts: List[str] = [f"mismatched fields: {', '.join(fields)}"]
    for name in fields:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        parts.append(f"{name}: {left_label}={left_value!r} vs {right_label}={right_value!r}")
    return "; ".join(parts)


def _geometry_to_sitk_reference(geometry: Geometry) -> Any:
    """
    Build an empty SimpleITK image that carries ``geometry``'s grid.

    Args:
        geometry: Target voxel grid (NumPy ``shape`` is ``(z, y, x)``).

    Returns:
        A ``SimpleITK.Image`` with matching size, spacing, origin, direction.
    """
    import SimpleITK as sitk

    # SimpleITK size is (x, y, z); Geometry.shape is NumPy (z, y, x).
    size = tuple(int(v) for v in reversed(tuple(geometry.shape)))
    reference = sitk.Image(size, sitk.sitkUInt8)
    reference.SetSpacing(tuple(float(v) for v in geometry.spacing))
    reference.SetOrigin(tuple(float(v) for v in geometry.origin))
    reference.SetDirection(tuple(float(v) for v in geometry.direction))
    return reference


def _array_geometry_to_sitk(array: np.ndarray, geometry: Geometry) -> Any:
    """
    Wrap a NumPy volume as a SimpleITK image with explicit geometry.

    Args:
        array: Voxel array in NumPy axis order ``(z, y, x)``.
        geometry: Spatial definition of ``array``.

    Returns:
        A ``SimpleITK.Image`` carrying the same voxels and metadata.
    """
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing(tuple(float(v) for v in geometry.spacing))
    image.SetOrigin(tuple(float(v) for v in geometry.origin))
    image.SetDirection(tuple(float(v) for v in geometry.direction))
    return image


def _reference_geometry(
    reference: Union[ImageVolume, Geometry],
) -> Geometry:
    """
    Extract a :class:`Geometry` from an image volume or geometry value.

    Args:
        reference: Image volume or geometry defining the target grid.

    Returns:
        The reference geometry.
    """
    if isinstance(reference, Geometry):
        return reference
    geometry = getattr(reference, "geometry", None)
    if isinstance(geometry, Geometry):
        return geometry
    raise HABITAPIError(
        "reference must be an ImageVolume or Geometry; "
        f"got {type(reference).__name__}."
    )


def _geometry_payload(geometry: Geometry) -> Dict[str, Any]:
    """Serialize a geometry for provenance / logging metadata."""
    return {
        "shape": tuple(int(v) for v in geometry.shape),
        "spacing": tuple(float(v) for v in geometry.spacing),
        "origin": tuple(float(v) for v in geometry.origin),
        "direction": tuple(float(v) for v in geometry.direction),
    }


def _align_event_from_mask_or_subject(
    mask: ImageVolume,
    subject_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[Mapping[str, Any]]:
    """
    Return the ``geometry_align`` event that produced ``mask``, if any.

    Args:
        mask: Possibly aligned ROI mask.
        subject_metadata: Optional ``Subject.metadata`` that stores a list of
            align events when :func:`align_subject_masks` rewrote the mapping.

    Returns:
        One align-event mapping, or ``None`` when the mask was never aligned.
    """
    metadata = getattr(mask, "metadata", None)
    if isinstance(metadata, Mapping):
        event = metadata.get(GEOMETRY_ALIGN_METADATA_KEY)
        if isinstance(event, Mapping) and event.get("action"):
            return event
    if not isinstance(subject_metadata, Mapping):
        return None
    bundle = subject_metadata.get(GEOMETRY_ALIGN_METADATA_KEY)
    if not isinstance(bundle, Mapping):
        return None
    events = bundle.get("events")
    if not isinstance(events, list):
        return None
    roi = getattr(mask, "roi_name", None) or getattr(mask, "modality", None)
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if roi is None or event.get("roi_name") == roi:
            return event
    return None


def anatomy_aware_field_geometry(
    mask: ImageVolume,
    *,
    subject_metadata: Optional[Mapping[str, Any]] = None,
) -> Geometry:
    """
    Geometry a voxel field / habitat map should carry for display.

    Same-shape alignment (:func:`resample_mask_to_reference` ``adopt_geometry``)
    copies the **image** header onto the mask so index-space
    ``is_compatible_with`` checks pass. Demo NRRDs often keep the anatomically
    correct superior/inferior sign only on the original mask (``+z = Inferior``)
    while the intensity header claims LPS identity. Habitat products that
    inherit the adopted image header then flip coronal / sagittal in
    :func:`habit.viz.orientation.resolve_display_geometry`.

    When provenance records a same-shape adopt that changed ``direction``,
    this helper keeps the image index grid (shape / spacing / origin) and
    restores the source-mask direction so :class:`~habit.contracts.habitat.HabitatMap`
    disagrees with the image header and the existing label-wins display rule
    orients superior up.

    Args:
        mask: ROI mask used to build the voxel field (possibly aligned).
        subject_metadata: Optional subject metadata with align events.

    Returns:
        Geometry for :class:`~habit.contracts.habitat.VoxelFeatureField`.
    """
    fallback = mask.geometry
    event = _align_event_from_mask_or_subject(mask, subject_metadata)
    if event is None:
        return fallback
    if event.get("action") != "adopt_geometry":
        return fallback
    mismatches = event.get("mismatches") or []
    if "direction" not in mismatches:
        return fallback
    source = event.get("source_geometry")
    if not isinstance(source, Mapping):
        return fallback
    source_direction = source.get("direction")
    if source_direction is None:
        return fallback
    restored = tuple(float(value) for value in source_direction)
    current = tuple(float(value) for value in fallback.direction)
    if restored == current:
        return fallback
    return Geometry(
        shape=tuple(int(value) for value in fallback.shape),
        spacing=tuple(float(value) for value in fallback.spacing),
        origin=tuple(float(value) for value in fallback.origin),
        direction=restored,
        frame_of_reference=fallback.frame_of_reference,
    )


def resample_mask_to_reference(
    mask: MaskVolume,
    reference: Union[ImageVolume, Geometry],
    *,
    subject_id: Optional[str] = None,
    roi_name: Optional[str] = None,
    reference_label: Optional[str] = None,
) -> MaskVolume:
    """
    Align a label mask onto a reference voxel grid.

    Same-shaped masks keep their voxels and adopt the reference geometry
    (index-space identity) so later ``is_compatible_with`` checks pass.
    :func:`anatomy_aware_field_geometry` then restores the source-mask
    direction onto voxel-field / habitat-map products when that adopt
    changed only the header. Differently shaped masks are nearest-neighbour
    resampled in physical space via SimpleITK.

    Args:
        mask: ROI mask to align.
        reference: Target image volume or geometry.
        subject_id: Optional subject id for logging / metadata.
        roi_name: Optional ROI key for logging / metadata.
        reference_label: Optional modality name of the reference grid.

    Returns:
        A new :class:`MaskVolume` on the reference grid. Metadata records the
        ``geometry_align`` action for provenance.
    """
    target = _reference_geometry(reference)
    mismatches = geometry_mismatch_fields(target, mask.geometry)
    source_array = np.asarray(mask.data)
    if tuple(int(v) for v in source_array.shape) == tuple(int(v) for v in target.shape):
        # Index grids already coincide; Origin/Direction drift is header-only.
        array = source_array
        action = "adopt_geometry"
    else:
        import SimpleITK as sitk

        if isinstance(reference, ImageVolume):
            reference_sitk = _array_geometry_to_sitk(
                np.asarray(reference.data), reference.geometry
            )
        else:
            reference_sitk = _geometry_to_sitk_reference(target)
        mask_sitk = _array_geometry_to_sitk(source_array, mask.geometry)
        resampled = sitk.Resample(
            mask_sitk,
            reference_sitk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            mask_sitk.GetPixelID(),
        )
        array = np.asarray(sitk.GetArrayFromImage(resampled))
        action = "resample_mask"

    align_meta: Dict[str, Any] = {
        "action": action,
        "policy": ON_GEOMETRY_MISMATCH_DEFAULT,
        "mismatches": list(mismatches),
        "reference_label": reference_label,
        "roi_name": roi_name or getattr(mask, "roi_name", None) or mask.modality,
        "subject_id": subject_id or mask.subject_id,
        "reference_geometry": _geometry_payload(target),
        "source_geometry": _geometry_payload(mask.geometry),
    }
    merged_metadata: Dict[str, Any] = dict(mask.metadata or {})
    merged_metadata[GEOMETRY_ALIGN_METADATA_KEY] = align_meta
    return MaskVolume.from_geometry(
        array,
        target,
        roi_name=roi_name or getattr(mask, "roi_name", None) or mask.modality,
        labels=tuple(mask.labels),
        label_names=dict(mask.label_names) if mask.label_names else None,
        subject_id=subject_id or mask.subject_id,
        timepoint=mask.timepoint,
        metadata=merged_metadata,
    )


def align_mask_to_reference(
    mask: MaskVolume,
    reference: Union[ImageVolume, Geometry],
    *,
    on_geometry_mismatch: str = ON_GEOMETRY_MISMATCH_DEFAULT,
    subject_id: Optional[str] = None,
    roi_name: Optional[str] = None,
    reference_label: Optional[str] = None,
) -> MaskVolume:
    """
    Ensure a mask shares the reference grid, resampling by default.

    Args:
        mask: ROI mask.
        reference: Target image volume or geometry.
        on_geometry_mismatch: ``"resample_mask"`` (default) or ``"strict"``.
        subject_id: Optional subject id for logs / errors.
        roi_name: Optional ROI key for logs / errors.
        reference_label: Optional modality name of the reference grid.

    Returns:
        The original mask when already compatible, otherwise a resampled mask
        under the default policy.

    Raises:
        GeometryError: When the shapes differ (any policy -- auto-resampling
            across shapes would silently drop or invent mask voxels), or when
            metadata-only geometries differ and policy is ``strict``.
        HABITAPIError: When the policy string is unknown.
    """
    policy = coerce_on_geometry_mismatch(on_geometry_mismatch)
    target = _reference_geometry(reference)
    if mask.geometry.is_compatible_with(target):
        return mask

    detail = describe_geometry_mismatch(
        target,
        mask.geometry,
        left_label=reference_label or "image",
        right_label=roi_name or "mask",
    )
    sid = subject_id or mask.subject_id or "?"
    roi = roi_name or getattr(mask, "roi_name", None) or mask.modality or "?"
    ref = reference_label or "reference"

    if "shape" in geometry_mismatch_fields(target, mask.geometry):
        raise GeometryError(
            f"subject {sid!r} ROI {roi!r} and modality {ref!r} cover "
            f"different physical extents ({detail}). Auto-resampling across "
            "shapes would silently drop or invent mask voxels, so it is "
            "never applied automatically. Regrid the mask explicitly "
            "(resample_mask_to_reference) or fix preprocessing so image "
            "and mask shapes match."
        )

    if policy == "strict":
        raise GeometryError(
            f"subject {sid!r} ROI {roi!r} and modality {ref!r} do not share "
            f"a compatible voxel grid ({detail}). Set "
            "on_geometry_mismatch: resample_mask (default) to nearest-neighbour "
            "resample the mask onto the image grid, or fix preprocessing so "
            "image and mask geometries match."
        )

    # INFO + habit_console_suppress: keep the audit trail in the log file
    # without flooding the terminal / progress bar (default auto-align path).
    _console_file_only: Dict[str, Any] = {"habit_console_suppress": True}
    same_shape = tuple(mask.geometry.shape) == tuple(target.shape)
    if same_shape:
        _logger.info(
            "Geometry mismatch for subject %r ROI %r vs %r (%s). "
            "Array shapes match, so adopting the image geometry metadata onto "
            "the mask voxels (on_geometry_mismatch=%s). Physical resampling "
            "through a flipped Direction would empty the ROI.",
            sid,
            roi,
            ref,
            detail,
            policy,
            extra=_console_file_only,
        )
    else:
        _logger.info(
            "Geometry mismatch for subject %r ROI %r vs %r (%s). "
            "Nearest-neighbour resampling the mask onto the image grid "
            "(on_geometry_mismatch=%s).",
            sid,
            roi,
            ref,
            detail,
            policy,
            extra=_console_file_only,
        )
    return resample_mask_to_reference(
        mask,
        reference,
        subject_id=subject_id,
        roi_name=roi_name,
        reference_label=reference_label,
    )


def align_subject_masks(
    subject: Subject,
    *,
    on_geometry_mismatch: str = ON_GEOMETRY_MISMATCH_DEFAULT,
    reference_modality: Optional[str] = None,
) -> Subject:
    """
    Align every ROI mask of a subject onto a reference image modality grid.

    Args:
        subject: Subject whose masks may sit on a drifted geometry.
        on_geometry_mismatch: ``"resample_mask"`` (default) or ``"strict"``.
        reference_modality: Image key to use as the grid reference. When
            ``None``, the first modality in insertion order is used.

    Returns:
        The same subject when nothing changed; otherwise a new subject whose
        ``masks`` mapping holds aligned :class:`MaskVolume` instances.

    Raises:
        GeometryError: Under ``strict`` when any mask disagrees with the
            reference image.
        HABITAPIError: When the subject has masks but no images to align to,
            or the policy string is unknown.
    """
    policy = coerce_on_geometry_mismatch(on_geometry_mismatch)
    if not subject.masks:
        return subject
    if not subject.images:
        raise HABITAPIError(
            f"subject {subject.subject_id!r} has ROI masks but no images to "
            "use as a geometry reference."
        )
    ref_key = (
        str(reference_modality)
        if reference_modality is not None
        else next(iter(subject.images))
    )
    if ref_key not in subject.images:
        raise HABITAPIError(
            f"subject {subject.subject_id!r} has no modality {ref_key!r} "
            f"to use as geometry reference; available: {sorted(subject.images)}."
        )
    reference = subject.image(ref_key)
    new_masks: Dict[str, MaskVolume] = {}
    changed = False
    align_events: List[Mapping[str, Any]] = []
    for roi_name in subject.masks:
        mask = subject.mask(roi_name)
        aligned = align_mask_to_reference(
            mask,
            reference,
            on_geometry_mismatch=policy,
            subject_id=subject.subject_id,
            roi_name=roi_name,
            reference_label=ref_key,
        )
        new_masks[roi_name] = aligned
        if aligned is not mask:
            changed = True
            event = (aligned.metadata or {}).get(GEOMETRY_ALIGN_METADATA_KEY)
            if isinstance(event, Mapping):
                align_events.append(event)

    if not changed:
        return subject

    metadata = dict(subject.metadata or {})
    if align_events:
        metadata[GEOMETRY_ALIGN_METADATA_KEY] = {
            "action": policy,
            "reference_modality": ref_key,
            "events": list(align_events),
        }
    return Subject(
        subject_id=subject.subject_id,
        images=subject.images,
        masks=new_masks,
        metadata=metadata,
    )
