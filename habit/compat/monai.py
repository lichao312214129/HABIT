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
"""MONAI / PyTorch interop: ``Subject`` <-> MONAI-style dict conversion.

HABIT subject-level operators are plain one-argument callables, which is
exactly what ``monai.transforms.Compose`` chains -- so they drop into MONAI
pipelines directly, and torch's ``DataLoader`` can drive them without HABIT
taking over execution control::

    from monai.data import DataLoader, Dataset
    from monai.transforms import Compose

    # Pattern A: Subjects end-to-end -- the HABIT op IS the transform.
    transform = Compose([to_habitat_map])
    loader = DataLoader(Dataset(list(cohort), transform=transform), num_workers=4)

    # Pattern B: an existing MONAI dict pipeline -- convert, run, write back.
    transform = Compose([
        LoadImaged(keys=["T1", "label"]),
        ...,
        AsDictTransform(to_habitat_map, result_key="habitat_map"),
    ])

    # Pattern C: feed a HABIT cohort into MONAI-style dict tooling.
    transform = Compose([AsMonaiDict(channel_first=True)])
    dataset = Dataset(list(cohort), transform=transform)

The conversions here are deliberately dependency-free: dicts follow MONAI's
canonical layout (one entry per array plus a ``"<key>_meta_dict"`` companion
carrying ``spacing`` / ``origin`` / ``direction`` / ``affine``), produced and
consumed with NumPy only. MONAI itself is an optional dependency; torch
tensors and ``MetaTensor`` values are converted via duck-typing when present.

Axis-order contract: arrays follow HABIT's NumPy convention ``(z, y, x)``;
the ``affine`` follows the ITK/NIfTI ``(x, y, z)`` world convention, exactly
as SimpleITK would report it for the same volume. ``channel_first=True``
prepends MONAI's post-``LoadImage`` channel axis.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from habit.api.exceptions import DataFormatError, HABITAPIError
from habit.contracts.geometry import Geometry
from habit.contracts.image import ArrayImageRef
from habit.contracts.subject import Subject

__all__ = [
    "to_monai_dict",
    "from_monai_dict",
    "AsMonaiDict",
    "FromMonaiDict",
    "AsDictTransform",
]

#: Companion-key suffix of MONAI's canonical dictionary layout.
_META_SUFFIX = "_meta_dict"

#: Keys reserved for the subject identity and free-form attributes.
_SUBJECT_ID_KEY = "subject_id"
_METADATA_KEY = "metadata"


def _as_numpy(value: Any) -> np.ndarray:
    """
    Convert array-likes (NumPy, torch tensors, MetaTensor) to ``ndarray``.

    torch and MONAI are optional dependencies, so conversion is duck-typed:
    ``detach().cpu().numpy()`` for torch tensors, plain ``numpy()`` for
    MetaTensor, and ``np.asarray`` for everything else.

    Args:
        value: Array-like value.

    Returns:
        The value as a NumPy array.
    """
    if isinstance(value, np.ndarray):
        return value
    for chain in (("detach", "cpu", "numpy"), ("cpu", "numpy"), ("numpy",)):
        converted = value
        try:
            for attribute in chain:
                converted = getattr(converted, attribute)()
            return np.asarray(converted)
        except (AttributeError, TypeError):
            continue
    return np.asarray(value)


def _is_array_like(value: Any) -> bool:
    """Report whether ``value`` can serve as a volume array in a dict sample."""
    if isinstance(value, np.ndarray):
        return True
    if hasattr(value, "detach") or hasattr(value, "cpu"):
        return True
    numpy_method = getattr(value, "numpy", None)
    return callable(numpy_method) and not isinstance(value, (list, tuple, str))


def _affine_from_geometry(geometry: Geometry) -> np.ndarray:
    """
    Build the ITK-convention 4x4 affine of a geometry.

    ``direction`` is the row-major 3x3 cosine matrix in SimpleITK ``(x, y, z)``
    order, so ``affine[:3, :3] = direction @ diag(spacing)`` and the last
    column is the origin -- the same affine nibabel reports for the file.

    Args:
        geometry: Grid definition to convert.

    Returns:
        The 4x4 homogeneous affine.
    """
    direction = np.asarray(geometry.direction, dtype=float).reshape(3, 3)
    spacing = np.asarray(geometry.spacing, dtype=float)
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = np.asarray(geometry.origin, dtype=float)
    return affine


def _geometry_from_affine(affine: np.ndarray, shape: Sequence[int]) -> Geometry:
    """
    Recover spacing/origin/direction from a 4x4 affine (inverse of the above).

    Column norms of the rotation-scale block give the spacing; normalising
    the columns gives the direction cosines; the last column is the origin.

    Args:
        affine: 4x4 homogeneous affine, ITK ``(x, y, z)`` convention.
        shape: Voxel grid size, NumPy ``(z, y, x)`` order.

    Returns:
        The reconstructed geometry.
    """
    matrix = np.asarray(affine, dtype=float)
    block = matrix[:3, :3]
    spacing = np.linalg.norm(block, axis=0)
    if np.any(spacing == 0):
        raise DataFormatError("Affine has a zero-norm column; cannot recover spacing.")
    direction = block / spacing
    return Geometry(
        shape=tuple(int(v) for v in shape),
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in matrix[:3, 3]),
        direction=tuple(float(v) for v in direction.reshape(-1)),
    )


def _meta_dict_for(geometry: Geometry) -> Dict[str, Any]:
    """Serialise a geometry into a MONAI-style ``*_meta_dict`` companion."""
    return {
        "spacing": tuple(float(v) for v in geometry.spacing),
        "origin": tuple(float(v) for v in geometry.origin),
        "direction": tuple(float(v) for v in geometry.direction),
        "affine": _affine_from_geometry(geometry),
    }


def _geometry_from_meta(meta: Mapping[str, Any], shape: Sequence[int]) -> Geometry:
    """
    Rebuild a geometry from a ``*_meta_dict`` companion.

    Explicit ``spacing`` / ``origin`` / ``direction`` entries win (they are
    what :func:`to_monai_dict` writes); otherwise the affine is decomposed;
    otherwise an identity grid over ``shape`` is the honest fallback.

    Args:
        meta: The companion mapping, possibly empty.
        shape: Voxel grid size of the array, NumPy order.

    Returns:
        The reconstructed geometry.
    """
    if "spacing" in meta and "origin" in meta and "direction" in meta:
        return Geometry(
            shape=tuple(int(v) for v in shape),
            spacing=tuple(float(v) for v in meta["spacing"]),
            origin=tuple(float(v) for v in meta["origin"]),
            direction=tuple(float(v) for v in meta["direction"]),
        )
    if "affine" in meta:
        return _geometry_from_affine(_as_numpy(meta["affine"]), shape)
    return Geometry.from_array(tuple(int(v) for v in shape))


def to_monai_dict(subject: Subject, *, channel_first: bool = False) -> Dict[str, Any]:
    """
    Convert a HABIT :class:`~habit.contracts.subject.Subject` to a MONAI dict.

    Args:
        subject: The subject to convert. Images and masks are materialised
            (lazy references are read).
        channel_first: Prepend MONAI's post-``LoadImage`` channel axis so
            arrays are ``(1, z, y, x)`` instead of ``(z, y, x)``.

    Returns:
        A dict with one entry per modality and ROI key, a
        ``"<key>_meta_dict"`` companion per entry (MONAI's canonical
        layout), ``"subject_id"`` and ``"metadata"`` keys.
    """
    sample: Dict[str, Any] = {_SUBJECT_ID_KEY: subject.subject_id}
    for modality in subject.images:
        volume = subject.image(modality)
        array = np.asarray(volume.data)
        if channel_first:
            array = array[np.newaxis, ...]
        sample[modality] = array
        sample[f"{modality}{_META_SUFFIX}"] = _meta_dict_for(volume.geometry)
    for roi_name in subject.masks:
        mask = subject.mask(roi_name)
        array = np.asarray(mask.data)
        if channel_first:
            array = array[np.newaxis, ...]
        sample[roi_name] = array
        sample[f"{roi_name}{_META_SUFFIX}"] = _meta_dict_for(mask.geometry)
    sample[_METADATA_KEY] = dict(subject.metadata)
    return sample


def from_monai_dict(
    data: Mapping[str, Any],
    *,
    mask_keys: Sequence[str] = ("label",),
    subject_id: Optional[str] = None,
    squeeze_channel: bool = False,
) -> Subject:
    """
    Rebuild a HABIT :class:`~habit.contracts.subject.Subject` from a MONAI dict.

    Key routing: keys in ``mask_keys`` become masks, ``"<key>_meta_dict"``
    companions supply geometry, ``"subject_id"`` and ``"metadata"`` are
    special, other array-likes become images, and remaining scalars are
    collected into ``Subject.metadata`` so clinical fields survive the trip.

    Args:
        data: MONAI-style sample (NumPy arrays, torch tensors and
            MetaTensors all accepted via duck-typing).
        mask_keys: Keys to materialise as masks (MONAI's ``"label"`` by
            default).
        subject_id: Explicit subject id; wins over ``data["subject_id"]``.
        squeeze_channel: Drop a leading singleton channel axis (inverse of
            ``to_monai_dict(channel_first=True)``).

    Returns:
        The reconstructed subject with in-memory array references.

    Raises:
        DataFormatError: If no subject id is available, a channel axis
            cannot be squeezed, or a mask array is not integer-valued.
    """
    resolved_id = subject_id if subject_id is not None else data.get(_SUBJECT_ID_KEY)
    if resolved_id is None or not str(resolved_id).strip():
        raise DataFormatError(
            "Cannot rebuild a Subject without an id: pass subject_id= or "
            f"include a {_SUBJECT_ID_KEY!r} key in the sample."
        )
    mask_key_set = set(mask_keys)
    images: Dict[str, ArrayImageRef] = {}
    masks: Dict[str, ArrayImageRef] = {}
    metadata: Dict[str, Any] = dict(data.get(_METADATA_KEY, {}))
    for key, value in data.items():
        if key.endswith(_META_SUFFIX) or key in (_SUBJECT_ID_KEY, _METADATA_KEY):
            continue
        if not _is_array_like(value):
            metadata.setdefault(key, value)
            continue
        array = _as_numpy(value)
        if squeeze_channel:
            if array.ndim < 1 or array.shape[0] != 1:
                raise DataFormatError(
                    f"Entry {key!r} has shape {array.shape}; expected a "
                    "leading singleton channel axis."
                )
            array = array[0]
        meta = data.get(f"{key}{_META_SUFFIX}", {})
        geometry = _geometry_from_meta(meta, array.shape)
        if key in mask_key_set:
            if not np.issubdtype(array.dtype, np.integer):
                rounded = np.rint(array)
                if not np.allclose(array, rounded):
                    raise DataFormatError(
                        f"Mask entry {key!r} is not integer-valued; refusing "
                        "to silently truncate it."
                    )
                array = rounded
            array = array.astype(np.int32)
            masks[key] = ArrayImageRef(array=array, geometry=geometry)
        else:
            images[key] = ArrayImageRef(array=array, geometry=geometry)
    return Subject(
        subject_id=str(resolved_id),
        images=images,
        masks=masks,
        metadata=metadata,
    )


class AsMonaiDict:
    """
    Transform form of :func:`to_monai_dict` (Subject -> MONAI dict).

    A plain callable, so ``monai.transforms.Compose`` accepts it as-is; no
    MONAI import is required. Typical role: the last HABIT-side transform
    before MONAI-native dict tooling takes over (e.g. torch collation).

    Args:
        channel_first: Prepend MONAI's channel axis to every array.
    """

    def __init__(self, *, channel_first: bool = False) -> None:
        self.channel_first = channel_first

    def __call__(self, subject: Subject) -> Dict[str, Any]:
        """Convert one subject; see :func:`to_monai_dict`."""
        if not isinstance(subject, Subject):
            raise HABITAPIError(
                f"AsMonaiDict expects a habit Subject; got {type(subject).__name__}."
            )
        return to_monai_dict(subject, channel_first=self.channel_first)


class FromMonaiDict:
    """
    Transform form of :func:`from_monai_dict` (MONAI dict -> Subject).

    Args:
        mask_keys: Keys to materialise as masks.
        squeeze_channel: Drop a leading singleton channel axis.
    """

    def __init__(
        self,
        *,
        mask_keys: Sequence[str] = ("label",),
        squeeze_channel: bool = False,
    ) -> None:
        self.mask_keys = tuple(mask_keys)
        self.squeeze_channel = squeeze_channel

    def __call__(self, data: Mapping[str, Any]) -> Subject:
        """Convert one sample; see :func:`from_monai_dict`."""
        return from_monai_dict(
            data,
            mask_keys=self.mask_keys,
            squeeze_channel=self.squeeze_channel,
        )


class AsDictTransform:
    """
    Wrap a HABIT subject-level operator as a dict -> dict MONAI transform.

    The wrapped operator receives the sample rebuilt as a
    :class:`~habit.contracts.subject.Subject`; with ``result_key`` the result
    is written back into the sample (the MONAI convention), otherwise the
    result replaces the sample outright.

    Args:
        op: Any HABIT subject-level operator (``op(subject) -> result``),
            e.g. a :class:`~habit.domain.pipeline.SubjectPipeline`.
        result_key: Dict key receiving the operator's result. ``None``
            returns the raw result instead of an updated dict.
        mask_keys: Forwarded to :func:`from_monai_dict`.
        squeeze_channel: Forwarded to :func:`from_monai_dict`.
    """

    def __init__(
        self,
        op: Callable[[Subject], Any],
        *,
        result_key: Optional[str] = None,
        mask_keys: Sequence[str] = ("label",),
        squeeze_channel: bool = False,
    ) -> None:
        if not callable(op):
            raise HABITAPIError(
                f"AsDictTransform wraps a callable operator; got {type(op).__name__}."
            )
        self.op = op
        self.result_key = result_key
        self.mask_keys = tuple(mask_keys)
        self.squeeze_channel = squeeze_channel

    def __call__(self, data: Mapping[str, Any]) -> Any:
        """Run the wrapped operator on one dict sample."""
        subject = from_monai_dict(
            data,
            mask_keys=self.mask_keys,
            squeeze_channel=self.squeeze_channel,
        )
        result = self.op(subject)
        if self.result_key is None:
            return result
        updated = dict(data)
        updated[self.result_key] = result
        return updated
