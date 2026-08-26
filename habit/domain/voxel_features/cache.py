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
"""On-disk cache for voxel radiomics fields.

The scientific payload of ``voxel_radiomics`` depends on the ROI, kernel
radius, bin width and enabled features -- not on ``voxel_batch`` or the
Torch device. Those execution knobs are therefore left out of the cache
key so a later run with a larger batch can reuse an earlier extraction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from habit.contracts.habitat import VoxelFeatureField
from habit.exceptions import CompatibilityError

__all__ = [
    "voxel_radiomics_cache_key",
    "voxel_radiomics_cache_path",
    "load_cached_voxel_field",
    "save_cached_voxel_field",
]

_LOG = logging.getLogger(__name__)
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Stable JSON for hashing (sorted keys, no whitespace drift)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def voxel_radiomics_cache_key(
    subject_id: str,
    *,
    kernel_radius: int,
    roi: Optional[str],
    modalities: Any,
    params: Optional[Mapping[str, Any]],
    params_file: Optional[str],
    output_float32: bool,
    crop_to_roi: bool,
    modality: Optional[str] = None,
    as_: Optional[str] = None,
) -> str:
    """
    Hash the settings that change voxel-radiomics numbers.

    Args:
        subject_id: Subject whose ROI was extracted.
        kernel_radius: Sliding-window radius in voxels.
        roi: Mask key, or ``None`` for the subject's only mask.
        modalities: Requested modality list (may be empty before resolve).
        params: Inline PyRadiomics mapping, or ``None``.
        params_file: Parameter YAML path, or ``None``.
        output_float32: Whether values were stored as float32.
        crop_to_roi: Whether the extractor cropped to the ROI bbox.
        modality: Singular modality form, when used.
        as_: Column alias, when used.

    Returns:
        Hex SHA-256 digest of the canonical payload.
    """
    payload = {
        "as_": as_,
        "crop_to_roi": bool(crop_to_roi),
        "kernel_radius": int(kernel_radius),
        "modalities": list(modalities) if modalities is not None else [],
        "modality": modality,
        "output_float32": bool(output_float32),
        "params": params if params is not None else {},
        "params_file": params_file,
        "roi": roi,
        "subject_id": str(subject_id),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def voxel_radiomics_cache_path(
    cache_dir: Union[str, Path],
    subject_id: str,
    cache_key: str,
) -> Path:
    """
    Return the archive path for one subject and one settings hash.

    Args:
        cache_dir: Directory that holds cached fields.
        subject_id: Owning subject.
        cache_key: Digest from :func:`voxel_radiomics_cache_key`.

    Returns:
        ``<cache_dir>/<safe_subject>__<12-char-key>.vxff.zip``.
    """
    safe = _SAFE_ID.sub("_", str(subject_id)).strip("._") or "subject"
    return Path(cache_dir) / f"{safe}__{cache_key[:12]}.vxff.zip"


def load_cached_voxel_field(
    cache_dir: Union[str, Path],
    subject_id: str,
    cache_key: str,
) -> Optional[VoxelFeatureField]:
    """
    Load a cached field when the archive exists and is readable.

    Args:
        cache_dir: Directory that holds cached fields.
        subject_id: Owning subject (must match the archive).
        cache_key: Digest from :func:`voxel_radiomics_cache_key`.

    Returns:
        The stored field, or ``None`` on a miss or a corrupt archive.
    """
    path = voxel_radiomics_cache_path(cache_dir, subject_id, cache_key)
    if not path.is_file():
        return None
    try:
        field = VoxelFeatureField.load(path)
    except (OSError, CompatibilityError, KeyError, ValueError) as exc:
        _LOG.warning("voxel_radiomics cache unreadable (%s): %s", path, exc)
        return None
    if str(field.subject_id) != str(subject_id):
        _LOG.warning(
            "voxel_radiomics cache subject mismatch at %s: stored %r, asked %r",
            path,
            field.subject_id,
            subject_id,
        )
        return None
    _LOG.info("voxel_radiomics cache hit: %s", path.name)
    return field


def save_cached_voxel_field(
    cache_dir: Union[str, Path],
    cache_key: str,
    field: VoxelFeatureField,
) -> Path:
    """
    Write ``field`` under the cache key (atomic replace).

    Args:
        cache_dir: Directory that holds cached fields.
        cache_key: Digest from :func:`voxel_radiomics_cache_key`.
        field: Extracted field to store.

    Returns:
        Path of the written archive.
    """
    path = voxel_radiomics_cache_path(cache_dir, field.subject_id, cache_key)
    written = field.save(path)
    _LOG.info("voxel_radiomics cache write: %s", written.name)
    return written
