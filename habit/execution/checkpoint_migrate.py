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
"""Migrate v0.1 habitat checkpoint directories to the v1 CheckpointStore layout.

Ownership: ``habit.execution`` (same layer as :class:`CheckpointStore`).
The v0.1 engine wrote ``manifest.json`` + ``subjects/{id}.pkl`` (joblib
``HabitatSubjectData``). The v1 store uses flat ``{sha256}.pkl`` /
``{sha256}.failed`` plus ``run_fingerprint.json``.

Scientific payloads are converted to :class:`~habit.contracts.Supervoxelization`
when geometry and labels are present. Slim Stage-1 pickles that only keep
``supervoxel_df`` cannot rebuild a usable v1 units object; those subjects are
logged and left to recompute, while failed-subject IDs are always migrated so
resume still honours the v0.1 failure-skip rule.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import CompatibilityError

__all__ = [
    "LegacyCheckpointMigrationReport",
    "is_v01_checkpoint_layout",
    "migrate_v01_checkpoint_if_needed",
]

logger = logging.getLogger(__name__)

#: v0.1 on-disk markers (see habit.compat.engines...checkpoint.manager).
_MANIFEST_FILENAME = "manifest.json"
_SUBJECTS_SUBDIR = "subjects"
_FINGERPRINT_FILENAME = "run_fingerprint.json"
_MIGRATION_REPORT_FILENAME = "v01_migration_report.json"
_LEGACY_ARCHIVE_DIRNAME = ".v01_legacy_archive"

_SUCCESS_SUFFIX = ".pkl"
_FAILURE_SUFFIX = ".failed"

#: Failure message written for subjects listed in the v0.1 failed list.
_MIGRATED_FAILURE_MESSAGE = (
    "Migrated from v0.1 checkpoint failure "
    "(subject failed individual-level processing)."
)


@dataclass
class LegacyCheckpointMigrationReport:
    """
    Outcome of one v0.1 → v1 checkpoint migration attempt.

    Attributes:
        migrated: Whether a v0.1 layout was detected and processed.
        completed_migrated: Subject IDs whose scientific payload was written
            as a v1 success entry.
        completed_payload_skipped: Subject IDs listed as completed in the
            v0.1 manifest but whose payload could not be scientifically
            reused (will recompute on resume).
        failed_migrated: Subject IDs written as v1 failure records.
        archive_dir: Directory holding the archived v0.1 tree, or ``None``.
        notes: Human-readable log lines describing conversion limits.
    """

    migrated: bool = False
    completed_migrated: List[str] = field(default_factory=list)
    completed_payload_skipped: List[str] = field(default_factory=list)
    failed_migrated: List[str] = field(default_factory=list)
    archive_dir: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return asdict(self)


def is_v01_checkpoint_layout(root: Union[str, Path]) -> bool:
    """
    Return whether ``root`` holds a v0.1 checkpoint tree.

    Args:
        root: Candidate checkpoint directory.

    Returns:
        ``True`` when ``manifest.json`` or ``subjects/`` is present at the
        root (not inside an archive subdirectory).
    """
    path = Path(root)
    return (path / _MANIFEST_FILENAME).exists() or (path / _SUBJECTS_SUBDIR).exists()


def migrate_v01_checkpoint_if_needed(
    root: Union[str, Path],
    *,
    run_fingerprint: Optional[str] = None,
    clustering_mode: Optional[str] = None,
) -> LegacyCheckpointMigrationReport:
    """
    Detect a v0.1 checkpoint under ``root`` and migrate it to v1 entries.

    Safe to call on every store open: when no v0.1 markers are present this
    is a no-op. After a successful migration the legacy ``manifest.json`` and
    ``subjects/`` tree are moved under ``.v01_legacy_archive/`` so subsequent
    opens see a pure v1 layout.

    Args:
        root: Checkpoint directory (v0.1 and/or v1 files).
        run_fingerprint: Current analysis fingerprint used to build v1 cache
            keys (``habitat.units:{fp}:…`` / ``habitat.one_step:{fp}:…``).
            When ``None``, failure/success IDs are still archived and reported
            but no fingerprint-scoped v1 keys are written.
        clustering_mode: Optional override; when omitted the value from
            ``manifest.json`` is used to choose key prefixes.

    Returns:
        A :class:`LegacyCheckpointMigrationReport` describing what was done.

    Raises:
        CompatibilityError: When the legacy tree is present but
            ``manifest.json`` is corrupt/unreadable.
    """
    path = Path(root)
    report = LegacyCheckpointMigrationReport()
    if not is_v01_checkpoint_layout(path):
        return report

    report.migrated = True
    manifest = _read_v01_manifest(path)
    mode = (clustering_mode or str(manifest.get("clustering_mode") or "")).strip()
    completed = [str(s) for s in manifest.get("completed_subjects") or []]
    failed = [str(s) for s in manifest.get("failed_subjects") or []]

    # Prefer subjects/ on disk when the manifest list is incomplete.
    subjects_dir = path / _SUBJECTS_SUBDIR
    on_disk_completed = _subject_ids_from_subjects_dir(subjects_dir)
    for subject_id in on_disk_completed:
        if subject_id not in completed and subject_id not in failed:
            completed.append(subject_id)

    key_prefixes = _key_prefixes_for(run_fingerprint, mode)
    if run_fingerprint is None:
        note = (
            f"v0.1 checkpoint detected under {path} but no run_fingerprint "
            "was supplied; archiving the legacy tree and recording subject "
            "ID lists only (fingerprint-scoped resume keys were not written)."
        )
        report.notes.append(note)
        logger.warning(note)
    elif not key_prefixes:
        note = (
            f"v0.1 checkpoint under {path}: unable to derive v1 cache-key "
            "prefixes; subject ID lists will be archived only."
        )
        report.notes.append(note)
        logger.warning(note)

    path.mkdir(parents=True, exist_ok=True)

    for subject_id in failed:
        if key_prefixes:
            for prefix in key_prefixes:
                key = f"{prefix}:{subject_id}"
                _write_failure_entry(path, key, _MIGRATED_FAILURE_MESSAGE)
        report.failed_migrated.append(subject_id)

    for subject_id in completed:
        payload, skip_reason = _load_and_convert_subject(
            subjects_dir, subject_id
        )
        if payload is None:
            report.completed_payload_skipped.append(subject_id)
            if skip_reason:
                report.notes.append(skip_reason)
                logger.warning(skip_reason)
            continue
        if key_prefixes:
            for prefix in key_prefixes:
                key = f"{prefix}:{subject_id}"
                _write_success_entry(path, key, payload)
            report.completed_migrated.append(subject_id)
        else:
            report.completed_payload_skipped.append(subject_id)
            note = (
                f"Subject {subject_id!r}: convertible payload loaded but no "
                "fingerprint-scoped key was available; left for recompute."
            )
            report.notes.append(note)
            logger.warning(note)

    if run_fingerprint is not None:
        _write_fingerprint(path, run_fingerprint)

    archive_dir = _archive_legacy_tree(path)
    report.archive_dir = str(archive_dir) if archive_dir is not None else None

    report_path = path / _MIGRATION_REPORT_FILENAME
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info(
        "Migrated v0.1 checkpoint under %s: %s success payload(s) reused, "
        "%s completed subject(s) left to recompute, %s failure record(s) "
        "migrated. Legacy tree archived at %s.",
        path,
        len(report.completed_migrated),
        len(report.completed_payload_skipped),
        len(report.failed_migrated),
        report.archive_dir,
    )
    return report


def _read_v01_manifest(root: Path) -> Dict[str, Any]:
    """
    Load ``manifest.json`` or return an empty dict when only ``subjects/``
    exists.

    Raises:
        CompatibilityError: When the file exists but cannot be parsed.
    """
    manifest_path = root / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        note_subjects = root / _SUBJECTS_SUBDIR
        if note_subjects.exists():
            logger.warning(
                "v0.1 subjects/ present under %s without manifest.json; "
                "migrating from on-disk pickles only.",
                root,
            )
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompatibilityError(
            f"v0.1 checkpoint manifest {manifest_path} is corrupt/unreadable "
            f"({type(exc).__name__}: {exc}). Delete or repair the checkpoint "
            "directory before resuming."
        ) from exc
    if not isinstance(payload, dict):
        raise CompatibilityError(
            f"v0.1 checkpoint manifest {manifest_path} is corrupt: expected "
            f"a JSON object, got {type(payload).__name__}."
        )
    return payload


def _subject_ids_from_subjects_dir(subjects_dir: Path) -> List[str]:
    """Return subject IDs inferred from ``subjects/*.pkl`` filenames."""
    if not subjects_dir.is_dir():
        return []
    ids: List[str] = []
    for path in sorted(subjects_dir.glob("*.pkl")):
        ids.append(path.stem)
    return ids


def _key_prefixes_for(
    run_fingerprint: Optional[str], clustering_mode: str
) -> Tuple[str, ...]:
    """
    Return v1 recipe cache-key prefixes for the given fingerprint and mode.

    When the clustering mode is unknown, both units and one-step prefixes are
    emitted so resume still finds failure/success entries regardless of which
    recipe the user re-runs.
    """
    if not run_fingerprint:
        return ()
    fp = str(run_fingerprint)
    mode = (clustering_mode or "").strip().lower()
    if mode == "one_step":
        return (f"habitat.one_step:{fp}",)
    if mode in {"two_step", "direct_pooling"}:
        return (f"habitat.units:{fp}",)
    return (f"habitat.units:{fp}", f"habitat.one_step:{fp}")


def _digest(key: str) -> str:
    """Return the filesystem-safe digest used by CheckpointStore."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _atomic_pickle_dump(payload: Any, path: Path) -> None:
    """Write ``payload`` atomically (write-then-rename), matching CheckpointStore."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _write_success_entry(root: Path, key: str, value: Any) -> None:
    """Write a v1 success pickle for ``key``."""
    path = root / f"{_digest(key)}{_SUCCESS_SUFFIX}"
    _atomic_pickle_dump(value, path)
    failure_path = root / f"{_digest(key)}{_FAILURE_SUFFIX}"
    failure_path.unlink(missing_ok=True)


def _write_failure_entry(root: Path, key: str, message: str) -> None:
    """Write a v1 failure record for ``key``."""
    path = root / f"{_digest(key)}{_FAILURE_SUFFIX}"
    _atomic_pickle_dump({"key": key, "message": str(message)}, path)


def _write_fingerprint(root: Path, fingerprint: str) -> None:
    """Write ``run_fingerprint.json`` for the migrated store."""
    path = root / _FINGERPRINT_FILENAME
    path.write_text(
        json.dumps({"fingerprint": fingerprint}, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _archive_legacy_tree(root: Path) -> Optional[Path]:
    """
    Move v0.1 ``manifest.json`` / ``subjects/`` under ``.v01_legacy_archive/``.

    Returns:
        The archive directory, or ``None`` when nothing needed moving.
    """
    manifest_path = root / _MANIFEST_FILENAME
    subjects_dir = root / _SUBJECTS_SUBDIR
    if not manifest_path.exists() and not subjects_dir.exists():
        return None

    archive_root = root / _LEGACY_ARCHIVE_DIRNAME
    # Avoid clobbering a previous archive from an interrupted migration.
    destination = archive_root
    suffix = 1
    while destination.exists():
        destination = root / f"{_LEGACY_ARCHIVE_DIRNAME}_{suffix}"
        suffix += 1
    destination.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        shutil.move(str(manifest_path), str(destination / _MANIFEST_FILENAME))
    if subjects_dir.exists():
        shutil.move(str(subjects_dir), str(destination / _SUBJECTS_SUBDIR))
    return destination


def _load_and_convert_subject(
    subjects_dir: Path, subject_id: str
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load one v0.1 subject pickle and convert it when scientifically feasible.

    Returns:
        ``(payload, None)`` on success, or ``(None, reason)`` when the entry
        is missing or cannot be reused as a v1 units object.
    """
    safe_name = str(subject_id).replace("/", "_").replace("\\", "_")
    subject_path = subjects_dir / f"{safe_name}.pkl"
    if not subject_path.is_file():
        return None, (
            f"Subject {subject_id!r}: listed as completed in v0.1 manifest "
            f"but {subject_path.name} is missing; will recompute."
        )

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - joblib is a hard dep
        return None, (
            f"Subject {subject_id!r}: cannot load v0.1 pickle "
            f"(joblib unavailable: {exc}); will recompute."
        )

    try:
        data = joblib.load(subject_path)
    except Exception as exc:
        return None, (
            f"Subject {subject_id!r}: v0.1 pickle unreadable "
            f"({type(exc).__name__}: {exc}); will recompute."
        )

    converted = _try_convert_to_supervoxelization(subject_id, data)
    if converted is not None:
        return converted, None

    type_name = type(data).__name__
    return None, (
        f"Subject {subject_id!r}: v0.1 payload type {type_name!r} cannot be "
        "scientifically reused as a v1 Supervoxelization (slim Stage-1 "
        "pickles often keep only supervoxel_df). Subject ID was recorded; "
        "the subject will be recomputed on resume."
    )


def _try_convert_to_supervoxelization(
    subject_id: str, data: Any
) -> Optional[Any]:
    """
    Best-effort conversion of a v0.1 ``HabitatSubjectData`` to Supervoxelization.

    Requires supervoxel labels (or a reconstructable label volume) plus a
    feature table and geometry metadata in ``mask_info``. Returns ``None``
    when any required piece is missing.
    """
    # Duck-typed: avoid importing HabitatSubjectData (keeps execution free of
    # the compat engine package at import time).
    labels = getattr(data, "supervoxel_labels", None)
    mask_info = getattr(data, "mask_info", None)
    feature_frame = _feature_frame_from_legacy(data)
    if feature_frame is None or labels is None:
        return None
    geometry = _geometry_from_mask_info(mask_info)
    if geometry is None:
        return None

    label_array = np.asarray(labels)
    if label_array.ndim != 3:
        return None

    from habit.contracts.habitat import Supervoxelization
    from habit.contracts.provenance import Provenance

    features = feature_frame.copy()
    # Prefer an explicit supervoxel id column as the index when present.
    for id_col in ("supervoxel", "Supervoxel", "supervoxel_id"):
        if id_col in features.columns:
            features = features.set_index(id_col, drop=True)
            break
    if features.index.name is None:
        features.index = pd.Index(
            np.arange(1, len(features) + 1), name="supervoxel"
        )

    # Drop non-feature bookkeeping columns that v0.1 tables often carry.
    drop_cols = [
        c
        for c in ("subject", "Subject", "count", "Count", "habitats", "Habitats")
        if c in features.columns
    ]
    if drop_cols:
        features = features.drop(columns=drop_cols)

    return Supervoxelization(
        subject_id=str(subject_id),
        label_array=label_array.astype(np.int32, copy=False),
        features=features,
        geometry=geometry,
        provenance=Provenance.source("checkpoint.migrate_v01"),
    )


def _feature_frame_from_legacy(data: Any) -> Optional[pd.DataFrame]:
    """Pick the richest feature table available on a legacy payload."""
    for attr in (
        "supervoxel_df",
        "supervoxel_features",
        "mean_voxel_features",
        "features",
    ):
        frame = getattr(data, attr, None)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            return frame
    return None


def _geometry_from_mask_info(mask_info: Any) -> Optional[Any]:
    """
    Build a :class:`~habit.contracts.Geometry` from v0.1 ``mask_info``.

    Returns:
        Geometry when shape and physical-space fields are recoverable,
        otherwise ``None``.
    """
    if not isinstance(mask_info, dict):
        return None
    mask_array = mask_info.get("mask_array")
    if mask_array is None:
        return None
    array = np.asarray(mask_array)
    if array.ndim != 3:
        return None

    spacing = mask_info.get("spacing")
    origin = mask_info.get("origin")
    direction = mask_info.get("direction")
    if spacing is None or origin is None or direction is None:
        return None

    from habit.contracts.geometry import Geometry

    direction_tuple = tuple(float(v) for v in np.asarray(direction).ravel())
    if len(direction_tuple) != 9:
        return None

    return Geometry(
        shape=tuple(int(v) for v in array.shape),
        spacing=tuple(float(v) for v in spacing),
        origin=tuple(float(v) for v in origin),
        direction=direction_tuple,
    )
