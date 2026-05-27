"""
Checkpoint storage and manifest management for resumable habitat train/predict runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import joblib

from ..config_schemas import HabitatAnalysisConfig
from ..pipelines.habitat_subject_data import HabitatSubjectData

CHECKPOINT_VERSION: int = 1
MANIFEST_FILENAME: str = "manifest.json"
SUBJECTS_SUBDIR: str = "subjects"
TRAIN_CHECKPOINT_DIRNAME: str = ".habitat_checkpoint"
PREDICT_CHECKPOINT_DIRNAME: str = ".habitat_predict_checkpoint"


@dataclass
class CheckpointManifest:
    """On-disk metadata for a resumable habitat train or predict run."""

    version: int = CHECKPOINT_VERSION
    config_hash: str = ""
    individual_config_hash: str = ""
    clustering_mode: str = ""
    run_mode: str = "train"
    completed_subjects: List[str] = field(default_factory=list)
    failed_subjects: List[str] = field(default_factory=list)
    stage: str = "individual"


def _hash_config_payload(payload: Dict[str, Any]) -> str:
    """
    Hash a canonical JSON payload for checkpoint config fingerprints.

    Args:
        payload: Serializable configuration dictionary.

    Returns:
        16-character hex digest prefix.
    """
    canonical: str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_individual_stage_config_payload(
    config: HabitatAnalysisConfig,
) -> Dict[str, Any]:
    """
    Build the Stage-1 (individual-level checkpoint) configuration fingerprint.

    Only fields that affect per-subject processing before ``checkpoint_save`` are
    included. Group-level preprocessing and group habitat clustering are excluded
    because they run after the checkpoint boundary.

    Args:
        config: Validated habitat analysis configuration.

    Returns:
        Dictionary payload used for individual-stage config hashing.
    """
    feature_payload: Optional[Dict[str, Any]] = None
    if config.FeatureConstruction is not None:
        feature_dump = config.FeatureConstruction.model_dump()
        # Group preprocessing runs in Stage 2; changing it must not invalidate pkl cache.
        feature_dump.pop("preprocessing_for_group_level", None)
        feature_payload = feature_dump

    habitat_seg_payload: Optional[Dict[str, Any]] = None
    if config.HabitatSegmentation is not None:
        hs = config.HabitatSegmentation
        mode = hs.clustering_mode
        habitat_seg_payload = {"clustering_mode": mode}
        if mode == "direct_pooling":
            # Stage 1: voxel features + subject preprocessing only.
            pass
        elif mode == "two_step":
            habitat_seg_payload["supervoxel"] = hs.supervoxel.model_dump()
            habitat_seg_payload["postprocess_supervoxel"] = (
                hs.postprocess_supervoxel.model_dump()
            )
        elif mode == "one_step":
            habitat_seg_payload["supervoxel"] = hs.supervoxel.model_dump()
            habitat_seg_payload["habitat"] = hs.habitat.model_dump()
            habitat_seg_payload["postprocess_habitat"] = (
                hs.postprocess_habitat.model_dump()
            )

    return {
        "data_dir": config.data_dir,
        "FeatureConstruction": feature_payload,
        "HabitatSegmentation": habitat_seg_payload,
    }


def build_legacy_full_config_payload(config: HabitatAnalysisConfig) -> Dict[str, Any]:
    """
    Build the pre-refactor full-configuration payload (for backward compatibility).

    Args:
        config: Validated habitat analysis configuration.

    Returns:
        Legacy dictionary payload that included group-stage fields in the hash.
    """
    return {
        "data_dir": config.data_dir,
        "FeatureConstruction": (
            config.FeatureConstruction.model_dump()
            if config.FeatureConstruction is not None
            else None
        ),
        "HabitatSegmentation": (
            config.HabitatSegmentation.model_dump()
            if config.HabitatSegmentation is not None
            else None
        ),
    }


def compute_config_hash(config: HabitatAnalysisConfig) -> str:
    """
    Build a stable hash of Stage-1 (checkpoint) configuration fields.

    Args:
        config: Validated habitat analysis configuration.

    Returns:
        Hex digest prefix used to detect individual-stage config changes on resume.
    """
    return _hash_config_payload(build_individual_stage_config_payload(config))


def resolve_predict_pipeline_path(config: HabitatAnalysisConfig) -> Path:
    """
    Resolve the trained pipeline path used for predict-mode checkpoint hashing.

    Args:
        config: Active habitat analysis configuration with ``pipeline_path`` set.

    Returns:
        Absolute path to the serialized pipeline file.

    Raises:
        ValueError: When ``pipeline_path`` is missing.
        FileNotFoundError: When the resolved pipeline file does not exist.
    """
    raw_path = getattr(config, "pipeline_path", None)
    if not raw_path:
        raise ValueError(
            "pipeline_path is required for predict-mode checkpoint fingerprinting."
        )

    path = Path(str(raw_path))
    if not path.is_absolute() and config.config_file:
        path = Path(config.config_file).resolve().parent / path
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Predict pipeline file not found for checkpoint hashing: {resolved}"
        )
    return resolved


def _pipeline_file_fingerprint(path: Path) -> Dict[str, Any]:
    """
    Build a stable fingerprint for a serialized pipeline file on disk.

    Args:
        path: Absolute path to ``habitat_pipeline.pkl``.

    Returns:
        Dictionary with resolved path, byte size, and modification time.
    """
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_predict_stage_config_payload(
    config: HabitatAnalysisConfig,
) -> Dict[str, Any]:
    """
    Build the Stage-1 predict checkpoint configuration fingerprint.

    Predict resume must invalidate when the trained pipeline file or input
    manifest changes, not when group-stage-only YAML drifts.

    Args:
        config: Validated habitat analysis configuration for a predict run.

    Returns:
        Dictionary payload used for predict-mode config hashing.

    Raises:
        ValueError: When ``pipeline_path`` is missing.
        FileNotFoundError: When the pipeline file does not exist.
    """
    pipeline_path = resolve_predict_pipeline_path(config)
    clustering_mode = (
        config.HabitatSegmentation.clustering_mode
        if config.HabitatSegmentation is not None
        else ""
    )
    return {
        "run_mode": "predict",
        "data_dir": config.data_dir,
        "pipeline": _pipeline_file_fingerprint(pipeline_path),
        "clustering_mode": clustering_mode,
    }


def compute_predict_config_hash(config: HabitatAnalysisConfig) -> str:
    """
    Build a stable hash of predict Stage-1 checkpoint configuration fields.

    Args:
        config: Validated habitat analysis configuration for a predict run.

    Returns:
        Hex digest prefix used to detect predict resume invalidation.
    """
    return _hash_config_payload(build_predict_stage_config_payload(config))


def compute_legacy_config_hash(config: HabitatAnalysisConfig) -> str:
    """
    Build the legacy full-configuration hash (pre Stage-1-only fingerprint).

    Args:
        config: Validated habitat analysis configuration.

    Returns:
        Legacy hex digest prefix for backward-compatible resume checks.
    """
    return _hash_config_payload(build_legacy_full_config_payload(config))


class CheckpointConfigHashError(RuntimeError):
    """
    Raised when resume=True, strict_checkpoint_hash=True, and the on-disk
    checkpoint manifest is incompatible with the active configuration.
    """


def is_checkpoint_config_compatible(
    stored_hash: str,
    config: HabitatAnalysisConfig,
    *,
    manifest: Optional[CheckpointManifest] = None,
) -> bool:
    """
    Decide whether an on-disk checkpoint manifest matches the active config.

    Compatible when:
    - Stored hash equals the Stage-1 fingerprint, or
    - Stored hash equals the legacy full-config fingerprint (unchanged YAML), or
    - Manifest records ``individual_config_hash`` matching the Stage-1 fingerprint, or
    - Legacy v1 manifest (no individual hash) with cached subjects: allow reuse when
      all completed pickles exist (typical group-stage-only drift; logs a warning).

    Args:
        stored_hash: ``config_hash`` value read from ``manifest.json``.
        config: Active habitat analysis configuration.
        manifest: Optional loaded manifest for extended compatibility checks.

    Returns:
        True when individual-level checkpoint data may be reused.
    """
    individual_hash = compute_config_hash(config)
    if stored_hash in (individual_hash, compute_legacy_config_hash(config)):
        return True

    if manifest is not None:
        stored_individual = getattr(manifest, "individual_config_hash", "") or ""
        if stored_individual and stored_individual == individual_hash:
            return True

        # Legacy v1 manifests only stored full-config hash. When group-stage YAML
        # changed but Stage-1 settings are unchanged, strict hash checks fail even
        # though cached pkls remain valid. Trust existing pickles when present.
        if (
            not stored_individual
            and manifest.completed_subjects
            and manifest.clustering_mode
            == (
                config.HabitatSegmentation.clustering_mode
                if config.HabitatSegmentation is not None
                else ""
            )
        ):
            return True

    return False


def is_predict_checkpoint_config_compatible(
    stored_hash: str,
    config: HabitatAnalysisConfig,
    *,
    manifest: Optional[CheckpointManifest] = None,
) -> bool:
    """
    Decide whether an on-disk predict checkpoint matches the active predict run.

    Args:
        stored_hash: ``config_hash`` value read from ``manifest.json``.
        config: Active habitat analysis configuration.
        manifest: Optional loaded manifest for run-mode validation.

    Returns:
        True when predict individual-level checkpoint data may be reused.
    """
    try:
        current_hash = compute_predict_config_hash(config)
    except (ValueError, FileNotFoundError):
        return False

    if stored_hash != current_hash:
        return False

    if manifest is not None:
        stored_run_mode = getattr(manifest, "run_mode", "train") or "train"
        if stored_run_mode != "predict":
            return False

    return True


class HabitatTrainCheckpoint:
    """
    Manage per-subject individual-level checkpoints for habitat train/predict runs.

    Args:
        checkpoint_dir: Root directory for manifest and subject cache files.
        config: Active habitat analysis configuration.
        logger: Optional logger for resume warnings and progress messages.
        run_mode: ``"train"`` or ``"predict"``; controls hashing and manifest metadata.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        config: HabitatAnalysisConfig,
        logger: Optional[logging.Logger] = None,
        *,
        run_mode: str = "train",
    ) -> None:
        if run_mode not in {"train", "predict"}:
            raise ValueError(f"Unsupported checkpoint run_mode: {run_mode!r}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.subjects_dir = self.checkpoint_dir / SUBJECTS_SUBDIR
        self.manifest_path = self.checkpoint_dir / MANIFEST_FILENAME
        self.config = config
        self.logger = logger
        self.run_mode = run_mode
        if run_mode == "predict":
            self.config_hash = compute_predict_config_hash(config)
        else:
            self.config_hash = compute_config_hash(config)
        self.clustering_mode = (
            config.HabitatSegmentation.clustering_mode
            if config.HabitatSegmentation is not None
            else ""
        )
        self.manifest = CheckpointManifest(
            config_hash=self.config_hash,
            individual_config_hash=self.config_hash,
            clustering_mode=self.clustering_mode,
            run_mode=run_mode,
        )

    @staticmethod
    def resolve_checkpoint_dir(
        out_dir: str,
        checkpoint_dir: Optional[str],
        *,
        run_mode: str = "train",
    ) -> Path:
        """
        Resolve the checkpoint directory from config overrides.

        Args:
            out_dir: Pipeline output directory.
            checkpoint_dir: Optional explicit checkpoint path from config.
            run_mode: ``"train"`` or ``"predict"``; selects the default subdirectory.

        Returns:
            Absolute checkpoint directory path.
        """
        if checkpoint_dir:
            return Path(checkpoint_dir)
        subdir = (
            PREDICT_CHECKPOINT_DIRNAME
            if run_mode == "predict"
            else TRAIN_CHECKPOINT_DIRNAME
        )
        return Path(out_dir) / subdir

    def initialize_for_run(self, *, resume: bool) -> None:
        """
        Prepare checkpoint storage for a training run.

        When ``resume`` is False, any existing checkpoint directory is removed.
        When ``resume`` is True, an existing manifest is loaded; if the stored
        config hash differs from the current config, a warning is logged and the
        checkpoint is cleared before restarting, unless
        ``config.strict_checkpoint_hash`` is True (then
        :class:`CheckpointConfigHashError` is raised instead).

        Args:
            resume: Whether to reuse completed subjects from disk.
        """
        strict_hash: bool = bool(getattr(self.config, "strict_checkpoint_hash", True))

        if not resume:
            if self.checkpoint_dir.exists():
                shutil.rmtree(self.checkpoint_dir)
            self._ensure_dirs()
            self._write_manifest()
            return

        if not self.manifest_path.exists():
            self._ensure_dirs()
            self._write_manifest()
            if self.logger:
                self.logger.info(
                    "Resume enabled but no checkpoint manifest found; starting fresh."
                )
            return

        loaded = self._read_manifest()
        loaded_run_mode = getattr(loaded, "run_mode", "train") or "train"
        if loaded_run_mode != self.run_mode:
            message = (
                f"Checkpoint run_mode mismatch ({loaded_run_mode} -> {self.run_mode}); "
                "cannot resume with strict_checkpoint_hash=True."
            )
            if strict_hash:
                raise CheckpointConfigHashError(message)
            if self.logger:
                self.logger.warning(
                    "Checkpoint run_mode mismatch (%s -> %s); "
                    "discarding checkpoint and restarting all subjects.",
                    loaded_run_mode,
                    self.run_mode,
                )
            shutil.rmtree(self.checkpoint_dir)
            self._ensure_dirs()
            self._write_manifest()
            return

        if self.run_mode == "predict":
            compatible = is_predict_checkpoint_config_compatible(
                loaded.config_hash,
                self.config,
                manifest=loaded,
            )
        else:
            compatible = is_checkpoint_config_compatible(
                loaded.config_hash,
                self.config,
                manifest=loaded,
            )

        if not compatible:
            message = (
                f"Checkpoint config hash changed ({loaded.config_hash} -> "
                f"{self.config_hash}); cannot resume with strict_checkpoint_hash=True."
            )
            if strict_hash:
                raise CheckpointConfigHashError(message)
            if self.logger:
                self.logger.warning(
                    "Checkpoint config hash changed (%s -> %s); "
                    "discarding checkpoint and restarting all subjects.",
                    loaded.config_hash,
                    self.config_hash,
                )
            shutil.rmtree(self.checkpoint_dir)
            self._ensure_dirs()
            self._write_manifest()
            return

        if (
            self.run_mode == "train"
            and loaded.config_hash != self.config_hash
            and self.logger is not None
        ):
            self.logger.warning(
                "Checkpoint hash mismatch but Stage-1 cache remains valid "
                "(typical when only group-stage preprocessing/clustering changed). "
                "Migrating manifest hash %s -> %s.",
                loaded.config_hash,
                self.config_hash,
            )

        self.manifest = loaded
        self.manifest.config_hash = self.config_hash
        self.manifest.individual_config_hash = self.config_hash
        self.manifest.run_mode = self.run_mode
        self._ensure_dirs()
        if self.logger:
            self.logger.info(
                "Loaded %s checkpoint: %s completed, %s failed subject(s).",
                self.run_mode,
                len(self.manifest.completed_subjects),
                len(self.manifest.failed_subjects),
            )

    def pending_subjects(
        self,
        all_subjects: Iterable[str],
        *,
        resume: bool,
        force_rerun_subjects: Optional[Iterable[str]] = None,
        retry_failed_subjects: bool = False,
    ) -> List[str]:
        """
        Return subject IDs that still need individual-level processing.

        Args:
            all_subjects: Full subject list for the current run.
            resume: Whether completed/failed subjects should be skipped.
            force_rerun_subjects: Subject IDs to remove from cache and reprocess.
            retry_failed_subjects: When True, also reprocess manifest failed subjects.

        Returns:
            Ordered list of subject IDs to dispatch to parallel workers.
        """
        subject_list = list(all_subjects)
        if not resume:
            return subject_list

        subject_set = set(subject_list)
        force_rerun: Set[str] = set(force_rerun_subjects or [])
        if retry_failed_subjects:
            force_rerun.update(
                subject_id
                for subject_id in self.manifest.failed_subjects
                if subject_id in subject_set
            )

        for subject_id in force_rerun:
            self._remove_completed_subject(subject_id)
            if subject_id in self.manifest.failed_subjects:
                self.manifest.failed_subjects.remove(subject_id)
            self._write_manifest()

        completed: Set[str] = set(self.manifest.completed_subjects)
        failed: Set[str] = set(self.manifest.failed_subjects)
        pending = [
            subject_id
            for subject_id in subject_list
            if subject_id not in completed and subject_id not in failed
        ]
        return pending

    def load_completed_results(self) -> Dict[str, HabitatSubjectData]:
        """
        Load cached individual-level results for completed subjects.

        Returns:
            Mapping of subject ID to cached ``HabitatSubjectData``.
        """
        results: Dict[str, HabitatSubjectData] = {}
        for subject_id in self.manifest.completed_subjects:
            subject_path = self._subject_path(subject_id)
            if not subject_path.exists():
                if self.logger:
                    self.logger.warning(
                        "Checkpoint entry missing for completed subject %s; "
                        "it will be reprocessed.",
                        subject_id,
                    )
                continue
            try:
                results[subject_id] = self.load_subject_pkl(subject_id)
            except (FileNotFoundError, TypeError) as exc:
                if self.logger:
                    self.logger.warning(
                        "Skipping invalid checkpoint entry for subject %s: %s",
                        subject_id,
                        exc,
                    )
        return results

    def load_subject_pkl(self, subject_id: str) -> HabitatSubjectData:
        """
        Load one subject's cached individual-level result from disk.

        Args:
            subject_id: Subject identifier.

        Returns:
            Deserialized ``HabitatSubjectData`` for the subject.

        Raises:
            FileNotFoundError: When the subject pickle does not exist.
            TypeError: When the pickle does not contain ``HabitatSubjectData``.
        """
        subject_path = self._subject_path(subject_id)
        if not subject_path.exists():
            raise FileNotFoundError(
                f"Checkpoint pickle missing for subject '{subject_id}': {subject_path}"
            )
        data = joblib.load(subject_path)
        if not isinstance(data, HabitatSubjectData):
            raise TypeError(
                f"Checkpoint pickle for subject '{subject_id}' contains "
                f"{type(data).__name__}, expected HabitatSubjectData."
            )
        return data

    def record_success(self, subject_id: str, data: HabitatSubjectData) -> None:
        """
        Persist one successful subject result and update the manifest.

        Args:
            subject_id: Subject identifier.
            data: Individual-level pipeline output for the subject.
        """
        self.save_subject_pkl(subject_id, data)
        self.record_success_manifest(subject_id)

    def save_subject_pkl(self, subject_id: str, data: HabitatSubjectData) -> None:
        """
        Write one subject pickle without updating the manifest.

        Called from :class:`CheckpointSaveStep` inside worker processes.

        Args:
            subject_id: Subject identifier.
            data: Individual-level pipeline output for the subject.
        """
        self._ensure_dirs()
        joblib.dump(data, self._subject_path(subject_id))

    def record_success_manifest(self, subject_id: str) -> None:
        """
        Mark one subject as completed in the manifest only.

        The subject pickle is expected to exist already (written by
        :class:`CheckpointSaveStep` in the worker).

        Args:
            subject_id: Subject identifier.
        """
        if subject_id not in self.manifest.completed_subjects:
            self.manifest.completed_subjects.append(subject_id)
        if subject_id in self.manifest.failed_subjects:
            self.manifest.failed_subjects.remove(subject_id)
        self.manifest.stage = "individual"
        self._write_manifest()

    def record_failure(self, subject_id: str) -> None:
        """
        Mark one subject as failed without deleting prior successful cache.

        Args:
            subject_id: Subject identifier that failed individual processing.
        """
        if subject_id not in self.manifest.failed_subjects:
            self.manifest.failed_subjects.append(subject_id)
        if subject_id in self.manifest.completed_subjects:
            self.manifest.completed_subjects.remove(subject_id)
        self._remove_completed_subject(subject_id)
        self.manifest.stage = "individual"
        self._write_manifest()

    def requeue_subjects(self, subject_ids: Iterable[str]) -> List[str]:
        """
        Clear checkpoint state so subjects can be re-dispatched in a retry pass.

        Removes each subject from ``failed_subjects`` and ``completed_subjects``,
        deletes any cached pickle, and persists the manifest once.

        Args:
            subject_ids: Subject identifiers to prepare for another parallel pass.

        Returns:
            Ordered list of subject IDs that were requeued (deduplicated).
        """
        requeued: List[str] = []
        seen: Set[str] = set()
        changed = False

        for subject_id in subject_ids:
            subject_key = str(subject_id)
            if subject_key in seen:
                continue
            seen.add(subject_key)

            if subject_key in self.manifest.failed_subjects:
                self.manifest.failed_subjects.remove(subject_key)
                changed = True
            if (
                subject_key in self.manifest.completed_subjects
                or self._subject_path(subject_key).exists()
            ):
                self._remove_completed_subject(subject_key)
                changed = True

            requeued.append(subject_key)

        if changed:
            self.manifest.stage = "individual"
            self._write_manifest()

        return requeued

    def mark_training_complete(self) -> None:
        """Mark the checkpoint as fully completed before optional cleanup."""
        self.manifest.stage = "done"
        self.manifest.run_mode = self.run_mode
        self._write_manifest()

    def clear(self) -> None:
        """Remove all checkpoint files from disk."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)

    def _ensure_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.subjects_dir.mkdir(parents=True, exist_ok=True)

    def _subject_path(self, subject_id: str) -> Path:
        safe_name = str(subject_id).replace("/", "_").replace("\\", "_")
        return self.subjects_dir / f"{safe_name}.pkl"

    def _remove_completed_subject(self, subject_id: str) -> None:
        subject_path = self._subject_path(subject_id)
        if subject_path.exists():
            subject_path.unlink()
        if subject_id in self.manifest.completed_subjects:
            self.manifest.completed_subjects.remove(subject_id)

    def _read_manifest(self) -> CheckpointManifest:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return CheckpointManifest(**payload)

    def _write_manifest(self) -> None:
        self._ensure_dirs()
        self.manifest.config_hash = self.config_hash
        self.manifest.individual_config_hash = self.config_hash
        self.manifest.run_mode = self.run_mode
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.manifest), handle, indent=2, sort_keys=True)
