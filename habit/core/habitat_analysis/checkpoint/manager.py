"""
Checkpoint storage and manifest management for resumable habitat training.
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


@dataclass
class CheckpointManifest:
    """On-disk metadata for a resumable habitat training run."""

    version: int = CHECKPOINT_VERSION
    config_hash: str = ""
    clustering_mode: str = ""
    completed_subjects: List[str] = field(default_factory=list)
    failed_subjects: List[str] = field(default_factory=list)
    stage: str = "individual"


def compute_config_hash(config: HabitatAnalysisConfig) -> str:
    """
    Build a stable hash of training-relevant configuration fields.

    Args:
        config: Validated habitat analysis configuration.

    Returns:
        Hex digest prefix used to detect config changes between resume runs.
    """
    payload: Dict[str, Any] = {
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
    canonical: str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class HabitatTrainCheckpoint:
    """
    Manage per-subject individual-level checkpoints for habitat training.

    Args:
        checkpoint_dir: Root directory for manifest and subject cache files.
        config: Active habitat analysis configuration.
        logger: Optional logger for resume warnings and progress messages.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        config: HabitatAnalysisConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.subjects_dir = self.checkpoint_dir / SUBJECTS_SUBDIR
        self.manifest_path = self.checkpoint_dir / MANIFEST_FILENAME
        self.config = config
        self.logger = logger
        self.config_hash = compute_config_hash(config)
        self.clustering_mode = (
            config.HabitatSegmentation.clustering_mode
            if config.HabitatSegmentation is not None
            else ""
        )
        self.manifest = CheckpointManifest(
            config_hash=self.config_hash,
            clustering_mode=self.clustering_mode,
        )

    @staticmethod
    def resolve_checkpoint_dir(out_dir: str, checkpoint_dir: Optional[str]) -> Path:
        """
        Resolve the checkpoint directory from config overrides.

        Args:
            out_dir: Pipeline output directory.
            checkpoint_dir: Optional explicit checkpoint path from config.

        Returns:
            Absolute checkpoint directory path.
        """
        if checkpoint_dir:
            return Path(checkpoint_dir)
        return Path(out_dir) / ".habitat_checkpoint"

    def initialize_for_run(self, *, resume: bool) -> None:
        """
        Prepare checkpoint storage for a training run.

        When ``resume`` is False, any existing checkpoint directory is removed.
        When ``resume`` is True, an existing manifest is loaded; if the stored
        config hash differs from the current config, a warning is logged and the
        checkpoint is cleared before restarting.

        Args:
            resume: Whether to reuse completed subjects from disk.
        """
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
        if loaded.config_hash != self.config_hash:
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

        self.manifest = loaded
        self._ensure_dirs()
        if self.logger:
            self.logger.info(
                "Loaded checkpoint: %s completed, %s failed subject(s).",
                len(self.manifest.completed_subjects),
                len(self.manifest.failed_subjects),
            )

    def pending_subjects(
        self,
        all_subjects: Iterable[str],
        *,
        resume: bool,
        force_rerun_subjects: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """
        Return subject IDs that still need individual-level processing.

        Args:
            all_subjects: Full subject list for the current run.
            resume: Whether completed/failed subjects should be skipped.
            force_rerun_subjects: Subject IDs to remove from cache and reprocess.

        Returns:
            Ordered list of subject IDs to dispatch to parallel workers.
        """
        subject_list = list(all_subjects)
        if not resume:
            return subject_list

        force_rerun: Set[str] = set(force_rerun_subjects or [])
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
            if subject_id not in completed
            and subject_id not in failed
            and subject_id not in force_rerun
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

    def mark_training_complete(self) -> None:
        """Mark the checkpoint as fully completed before optional cleanup."""
        self.manifest.stage = "done"
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
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self.manifest), handle, indent=2, sort_keys=True)
