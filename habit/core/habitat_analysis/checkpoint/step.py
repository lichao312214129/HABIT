"""
Individual-level pipeline step that persists per-subject checkpoint pickles.
"""

from __future__ import annotations

from typing import Optional

from habit.utils.log_utils import get_module_logger

from ..config_schemas import HabitatAnalysisConfig
from ..pipelines.base_pipeline import IndividualLevelStep
from ..pipelines.habitat_subject_data import HabitatSubjectData
from .manager import HabitatTrainCheckpoint


class CheckpointSaveStep(IndividualLevelStep):
    """
    Save one subject's individual-level result to the training checkpoint directory.

    Stateless aside from an optional injected :class:`HabitatTrainCheckpoint`
    instance set by :class:`IndividualCheckpointStage` before parallel dispatch.

    In train mode the payload is written to disk for resume, then returned to the
    parent via IPC. This step runs after merge, so the returned object is small
    (typically ``supervoxel_df`` only).
    """

    def __init__(self, config: HabitatAnalysisConfig) -> None:
        super().__init__()
        self.config = config
        self.logger = get_module_logger(__name__)
        self._checkpoint: Optional[HabitatTrainCheckpoint] = None

    def set_checkpoint(self, checkpoint: Optional[HabitatTrainCheckpoint]) -> None:
        """
        Inject the active checkpoint manager for this run.

        Args:
            checkpoint: Checkpoint manager created in the parent process, or
                ``None`` when checkpointing is disabled.
        """
        self._checkpoint = checkpoint

    def transform_one(
        self,
        subject_id: str,
        subject_data: HabitatSubjectData,
    ) -> HabitatSubjectData:
        """
        Write ``subject_data`` to ``subjects/{subject_id}.pkl`` when checkpointing is active.

        Args:
            subject_id: Subject identifier.
            subject_data: Payload produced by upstream individual-level steps.

        Returns:
            Unmodified ``subject_data`` (small post-merge payload for group stage).
        """
        if self._checkpoint is None:
            return subject_data

        self._checkpoint.save_subject_pkl(subject_id, subject_data)
        if self.config.verbose:
            self.logger.info(
                "Checkpoint saved individual-level result for subject '%s'.",
                subject_id,
            )
        return subject_data
