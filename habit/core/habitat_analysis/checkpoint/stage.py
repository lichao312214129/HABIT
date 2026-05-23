"""
Stage wrapper for checkpoint-aware individual-level parallel processing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from habit.utils.log_utils import get_module_logger
from habit.utils.parallel_utils import ProcessingResult, parallel_map

from ..pipelines.habitat_subject_data import HabitatSubjectData
from .manager import HabitatTrainCheckpoint
from .step import CheckpointSaveStep

if TYPE_CHECKING:
    from ..pipelines.base_pipeline import HabitatPipeline


class IndividualCheckpointStage:
    """
    Run individual-level steps with optional resume and manifest tracking.

    Args:
        pipeline: Habitat pipeline whose individual steps will be executed.
        logger: Logger for resume and failure messages.
    """

    def __init__(
        self,
        pipeline: "HabitatPipeline",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.pipeline = pipeline
        self.logger = logger or get_module_logger(__name__)
        self.config = pipeline.config
        self.checkpoint: Optional[HabitatTrainCheckpoint] = None

    def run(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute individual-level processing with checkpoint orchestration.

        Args:
            X: Mapping of subject ID to initial per-subject payload.

        Returns:
            Mapping of subject ID to completed individual-level payload.
        """
        self.checkpoint = self._create_checkpoint()
        self.pipeline._train_checkpoint = self.checkpoint
        self._inject_checkpoint_into_steps(self.checkpoint)

        cached_results, pending_items = self._resolve_pending(X)
        n_subjects = len(X)
        n_pending = len(pending_items)
        n_processes = getattr(self.config, "processes", 4) if self.config else 4

        self.logger.info(
            "Processing %s/%s pending subjects with %s parallel workers "
            "(individual-level pipeline)...",
            n_pending,
            n_subjects,
            n_processes,
        )

        parallel_kwargs = self._parallel_kwargs()
        on_item_done = self._build_on_item_done_callback()

        successful_results, failed_subjects = parallel_map(
            func=self.pipeline._process_single_subject,
            items=pending_items,
            n_processes=n_processes,
            desc="Processing subjects (individual-level pipeline)",
            logger=self.logger,
            show_progress=True,
            on_item_done=on_item_done,
            **parallel_kwargs,
        )

        results: Dict[str, Any] = dict(cached_results)
        for proc_result in successful_results:
            results[proc_result.item_id] = proc_result.result

        self.pipeline.mask_info_cache = {
            subject_id: data.mask_info
            for subject_id, data in results.items()
            if isinstance(data, HabitatSubjectData) and data.mask_info is not None
        }

        if failed_subjects:
            self.logger.error(
                "Failed to process %s subject(s) in this run: %s",
                len(failed_subjects),
                ", ".join(str(subject_id) for subject_id in failed_subjects),
            )
            self._handle_failures(failed_subjects, results)

        self.logger.info(
            "Individual-level processing completed: %s/%s subjects successful",
            len(results),
            n_subjects,
        )
        return results

    def _create_checkpoint(self) -> Optional[HabitatTrainCheckpoint]:
        if self.config is None or getattr(self.config, "run_mode", "train") != "train":
            return None

        checkpoint_dir = HabitatTrainCheckpoint.resolve_checkpoint_dir(
            self.config.out_dir,
            getattr(self.config, "checkpoint_dir", None),
        )
        checkpoint = HabitatTrainCheckpoint(checkpoint_dir, self.config, self.logger)
        resume = bool(getattr(self.config, "resume", False))
        checkpoint.initialize_for_run(resume=resume)
        return checkpoint

    def _inject_checkpoint_into_steps(
        self,
        checkpoint: Optional[HabitatTrainCheckpoint],
    ) -> None:
        for _, step in self.pipeline.individual_steps:
            if isinstance(step, CheckpointSaveStep):
                step.set_checkpoint(checkpoint)

    def _resolve_pending(
        self,
        X: Dict[str, Any],
    ) -> Tuple[Dict[str, HabitatSubjectData], List[Tuple[str, Any]]]:
        cached_results: Dict[str, HabitatSubjectData] = {}
        pending_items: List[Tuple[str, Any]] = list(X.items())

        if self.checkpoint is None:
            return cached_results, pending_items

        resume = bool(getattr(self.config, "resume", False))
        force_rerun = getattr(self.config, "force_rerun_subjects", None) or []
        pending_ids = self.checkpoint.pending_subjects(
            X.keys(),
            resume=resume,
            force_rerun_subjects=force_rerun,
        )
        cached_results = self.checkpoint.load_completed_results()
        pending_items = [(subject_id, X[subject_id]) for subject_id in pending_ids]

        if cached_results:
            self.logger.info(
                "Resume: loaded %s completed subject(s) from checkpoint.",
                len(cached_results),
            )
        if resume and self.checkpoint.manifest.failed_subjects:
            self.logger.info(
                "Resume: skipping %s previously failed subject(s): %s",
                len(self.checkpoint.manifest.failed_subjects),
                ", ".join(self.checkpoint.manifest.failed_subjects),
            )

        return cached_results, pending_items

    def _parallel_kwargs(self) -> Dict[str, Any]:
        graceful_shutdown_sec = 15.0
        oom_backoff = True
        oom_reduce_workers_by = 1
        if self.config is not None:
            graceful_shutdown_sec = getattr(
                self.config,
                "individual_subject_graceful_shutdown_sec",
                15.0,
            )
            oom_backoff = getattr(self.config, "oom_backoff", True)
            oom_reduce_workers_by = getattr(self.config, "oom_reduce_workers_by", 1)

        return {
            "per_item_timeout_sec": (
                getattr(self.config, "individual_subject_timeout_sec", None)
                if self.config is not None
                else None
            ),
            "graceful_shutdown_sec": graceful_shutdown_sec,
            "oom_backoff": oom_backoff,
            "oom_reduce_workers_by": oom_reduce_workers_by,
        }

    def _build_on_item_done_callback(
        self,
    ) -> Optional[Callable[[ProcessingResult], None]]:
        if self.checkpoint is None:
            return None

        checkpoint = self.checkpoint

        def _on_item_done(proc_result: ProcessingResult) -> None:
            subject_id = str(proc_result.item_id)
            if (
                proc_result.success
                and isinstance(proc_result.result, HabitatSubjectData)
            ):
                checkpoint.record_success_manifest(subject_id)
                return
            checkpoint.record_failure(subject_id)

        return _on_item_done

    def _handle_failures(
        self,
        failed_subjects: List[Any],
        results: Dict[str, Any],
    ) -> None:
        on_failure = (
            getattr(self.config, "on_subject_failure", "continue")
            if self.config is not None
            else "continue"
        )
        if on_failure == "fail_fast":
            raise RuntimeError(
                f"Individual-level processing failed for {len(failed_subjects)} "
                f"subject(s) (on_subject_failure=fail_fast). "
                f"Failed: {', '.join(str(subject_id) for subject_id in failed_subjects)}"
            )
        if not results:
            raise ValueError(
                "All subjects failed during individual-level processing. "
                "Check the errors above before running group-level pipeline steps."
            )
