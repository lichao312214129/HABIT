# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Stage wrapper for checkpoint-aware individual-level parallel processing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from habit.utils.log_utils import get_module_logger, LoggerManager
from habit.utils.parallel_gpu_utils import apply_gpu_pool_process_cap
from habit.utils.parallel_utils import (
    ProcessingResult,
    _should_use_spawn_workers,
    parallel_map,
)
from habit.utils.persistent_worker_runner import PersistentWorkerPoolSession

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
        self._persistent_pool: Optional[PersistentWorkerPoolSession] = None

    def run(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute individual-level processing with checkpoint orchestration.

        When train-mode checkpointing is enabled and
        ``individual_subject_auto_retry_rounds`` is greater than zero, subjects
        that remain in ``manifest.failed_subjects`` after the initial parallel
        pass are automatically re-dispatched up to that many additional rounds.

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
        results: Dict[str, Any] = dict(cached_results)

        try:
            if pending_items:
                self._ensure_persistent_pool(n_subjects)
                self._log_parallel_pass_start(len(pending_items), n_subjects)
                self._run_parallel_pass(pending_items, results)

            auto_retry_rounds = self._auto_retry_rounds()
            if self.checkpoint is not None and auto_retry_rounds > 0:
                self._run_auto_retries(X, results, max_rounds=auto_retry_rounds)
        finally:
            if self._persistent_pool is not None:
                self._persistent_pool.shutdown(logger=self.logger)
                self._persistent_pool = None

        remaining_failed = self._remaining_failed_subjects(X)
        if remaining_failed:
            self.logger.error(
                "Failed to process %s subject(s) after individual-level processing: %s",
                len(remaining_failed),
                ", ".join(str(subject_id) for subject_id in remaining_failed),
            )
            self._handle_failures(remaining_failed, results)

        self.logger.info(
            "Individual-level processing completed: %s/%s subjects successful",
            len(results),
            n_subjects,
        )
        return results

    def _run_auto_retries(
        self,
        X: Dict[str, Any],
        results: Dict[str, Any],
        *,
        max_rounds: int,
    ) -> None:
        """
        Re-dispatch checkpoint failed subjects within the same train run.

        Args:
            X: Full subject mapping for the current run.
            results: Mutable aggregate of successful per-subject outputs.
            max_rounds: Maximum number of additional retry rounds after the first pass.
        """
        assert self.checkpoint is not None

        for round_index in range(1, max_rounds + 1):
            failed_ids = self._remaining_failed_subjects(X)
            if not failed_ids:
                return

            self.logger.info(
                "Auto-retry round %s/%s for %s failed subject(s): %s",
                round_index,
                max_rounds,
                len(failed_ids),
                ", ".join(str(subject_id) for subject_id in failed_ids),
            )

            self.checkpoint.requeue_subjects(failed_ids)
            retry_items = [
                (subject_id, X[subject_id])
                for subject_id in failed_ids
                if subject_id in X
            ]
            if not retry_items:
                return

            self._ensure_persistent_pool(len(X))
            self._log_parallel_pass_start(len(retry_items), len(X))
            self._run_parallel_pass(retry_items, results)

            if not self._remaining_failed_subjects(X):
                return

        still_failed = self._remaining_failed_subjects(X)
        if still_failed:
            self.logger.error(
                "Auto-retry exhausted (%s round(s)); %s subject(s) still failed: %s",
                max_rounds,
                len(still_failed),
                ", ".join(str(subject_id) for subject_id in still_failed),
            )

    def _run_parallel_pass(
        self,
        pending_items: List[Tuple[str, Any]],
        results: Dict[str, Any],
    ) -> List[Any]:
        """
        Execute one bounded parallel batch and merge successful outputs.

        Args:
            pending_items: Subject items to dispatch to workers.
            results: Mutable aggregate updated with successful outputs.

        Returns:
            Subject IDs that failed during this pass.
        """
        if not pending_items:
            return []

        n_processes = self._resolve_parallel_processes(len(pending_items))
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
            parallel_mode=self._parallel_mode(),
            persistent_pool_session=self._persistent_pool,
            max_consecutive_failures=self._persistent_max_consecutive_failures(),
            recycle_after_tasks=self._persistent_recycle_after_tasks(),
            **parallel_kwargs,
        )

        for proc_result in successful_results:
            results[proc_result.item_id] = proc_result.result

        self._refresh_mask_info_cache(results)

        if failed_subjects:
            self.logger.error(
                "Failed to process %s subject(s) in this pass: %s",
                len(failed_subjects),
                ", ".join(str(subject_id) for subject_id in failed_subjects),
            )

        return failed_subjects

    def _remaining_failed_subjects(self, X: Dict[str, Any]) -> List[str]:
        """
        Return failed subject IDs for the current run, in manifest order when available.

        Args:
            X: Full subject mapping for the current run.

        Returns:
            Ordered list of subject IDs still marked failed for this run.
        """
        if self.checkpoint is None:
            return []

        subject_set = set(X.keys())
        return [
            subject_id
            for subject_id in self.checkpoint.manifest.failed_subjects
            if subject_id in subject_set
        ]

    def _auto_retry_rounds(self) -> int:
        """Read configured in-run auto-retry rounds (0 disables)."""
        if self.config is None:
            return 0
        return int(
            getattr(self.config, "individual_subject_auto_retry_rounds", 0) or 0
        )

    def _parallel_mode(self) -> str:
        """Read configured individual-level parallel execution mode."""
        if self.config is None:
            return "persistent"
        return str(
            getattr(self.config, "individual_subject_parallel_mode", "persistent")
            or "persistent"
        )

    def _persistent_max_consecutive_failures(self) -> int:
        if self.config is None:
            return 1
        return int(
            getattr(self.config, "persistent_worker_max_consecutive_failures", 1) or 1
        )

    def _persistent_recycle_after_tasks(self) -> int:
        if self.config is None:
            return 0
        return int(
            getattr(self.config, "persistent_worker_recycle_after_tasks", 0) or 0
        )

    def _resolve_max_pool_workers(self, n_subjects: int) -> int:
        """
        Resolve worker count for a persistent pool (not limited by one pass batch).

        Args:
            n_subjects: Total subjects in the current run.

        Returns:
            int: Configured worker count capped by GPU pool and subject count.
        """
        configured = getattr(self.config, "processes", 4) if self.config else 4
        requested = apply_gpu_pool_process_cap(
            configured,
            self.config,
            log=self.logger,
        )

        return min(requested, max(1, n_subjects))

    def _ensure_persistent_pool(self, n_subjects: int) -> None:
        """
        Lazily create a persistent worker pool for the current ``run()`` call.

        Args:
            n_subjects: Total subjects in the current run (pool sizing).
        """
        if self._parallel_mode() != "persistent":
            return
        if self._persistent_pool is not None:
            return

        n_workers = self._resolve_max_pool_workers(n_subjects)
        parallel_kwargs = self._parallel_kwargs()
        timeout_sec = parallel_kwargs.get("per_item_timeout_sec")
        if not _should_use_spawn_workers(n_workers, max(1, n_subjects), timeout_sec):
            return

        manager = LoggerManager()
        log_file_path = manager.get_log_file()
        log_queue = manager.get_log_queue()
        log_level = logging.INFO
        if manager._root_logger:
            log_level = manager._root_logger.getEffectiveLevel()

        pool = PersistentWorkerPoolSession(
            max_workers=n_workers,
            func=self.pipeline._process_single_subject,
            log_file_path=log_file_path,
            log_queue=log_queue,
            log_level=log_level,
            per_item_timeout_sec=timeout_sec,
            graceful_shutdown_sec=parallel_kwargs["graceful_shutdown_sec"],
            max_consecutive_failures=self._persistent_max_consecutive_failures(),
            recycle_after_tasks=self._persistent_recycle_after_tasks(),
            spawn_startup_timeout_sec=parallel_kwargs.get("spawn_startup_timeout_sec"),
            oom_backoff=parallel_kwargs["oom_backoff"],
            oom_reduce_workers_by=parallel_kwargs["oom_reduce_workers_by"],
        )
        pool.start(logger=self.logger)
        self._persistent_pool = pool

    def _log_parallel_pass_start(self, n_pending: int, n_subjects: int) -> None:
        """Log worker count for an individual-level parallel pass."""
        n_processes = self._resolve_parallel_processes(n_pending)
        self.logger.info(
            "Processing %s/%s pending subjects with %s parallel workers "
            "(individual-level pipeline)...",
            n_pending,
            n_subjects,
            n_processes,
        )

    def _resolve_parallel_processes(self, n_pending: int) -> int:
        """
        Resolve effective Stage-1 worker count with optional GPU pool capping.

        Args:
            n_pending: Number of subjects scheduled in this parallel pass.

        Returns:
            int: Worker count in ``[1, processes]``, optionally capped by Torch GPU pool.
        """
        configured = getattr(self.config, "processes", 4) if self.config else 4
        requested = apply_gpu_pool_process_cap(
            configured,
            self.config,
            log=self.logger,
        )

        if n_pending > 0:
            return min(requested, n_pending)
        return requested

    def _refresh_mask_info_cache(self, results: Dict[str, Any]) -> None:
        """Rebuild pipeline mask metadata from successful individual outputs."""
        self.pipeline.mask_info_cache = {
            subject_id: data.mask_info
            for subject_id, data in results.items()
            if isinstance(data, HabitatSubjectData) and data.mask_info is not None
        }

    def _create_checkpoint(self) -> Optional[HabitatTrainCheckpoint]:
        if self.config is None:
            return None

        run_mode = getattr(self.config, "run_mode", "train")
        if run_mode not in {"train", "predict"}:
            return None

        if run_mode == "predict" and not getattr(self.config, "pipeline_path", None):
            self.logger.warning(
                "Predict mode without pipeline_path; checkpoint/resume disabled."
            )
            return None

        checkpoint_dir = HabitatTrainCheckpoint.resolve_checkpoint_dir(
            self.config.out_dir,
            getattr(self.config, "checkpoint_dir", None),
            run_mode=run_mode,
        )
        checkpoint = HabitatTrainCheckpoint(
            checkpoint_dir,
            self.config,
            self.logger,
            run_mode=run_mode,
        )
        resume = bool(getattr(self.config, "resume", True))
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

        resume = bool(getattr(self.config, "resume", True))
        force_rerun = getattr(self.config, "force_rerun_subjects", None) or []
        retry_failed = bool(getattr(self.config, "retry_failed_subjects", False))
        failed_snapshot = list(self.checkpoint.manifest.failed_subjects)
        pending_ids = self.checkpoint.pending_subjects(
            X.keys(),
            resume=resume,
            force_rerun_subjects=force_rerun,
            retry_failed_subjects=retry_failed,
        )
        cached_results = self.checkpoint.load_completed_results()
        pending_items = [(subject_id, X[subject_id]) for subject_id in pending_ids]

        if cached_results:
            self.logger.info(
                "Resume: loaded %s completed subject(s) from checkpoint.",
                len(cached_results),
            )
        if retry_failed and failed_snapshot:
            retried = [subject_id for subject_id in failed_snapshot if subject_id in pending_ids]
            if retried:
                self.logger.info(
                    "Resume: retrying %s previously failed subject(s): %s",
                    len(retried),
                    ", ".join(str(subject_id) for subject_id in retried),
                )
        elif resume and self.checkpoint.manifest.failed_subjects:
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
            "spawn_startup_timeout_sec": (
                getattr(self.config, "individual_subject_spawn_timeout_sec", None)
                if self.config is not None
                else None
            ),
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
