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
Persistent spawn worker pool with one long-lived child process per worker slot.

Each slot processes at most one item at a time. The parent can kill and restart a
slot process on per-item timeout or worker death, similar to isolated mode but
without respawning the entire pool for every subject.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable, List, Optional, Set, Tuple

from habit.utils.isolated_runner import (
    DEFAULT_GRACEFUL_SHUTDOWN_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    ProcessingResult,
    is_fatal_memory_error,
    requires_worker_restart,
)
from habit.utils.persistent_worker_entry import persistent_worker_main
from habit.utils.persistent_worker_protocol import (
    WorkerExitReply,
    WorkerReadyReply,
    WorkerResultReply,
    WorkerRunCommand,
    WorkerStopCommand,
)
from habit.utils.progress_utils import CustomTqdm

logger = logging.getLogger(__name__)


@dataclass
class _SlotRuntime:
    """Parent-side state for one persistent worker slot."""

    worker_slot: int
    task_queue: Any
    proc: Optional[multiprocessing.Process] = None
    busy: bool = False
    current_item_id: Any = None
    started_at: Optional[float] = None
    consecutive_failures: int = 0
    successful_tasks: int = 0
    ready: bool = False
    awaiting_ready_since: Optional[float] = None
    suppress_exit_respawn: bool = False


class PersistentWorkerPoolSession:
    """
    Manage a fixed pool of long-lived spawn workers for repeated ``map_items`` calls.

    Args:
        max_workers: Number of concurrent worker slots.
        func: Picklable callable applied to each item in worker processes.
        log_file_path: Optional log file restored in each child.
        log_level: Logging level for children.
        per_item_timeout_sec: Optional wall-clock timeout per item.
        graceful_shutdown_sec: Seconds to wait after terminate before kill.
        max_consecutive_failures: Restart slot after fatal-class failures (reserved).
        recycle_after_tasks: Restart slot after this many successes (0 disables).
        spawn_startup_timeout_sec: Optional startup timeout when (re)spawning a slot.
        oom_backoff: When True, reduce active worker count after a fatal memory error.
        oom_reduce_workers_by: Decrement applied to effective workers after each OOM failure.
    """

    def __init__(
        self,
        max_workers: int,
        func: Callable[[Any], Any],
        *,
        log_file_path: Optional[Path] = None,
        log_queue: Any = None,
        log_level: int = logging.INFO,
        per_item_timeout_sec: Optional[float] = None,
        graceful_shutdown_sec: float = DEFAULT_GRACEFUL_SHUTDOWN_SEC,
        max_consecutive_failures: int = 1,
        recycle_after_tasks: int = 0,
        spawn_startup_timeout_sec: Optional[float] = 120.0,
        mp_start_method: str = "spawn",
        oom_backoff: bool = True,
        oom_reduce_workers_by: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_consecutive_failures < 1:
            raise ValueError("max_consecutive_failures must be >= 1")
        if oom_reduce_workers_by < 1:
            raise ValueError("oom_reduce_workers_by must be >= 1")

        self.max_workers = max_workers
        self.func = func
        self.log_file_path = log_file_path
        self.log_queue = log_queue
        self.log_level = log_level
        self.per_item_timeout_sec = per_item_timeout_sec
        self.graceful_shutdown_sec = graceful_shutdown_sec
        self.max_consecutive_failures = max_consecutive_failures
        self.recycle_after_tasks = recycle_after_tasks
        self.spawn_startup_timeout_sec = spawn_startup_timeout_sec
        self.oom_backoff = oom_backoff
        self.oom_reduce_workers_by = oom_reduce_workers_by
        self._ctx = multiprocessing.get_context(mp_start_method)
        self._result_queue = self._ctx.Queue()
        self._slots: List[_SlotRuntime] = []
        self._started = False
        self._shutdown = False
        self._effective_max_workers: int = max_workers

    def start(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Spawn worker processes for every slot and wait until all are READY.

        Args:
            logger: Optional logger for startup messages.

        Raises:
            RuntimeError: When startup times out or a worker exits before READY.
        """
        if self._started:
            return

        for slot_index in range(self.max_workers):
            task_queue = self._ctx.Queue()
            self._slots.append(
                _SlotRuntime(worker_slot=slot_index, task_queue=task_queue)
            )

        for slot in self._slots:
            self._spawn_slot(slot)

        ready_slots: Set[int] = set()
        deadline = None
        if self.spawn_startup_timeout_sec is not None:
            deadline = time.monotonic() + self.spawn_startup_timeout_sec

        while len(ready_slots) < len(self._slots):
            if deadline is not None and time.monotonic() > deadline:
                raise RuntimeError(
                    "Persistent worker pool startup timed out before all workers "
                    "reported READY."
                )
            self._poll_dead_workers(fail_inflight=True, logger=logger)
            wait_sec = DEFAULT_POLL_INTERVAL_SEC
            if deadline is not None:
                remaining = deadline - time.monotonic()
                wait_sec = max(0.001, min(wait_sec, remaining))
            try:
                reply = self._result_queue.get(timeout=wait_sec)
            except Empty:
                continue
            if isinstance(reply, WorkerReadyReply):
                ready_slots.add(reply.worker_slot)
                self._mark_slot_ready(reply.worker_slot)

        self._started = True
        if logger is not None:
            logger.info(
                "Using persistent worker pool: max_workers=%s%s",
                self.max_workers,
                (
                    f", per-item timeout={self.per_item_timeout_sec}s"
                    if self.per_item_timeout_sec is not None
                    else ", no per-item timeout"
                ),
            )

    def map_items(
        self,
        items_list: List[Any],
        desc: str,
        logger: Optional[logging.Logger],
        show_progress: bool,
        on_item_done: Optional[Callable[[ProcessingResult], None]] = None,
    ) -> Tuple[List[ProcessingResult], List[Any]]:
        """
        Process items using the started worker pool.

        Args:
            items_list: Work items (already materialised).
            desc: Progress bar description.
            logger: Optional logger for failures and timeouts.
            show_progress: Whether to show ``CustomTqdm``.
            on_item_done: Optional callback after each item completes.

        Returns:
            Tuple of successful ProcessingResult list and failed item ids.
        """
        if not self._started:
            self.start(logger=logger)
        if self._shutdown:
            raise RuntimeError("PersistentWorkerPoolSession has been shut down.")

        total = len(items_list)
        if total == 0:
            return [], []

        successful_results: List[ProcessingResult] = []
        failed_items: List[Any] = []
        pending_items = list(items_list)
        idle_slots = [slot for slot in self._slots if not slot.busy]
        _last_dead_worker_poll_at: float = 0.0

        progress_bar: Optional[CustomTqdm] = None
        if show_progress:
            progress_bar = CustomTqdm(total=total, desc=desc)

        def _dispatch_to_slot(slot: _SlotRuntime) -> None:
            item = pending_items.pop(0)
            if hasattr(item, "__getitem__") and len(item) > 0:
                item_id = item[0] if isinstance(item, (list, tuple)) else item
            else:
                item_id = item
            slot.busy = True
            slot.current_item_id = item_id
            slot.started_at = time.monotonic()
            slot.task_queue.put(WorkerRunCommand(item=item))

        while pending_items or any(slot.busy for slot in self._slots):
            idle_slots = [slot for slot in self._slots if slot.ready and not slot.busy]
            n_busy = sum(1 for slot in self._slots if slot.busy)
            while pending_items and idle_slots and n_busy < self._effective_max_workers:
                slot = idle_slots.pop(0)
                _dispatch_to_slot(slot)
                n_busy += 1

            self._handle_timeouts(
                failed_items=failed_items,
                successful_results=successful_results,
                logger=logger,
                on_item_done=on_item_done,
                progress_bar=progress_bar,
            )
            self._check_awaiting_ready_timeouts(logger=logger)
            now_for_poll = time.monotonic()
            if now_for_poll - _last_dead_worker_poll_at >= DEFAULT_POLL_INTERVAL_SEC:
                self._poll_dead_workers(
                    fail_inflight=True,
                    failed_items=failed_items,
                    logger=logger,
                    on_item_done=on_item_done,
                    progress_bar=progress_bar,
                )
                _last_dead_worker_poll_at = now_for_poll

            try:
                wait_sec = self._next_deadline_wait_sec()
                reply = self._result_queue.get(timeout=wait_sec)
            except Empty:
                self._poll_dead_workers(
                    fail_inflight=True,
                    failed_items=failed_items,
                    logger=logger,
                    on_item_done=on_item_done,
                    progress_bar=progress_bar,
                )
                _last_dead_worker_poll_at = time.monotonic()
                continue

            if isinstance(reply, WorkerResultReply):
                self._finish_result(
                    reply,
                    successful_results=successful_results,
                    failed_items=failed_items,
                    logger=logger,
                    on_item_done=on_item_done,
                    progress_bar=progress_bar,
                )
            elif isinstance(reply, WorkerReadyReply):
                self._mark_slot_ready(reply.worker_slot)
            elif isinstance(reply, WorkerExitReply):
                self._handle_worker_exit_reply(
                    reply,
                    failed_items=failed_items,
                    logger=logger,
                    on_item_done=on_item_done,
                    progress_bar=progress_bar,
                )

        if logger and failed_items:
            logger.warning("Failed or timed out %s item(s)", len(failed_items))

        return successful_results, failed_items

    def shutdown(self, logger: Optional[logging.Logger] = None) -> None:
        """Send STOP to all workers and join child processes."""
        if self._shutdown:
            return
        self._shutdown = True

        for slot in self._slots:
            if slot.proc is not None and slot.proc.is_alive():
                try:
                    slot.task_queue.put(WorkerStopCommand())
                except Exception:
                    pass

        deadline = time.monotonic() + self.graceful_shutdown_sec
        for slot in self._slots:
            if slot.proc is None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            slot.proc.join(timeout=remaining)
            if slot.proc.is_alive():
                self._terminate_process(slot.proc, logger=logger)

        self._started = False

    def _spawn_slot(self, slot: _SlotRuntime) -> None:
        log_path_str = str(self.log_file_path) if self.log_file_path else None
        proc = self._ctx.Process(
            target=persistent_worker_main,
            args=(
                slot.worker_slot,
                log_path_str,
                self.log_queue,
                self.log_level,
                slot.task_queue,
                self._result_queue,
                self.func,
            ),
            kwargs={"recycle_after_tasks": self.recycle_after_tasks},
        )
        proc.start()
        slot.proc = proc
        slot.busy = False
        slot.current_item_id = None
        slot.started_at = None
        slot.ready = False
        slot.awaiting_ready_since = time.monotonic()

    def _mark_slot_ready(self, worker_slot: int) -> None:
        """Record that a worker slot finished startup and can accept tasks."""
        slot = self._slot_by_index(worker_slot)
        if slot is None:
            return
        slot.ready = True
        slot.awaiting_ready_since = None

    def _restart_worker_slot(
        self,
        slot: _SlotRuntime,
        logger: Optional[logging.Logger],
    ) -> None:
        """
        Terminate the current worker (if needed) and spawn a replacement.

        Sets ``suppress_exit_respawn`` so a stale ``WorkerExitReply`` from the
        terminated process does not trigger a second respawn.
        """
        slot.suppress_exit_respawn = True
        slot.consecutive_failures = 0
        slot.successful_tasks = 0
        if slot.proc is not None and slot.proc.is_alive():
            self._terminate_process(slot.proc, logger=logger)
        self._spawn_slot(slot)

    def _respawn_slot(self, slot: _SlotRuntime, logger: Optional[logging.Logger]) -> None:
        """Backward-compatible alias for intentional slot replacement."""
        self._restart_worker_slot(slot, logger=logger)

    def _check_awaiting_ready_timeouts(
        self,
        logger: Optional[logging.Logger],
    ) -> None:
        """
        Enforce ``spawn_startup_timeout_sec`` for workers respawned mid-batch.

        Without this guard the parent can spin forever when a replacement worker
        hangs during CUDA/Torch re-initialisation after a prior slot restart.
        """
        if self.spawn_startup_timeout_sec is None:
            return

        now = time.monotonic()
        for slot in self._slots:
            if slot.ready or slot.awaiting_ready_since is None:
                continue
            elapsed = now - slot.awaiting_ready_since
            if elapsed <= self.spawn_startup_timeout_sec:
                continue
            if logger is not None:
                logger.error(
                    "Persistent worker slot %s failed to report READY within %ss "
                    "after restart; respawning again.",
                    slot.worker_slot,
                    self.spawn_startup_timeout_sec,
                )
            self._restart_worker_slot(slot, logger=logger)

    def _handle_worker_exit_reply(
        self,
        reply: WorkerExitReply,
        *,
        failed_items: List[Any],
        logger: Optional[logging.Logger],
        on_item_done: Optional[Callable[[ProcessingResult], None]],
        progress_bar: Optional[CustomTqdm],
    ) -> None:
        """
        Handle worker shutdown notifications without duplicate respawns.

        Intentional restarts set ``suppress_exit_respawn`` before terminate so
        the exit message from the old process is ignored once the replacement
        worker is already being started.
        """
        slot = self._slot_by_index(reply.worker_slot)
        if slot is None:
            return
        if slot.suppress_exit_respawn:
            slot.suppress_exit_respawn = False
            return
        if slot.busy:
            self._fail_slot(
                slot,
                error=RuntimeError(
                    f"Persistent worker {slot.worker_slot} exited unexpectedly"
                ),
                failed_items=failed_items,
                logger=logger,
                on_item_done=on_item_done,
                progress_bar=progress_bar,
            )
        if not self._shutdown and not slot.ready:
            self._restart_worker_slot(slot, logger=logger)

    def _wait_slot_ready(self, slot: _SlotRuntime, logger: Optional[logging.Logger]) -> None:
        """Block until one slot reports READY (startup / synchronous tests only)."""
        deadline = None
        if self.spawn_startup_timeout_sec is not None:
            deadline = time.monotonic() + self.spawn_startup_timeout_sec
        while not slot.ready:
            if deadline is not None and time.monotonic() > deadline:
                raise RuntimeError(
                    f"Persistent worker slot {slot.worker_slot} failed to restart."
                )
            try:
                reply = self._result_queue.get(timeout=DEFAULT_POLL_INTERVAL_SEC)
            except Empty:
                self._poll_dead_workers(fail_inflight=False, logger=logger)
                continue
            if isinstance(reply, WorkerReadyReply) and reply.worker_slot == slot.worker_slot:
                self._mark_slot_ready(reply.worker_slot)
                return
            if isinstance(reply, WorkerReadyReply):
                self._mark_slot_ready(reply.worker_slot)

    def _slot_by_index(self, worker_slot: int) -> Optional[_SlotRuntime]:
        for slot in self._slots:
            if slot.worker_slot == worker_slot:
                return slot
        return None

    def _next_deadline_wait_sec(self) -> float:
        """
        Compute an adaptive ``queue.get`` timeout based on pending deadlines.

        When a per-item timeout is configured, we must wake up before any busy
        slot exceeds its deadline. This method returns the time until the
        nearest deadline (capped by :data:`DEFAULT_POLL_INTERVAL_SEC`) so the
        main loop can react promptly even when timeouts are short.
        """
        max_wait = DEFAULT_POLL_INTERVAL_SEC
        if self.per_item_timeout_sec is None:
            return max_wait

        now = time.monotonic()
        for slot in self._slots:
            if not slot.busy or slot.started_at is None:
                continue
            remaining = (slot.started_at + self.per_item_timeout_sec) - now
            if remaining <= 0:
                return 0.001
            if remaining < max_wait:
                max_wait = remaining

        if self.spawn_startup_timeout_sec is not None:
            for slot in self._slots:
                if slot.ready or slot.awaiting_ready_since is None:
                    continue
                remaining = (slot.awaiting_ready_since + self.spawn_startup_timeout_sec) - now
                if remaining <= 0:
                    return 0.001
                if remaining < max_wait:
                    max_wait = remaining

        return max(0.001, max_wait)

    def _finish_result(
        self,
        reply: WorkerResultReply,
        *,
        successful_results: List[ProcessingResult],
        failed_items: List[Any],
        logger: Optional[logging.Logger],
        on_item_done: Optional[Callable[[ProcessingResult], None]],
        progress_bar: Optional[CustomTqdm],
    ) -> None:
        slot = self._slot_by_index(reply.worker_slot)
        if slot is None:
            return

        proc_result = reply.proc_result
        slot.busy = False
        slot.current_item_id = None
        slot.started_at = None

        if proc_result is None:
            self._fail_slot(
                slot,
                error=RuntimeError(
                    f"Persistent worker {slot.worker_slot} returned no result"
                ),
                failed_items=failed_items,
                logger=logger,
                on_item_done=on_item_done,
                progress_bar=progress_bar,
            )
            if requires_worker_restart(
                proc_result,
                worker_alive=slot.proc is not None and slot.proc.is_alive(),
            ):
                self._restart_worker_slot(slot, logger=logger)
            return

        if proc_result.success:
            slot.consecutive_failures = 0
            slot.successful_tasks += 1
            successful_results.append(proc_result)
            if on_item_done is not None:
                on_item_done(proc_result)
        else:
            failed_items.append(proc_result.item_id)
            if logger is not None:
                if is_fatal_memory_error(proc_result.error):
                    logger.error(
                        "Fatal memory error for item %s: %s",
                        proc_result.item_id,
                        proc_result.error,
                    )
                else:
                    logger.error(
                        "Error processing %s: %s",
                        proc_result.item_id,
                        proc_result.error,
                    )
            if on_item_done is not None:
                on_item_done(proc_result)
            if is_fatal_memory_error(proc_result.error):
                slot.consecutive_failures += 1
                self._apply_oom_backoff(proc_result.item_id, logger=logger)

        if progress_bar is not None:
            progress_bar.update(1)

        worker_alive = slot.proc is not None and slot.proc.is_alive()
        if requires_worker_restart(proc_result, worker_alive=worker_alive):
            self._restart_worker_slot(slot, logger=logger)

    def _apply_oom_backoff(
        self,
        item_id: Any,
        *,
        logger: Optional[logging.Logger],
    ) -> None:
        """
        Reduce the effective parallel worker count after a fatal memory error.

        Mirrors the OOM backoff in :class:`IsolatedTaskRunner`: shrink the
        concurrency ceiling so remaining subjects have more memory headroom.
        """
        if not self.oom_backoff:
            return
        new_max = max(1, self._effective_max_workers - self.oom_reduce_workers_by)
        if new_max < self._effective_max_workers and logger is not None:
            logger.warning(
                "Fatal memory error for item %s in persistent pool; reducing "
                "effective parallel workers %s -> %s",
                item_id,
                self._effective_max_workers,
                new_max,
            )
        self._effective_max_workers = new_max

    def _fail_slot(
        self,
        slot: _SlotRuntime,
        *,
        error: Exception,
        failed_items: List[Any],
        logger: Optional[logging.Logger],
        on_item_done: Optional[Callable[[ProcessingResult], None]],
        progress_bar: Optional[CustomTqdm],
    ) -> None:
        item_id = slot.current_item_id
        if item_id is None:
            return
        failed_items.append(item_id)
        if logger is not None:
            logger.error("Error processing %s: %s", item_id, error)
        proc_result = ProcessingResult(item_id=item_id, error=error)
        if on_item_done is not None:
            on_item_done(proc_result)
        slot.busy = False
        slot.current_item_id = None
        slot.started_at = None
        slot.consecutive_failures += 1
        if progress_bar is not None:
            progress_bar.update(1)

    def _handle_timeouts(
        self,
        *,
        failed_items: List[Any],
        successful_results: List[ProcessingResult],
        logger: Optional[logging.Logger],
        on_item_done: Optional[Callable[[ProcessingResult], None]],
        progress_bar: Optional[CustomTqdm],
    ) -> None:
        if self.per_item_timeout_sec is None:
            return
        now = time.monotonic()
        for slot in self._slots:
            if not slot.busy or slot.started_at is None:
                continue
            if now - slot.started_at <= self.per_item_timeout_sec:
                continue
            if logger is not None:
                logger.error(
                    "Timeout (>%ss) for item %s on persistent worker %s; "
                    "restarting worker slot.",
                    self.per_item_timeout_sec,
                    slot.current_item_id,
                    slot.worker_slot,
                )
            if slot.proc is not None and slot.proc.is_alive():
                self._terminate_process(slot.proc, logger=logger)
            self._fail_slot(
                slot,
                error=TimeoutError(
                    f"Item {slot.current_item_id} exceeded "
                    f"{self.per_item_timeout_sec}s on persistent worker "
                    f"{slot.worker_slot}"
                ),
                failed_items=failed_items,
                logger=logger,
                on_item_done=on_item_done,
                progress_bar=progress_bar,
            )
            if not self._shutdown:
                self._restart_worker_slot(slot, logger=logger)

    def _poll_dead_workers(
        self,
        *,
        fail_inflight: bool,
        failed_items: Optional[List[Any]] = None,
        logger: Optional[logging.Logger] = None,
        on_item_done: Optional[Callable[[ProcessingResult], None]] = None,
        progress_bar: Optional[CustomTqdm] = None,
    ) -> None:
        if failed_items is None:
            failed_items = []
        for slot in self._slots:
            if slot.proc is None or slot.proc.is_alive():
                continue
            if fail_inflight and slot.busy:
                exit_code = slot.proc.exitcode
                self._fail_slot(
                    slot,
                    error=RuntimeError(
                        f"Persistent worker {slot.worker_slot} died "
                        f"(exit code {exit_code})"
                    ),
                    failed_items=failed_items,
                    logger=logger,
                    on_item_done=on_item_done,
                    progress_bar=progress_bar,
                )
            if not self._shutdown:
                self._restart_worker_slot(slot, logger=logger)

    def _terminate_process(
        self,
        proc: multiprocessing.Process,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not proc.is_alive():
            return
        try:
            proc.terminate()
        except OSError as exc:
            if logger is not None:
                logger.warning(
                    "terminate() failed for persistent worker pid %s: %s",
                    proc.pid,
                    exc,
                )
            return
        proc.join(timeout=self.graceful_shutdown_sec)
        if proc.is_alive():
            try:
                proc.kill()
            except OSError as exc:
                if logger is not None:
                    logger.warning(
                        "kill() failed for persistent worker pid %s: %s",
                        proc.pid,
                        exc,
                    )
                return
            proc.join(timeout=10.0)
