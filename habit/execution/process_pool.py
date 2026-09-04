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
"""Process-pool execution backend: v0.1's parallel engineering, relocated.

Every scheduling concern that v0.1 kept in the configuration schema lives
here now, so algorithms contain no scheduling code at all:

* per-subject wall-clock timeout with graceful shutdown
  (``terminate`` -> ``graceful_shutdown_sec`` -> ``kill``),
* spawn-startup timeout for per-subject child processes,
* subject failure isolation (``continue``) or abort (``fail_fast``),
* OOM backoff that reduces the effective worker count after a fatal
  memory error,
* two worker lifecycles -- ``persistent`` (one long-lived worker per slot)
  and ``isolated`` (one child process per subject),
* optional GPU-pool capping of the worker count,
* checkpoint-aware resume with the v0.1 failure-skip rule
  (``retry_failed_subjects`` / ``force_rerun_subjects`` /
  ``clear_checkpoint_on_success``),
* automatic re-dispatch rounds for failed subjects within one run
  (``auto_retry_rounds``).

The constructor surface mirrors :class:`~habit.spec.policy.RunPolicy`
field-by-field, so the YAML form and the Python form stay one-to-one and
:meth:`from_policy` is a pure transcription.

Children run under the ``spawn`` context: fork-safety is not negotiable in
a stack that may hold native imaging libraries. Operators and items must
therefore be picklable -- which is exactly the boundary Phase 1 designed
``Subject``/``ImageRef`` for: light references cross the process boundary,
arrays never do until the child loads them.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import queue as queue_module
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from habit.exceptions import HABITAPIError
from habit.contracts.ops import SubjectOperator, SubjectResult
from habit.execution.backends import _cache_key_of, _subject_id_of
from habit.execution.checkpoint import CheckpointStore
from habit.utils.parallel_gpu_utils import pin_worker_visible_cuda_device

if TYPE_CHECKING:
    # Typing-only reference: ``habit.spec`` sits outside the layers the
    # execution package may import at module load time, so the runtime
    # import happens lazily inside ``__init__`` / ``from_policy``.
    from habit.spec.policy import RunPolicy

__all__ = ["ProcessPoolBackend", "SubjectTimeoutError"]

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

#: Result-queue message kinds exchanged with child processes.
_STATUS_OK = "ok"
_STATUS_ERROR = "error"
_STATUS_OOM = "oom"
_MSG_STARTED = "started"
_MSG_BOUND = "bound"

#: Parent -> worker command tags for the persistent protocol.
_CMD_BIND = "bind"
_CMD_RUN = "run"

#: Poll interval of the parent-side scheduling loops, in seconds.
_POLL_INTERVAL_SEC = 0.05

#: BLAS / OpenMP thread caps applied once in every spawned worker so that
#: ``workers × default_OMP`` cannot oversubscribe the machine (v0.1
#: ``cluster_search_parallel`` parity; the process-pool path previously
#: omitted this and could freeze laptop hosts under sklearn / numpy).
_WORKER_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _worker_thread_cap() -> str:
    """
    Resolve the per-worker BLAS/OpenMP thread budget.

    Honours ``HABIT_WORKER_THREADS`` when set to a positive integer;
    otherwise forces ``1`` so ``workers × parent_OMP`` cannot oversubscribe
    the host (``setdefault`` would inherit a large parent value).

    Returns:
        Decimal string used for every ``*_NUM_THREADS`` env var.
    """
    raw = os.environ.get("HABIT_WORKER_THREADS", "1").strip() or "1"
    try:
        n_threads = max(1, int(raw))
    except ValueError:
        n_threads = 1
    return str(n_threads)


def _configure_worker_runtime(worker_index: int) -> None:
    """
    Cap nested threading and publish the GPU slot for one child process.

    Args:
        worker_index: Zero-based slot index for this worker (also written to
            ``HABIT_GPU_SLOT_INDEX`` for TorchRadiomics device selection).
    """
    # Force (do not setdefault): parent shells often export OMP_NUM_THREADS
    # equal to the host core count; inheriting that makes every child
    # oversubscribe and can make parallel runs slower than serial.
    thread_cap = _worker_thread_cap()
    for key in _WORKER_THREAD_ENV:
        os.environ[key] = thread_cap
    # Hide every GPU except this slot *before* importing torch. Otherwise
    # both workers see cuda:0 and cuda:1, initialize both, and kernels
    # pile onto GPU 0 while GPU 1 stays idle. On a single-GPU host,
    # workers beyond slot 0 fall back to CPU (see parallel_gpu_utils).
    pin_worker_visible_cuda_device(int(worker_index))
    try:
        import torch

        torch.set_num_threads(int(thread_cap))
    except Exception:  # noqa: BLE001 - torch optional / broken is fine
        pass


class SubjectTimeoutError(TimeoutError):
    """A subject exceeded its wall-clock or spawn-startup budget."""


class _WorkerDiedError(RuntimeError):
    """A worker process exited without reporting an outcome."""


def _picklable(exc: BaseException) -> BaseException:
    """
    Return ``exc`` when it survives pickling, else a wrapping error.

    Exceptions raised inside a child process cross back through a queue,
    which pickles them; third-party exceptions are not always picklable, so
    the text is preserved in a :class:`HABITAPIError` fallback rather than
    losing the failure entirely.

    Args:
        exc: The exception captured in the child.

    Returns:
        The original exception, or a picklable wrapper.
    """
    try:
        pickle.dumps(exc)
    except Exception:
        # ``HABITAPIError`` is an exception subclass, but mypy sees it as
        # ``Any`` when ``habit.exceptions`` is outside the checked set
        # (``follow_imports = "skip"``); the cast keeps that boundary explicit.
        return cast(
            BaseException,
            HABITAPIError(
                f"{type(exc).__name__} raised in worker (not picklable): {exc}"
            ),
        )
    return exc


def _isolated_worker(
    op: Any,
    item: Any,
    result_queue: Any,
    worker_index: int = 0,
) -> None:
    """
    Run one subject in a dedicated child process.

    The first message is always the startup signal -- the parent's spawn
    timeout keys on it. The outcome is reported exactly once, as
    ``(status, payload)``; ``MemoryError`` gets its own status so the
    parent can apply OOM backoff.

    Args:
        op: The subject-level operator (pickled in).
        item: The subject-scoped payload (pickled in).
        result_queue: Parent-owned queue for messages.
        worker_index: Slot index for GPU/thread configuration.
    """
    _configure_worker_runtime(worker_index)
    result_queue.put(_MSG_STARTED)
    try:
        value = op(item)
    except MemoryError as exc:
        result_queue.put((_STATUS_OOM, _picklable(exc)))
    except BaseException as exc:  # noqa: BLE001 - isolation is the point
        result_queue.put((_STATUS_ERROR, _picklable(exc)))
    else:
        # ``Queue.put`` serialises lazily in a background feeder thread, so
        # the outcome is pre-serialised here, synchronously: an unpicklable
        # result must become an error message, not a silently dying feeder.
        try:
            pickle.dumps(value)
        except BaseException as exc:  # noqa: BLE001 - unpicklable result
            result_queue.put(
                (
                    _STATUS_ERROR,
                    HABITAPIError(
                        f"Result of {type(op).__name__} is not picklable: {exc}"
                    ),
                )
            )
        else:
            result_queue.put((_STATUS_OK, value))


def _persistent_worker(
    task_queue: Any,
    result_queue: Any,
    worker_index: int,
    recycle_after_tasks: int = 0,
) -> None:
    """
    Serve bind/run commands from a private queue until poisoned (``None``).

    The operator is bound via ``(_CMD_BIND, op)`` so a long-lived pool can
    be reused across recipe stages with different operators (v0.1
    ``PersistentWorkerPoolSession`` parity). Each worker owns its task
    queue: the parent dispatches exactly one run at a time and only sends
    the next after the previous outcome arrives.

    Args:
        task_queue: This worker's private command queue (``None`` ends the
            loop).
        result_queue: Parent-owned queue for messages.
        worker_index: Slot index attached to every message.
        recycle_after_tasks: Exit cleanly after this many successful runs
            (``0`` disables); the parent respawns the slot.
    """
    _configure_worker_runtime(worker_index)
    result_queue.put((worker_index, None, _MSG_STARTED, None))
    op: Any = None
    successful_tasks = 0
    while True:
        message = task_queue.get()
        if message is None:
            return
        if not isinstance(message, tuple) or not message:
            continue
        command = message[0]
        if command == _CMD_BIND:
            op = message[1]
            successful_tasks = 0
            result_queue.put((worker_index, None, _MSG_BOUND, None))
            continue
        if command != _CMD_RUN:
            continue
        task = message[1]
        if op is None:
            result_queue.put(
                (
                    worker_index,
                    task.task_id,
                    _STATUS_ERROR,
                    HABITAPIError(
                        f"Persistent worker {worker_index} received a run "
                        "before its operator was bound."
                    ),
                )
            )
            continue
        try:
            value = op(task.item)
        except MemoryError as exc:
            result_queue.put((worker_index, task.task_id, _STATUS_OOM, _picklable(exc)))
        except BaseException as exc:  # noqa: BLE001 - isolation is the point
            result_queue.put(
                (worker_index, task.task_id, _STATUS_ERROR, _picklable(exc))
            )
        else:
            # Pre-serialise synchronously: ``Queue.put`` pickles in a
            # background feeder thread whose failure would otherwise drop
            # the outcome silently and hang the parent's scheduling loop.
            try:
                pickle.dumps(value)
            except BaseException as exc:  # noqa: BLE001
                result_queue.put(
                    (
                        worker_index,
                        task.task_id,
                        _STATUS_ERROR,
                        HABITAPIError(
                            f"Result of {type(op).__name__} is not picklable: " f"{exc}"
                        ),
                    )
                )
            else:
                result_queue.put((worker_index, task.task_id, _STATUS_OK, value))
                successful_tasks += 1
                if recycle_after_tasks > 0 and successful_tasks >= recycle_after_tasks:
                    return


@dataclass(frozen=True)
class _Task:
    """One pending subject computation."""

    task_id: int
    subject_id: str
    cache_key: str
    item: Any


@dataclass
class _PersistentSlot:
    """Parent-side bookkeeping for one persistent worker."""

    worker_index: int
    proc: Any
    task_queue: Any
    started: bool
    bound: bool
    in_flight_task: Optional[int]
    dispatched_at: float
    consecutive_failures: int = 0
    successful_tasks: int = 0


def _detect_gpu_pool_size() -> int:
    """
    Probe the usable GPU count for ``cap_workers_to_gpu_pool``.

    PyTorch is asked first (it honours ``CUDA_VISIBLE_DEVICES``);
    ``nvidia-smi -L`` is the fallback. A zero means "no pool could be
    determined", in which case no capping is applied -- capping to zero
    workers would be absurd.

    Returns:
        The detected GPU count, or ``0`` when undetectable.
    """
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 - torch absent or broken is fine
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            return sum(
                1
                for line in completed.stdout.splitlines()
                if line.strip().startswith("GPU ")
            )
    except Exception:  # noqa: BLE001 - no nvidia-smi is fine
        pass
    return 0


class ProcessPoolBackend:
    """
    Execute subject-level work across child processes.

    This backend ports the v0.1 individual-level parallel machinery --
    timeouts, graceful shutdown, failure isolation, OOM backoff, resume --
    behind the :class:`~habit.contracts.ops.ExecutionBackend` protocol, so
    no algorithm ever manages a process pool itself.

    Args:
        workers: Parallel worker processes; ``1`` still runs the work in a
            child (the process boundary is the point of this backend).
        subject_timeout_sec: Wall-clock seconds per subject; ``None``
            disables the per-subject timeout.
        subject_spawn_timeout_sec: Seconds allowed for an isolated child to
            start; ``None`` disables it. Only meaningful in ``isolated``
            mode (persistent workers start once per round, not per
            subject).
        graceful_shutdown_sec: Seconds between ``terminate()`` and
            ``kill()`` when a process must be stopped.
        on_subject_failure: ``"continue"`` isolates a subject failure in
            its result slot; ``"fail_fast"`` aborts the run.
        oom_backoff: Reduce the effective worker count after a fatal
            memory error.
        oom_reduce_workers_by: Workers subtracted per OOM event; the
            effective count never drops below one.
        cap_workers_to_gpu_pool: Clamp ``workers`` to the detected GPU
            pool; no-op when no pool is detectable.
        parallel_mode: ``"persistent"`` keeps one long-lived worker per
            slot; ``"isolated"`` spawns one child process per subject.
        auto_retry_rounds: Extra dispatch rounds for failed subjects
            within one run; ``0`` disables.
        resume: Reuse checkpointed successes and honour recorded failures.
        retry_failed_subjects: Re-run subjects whose checkpoint records a
            failure instead of skipping them.
        force_rerun_subjects: Subject ids reprocessed even when a
            checkpoint success exists.
        clear_checkpoint_on_success: Clear the checkpoint store after a
            run with zero failures.
        persistent_worker_max_consecutive_failures: Restart a persistent
            slot after this many consecutive fatal-class failures.
        persistent_worker_recycle_after_tasks: Restart a persistent worker
            after this many successes (``0`` disables).
    """

    def __init__(
        self,
        workers: int = 1,
        *,
        subject_timeout_sec: Optional[float] = 900.0,
        subject_spawn_timeout_sec: Optional[float] = 120.0,
        graceful_shutdown_sec: float = 15.0,
        on_subject_failure: str = "continue",
        oom_backoff: bool = True,
        oom_reduce_workers_by: int = 1,
        cap_workers_to_gpu_pool: bool = False,
        parallel_mode: str = "persistent",
        auto_retry_rounds: int = 2,
        resume: bool = True,
        retry_failed_subjects: bool = False,
        force_rerun_subjects: Tuple[str, ...] = (),
        clear_checkpoint_on_success: bool = False,
        persistent_worker_max_consecutive_failures: int = 1,
        persistent_worker_recycle_after_tasks: int = 0,
    ) -> None:
        # RunPolicy owns the validation rules; building one here keeps the
        # two surfaces consistent by construction. The import is lazy on
        # purpose: ``habit.spec`` is not a layer ``habit.execution`` may
        # import at module load time (see tests/test_architecture_contracts).
        from habit.spec.policy import RunPolicy

        policy = RunPolicy(
            workers=workers,
            backend="process",
            subject_timeout_sec=subject_timeout_sec,
            subject_spawn_timeout_sec=subject_spawn_timeout_sec,
            graceful_shutdown_sec=graceful_shutdown_sec,
            on_subject_failure=on_subject_failure,
            oom_backoff=oom_backoff,
            oom_reduce_workers_by=oom_reduce_workers_by,
            cap_workers_to_gpu_pool=cap_workers_to_gpu_pool,
            resume=resume,
            parallel_mode=parallel_mode,
            auto_retry_rounds=auto_retry_rounds,
            retry_failed_subjects=retry_failed_subjects,
            force_rerun_subjects=tuple(force_rerun_subjects),
            clear_checkpoint_on_success=clear_checkpoint_on_success,
            persistent_worker_max_consecutive_failures=(
                persistent_worker_max_consecutive_failures
            ),
            persistent_worker_recycle_after_tasks=(
                persistent_worker_recycle_after_tasks
            ),
        )
        self.gpu_pool_size = _detect_gpu_pool_size() if cap_workers_to_gpu_pool else 0
        if cap_workers_to_gpu_pool and self.gpu_pool_size > 0:
            policy = replace(
                policy, workers=max(1, min(policy.workers, self.gpu_pool_size))
            )
        self._policy = policy
        # Optional multi-map session (see :meth:`reuse_workers`).
        self._reuse_depth = 0
        self._session_ctx: Any = None
        self._session_result_queue: Any = None
        self._session_slots: Dict[int, _PersistentSlot] = {}
        self._session_next_worker_index = 0
        self._session_bound_op_id: Optional[int] = None

    @classmethod
    def from_policy(cls, policy: "RunPolicy") -> "ProcessPoolBackend":
        """
        Build a backend from its declarative snapshot.

        Args:
            policy: The run policy to transcribe; every field maps onto the
                constructor parameter of the same name.

        Returns:
            The configured backend.
        """
        return cls(
            workers=policy.workers,
            subject_timeout_sec=policy.subject_timeout_sec,
            subject_spawn_timeout_sec=policy.subject_spawn_timeout_sec,
            graceful_shutdown_sec=policy.graceful_shutdown_sec,
            on_subject_failure=policy.on_subject_failure,
            oom_backoff=policy.oom_backoff,
            oom_reduce_workers_by=policy.oom_reduce_workers_by,
            cap_workers_to_gpu_pool=policy.cap_workers_to_gpu_pool,
            parallel_mode=policy.parallel_mode,
            auto_retry_rounds=policy.auto_retry_rounds,
            resume=policy.resume,
            retry_failed_subjects=policy.retry_failed_subjects,
            force_rerun_subjects=policy.force_rerun_subjects,
            clear_checkpoint_on_success=policy.clear_checkpoint_on_success,
            persistent_worker_max_consecutive_failures=(
                policy.persistent_worker_max_consecutive_failures
            ),
            persistent_worker_recycle_after_tasks=(
                policy.persistent_worker_recycle_after_tasks
            ),
        )

    @contextmanager
    def reuse_workers(self) -> Iterator["ProcessPoolBackend"]:
        """
        Keep persistent workers alive across successive :meth:`map` calls.

        Nested enters are reference-counted. Isolated mode is a no-op (each
        subject already owns a short-lived child). Recipes use this to avoid
        paying Windows spawn/import twice for two_step units + labels.
        """
        if self._policy.parallel_mode != "persistent":
            yield self
            return
        self._reuse_depth += 1
        try:
            yield self
        finally:
            self._reuse_depth -= 1
            if self._reuse_depth == 0:
                self._shutdown_worker_session()

    @property
    def policy(self) -> "RunPolicy":
        """Return the validated policy snapshot behind this backend."""
        return self._policy

    @property
    def workers(self) -> int:
        """Return the effective worker count (after any GPU capping)."""
        return self._policy.workers

    def map(
        self,
        op: SubjectOperator[TIn, TOut],
        items: Iterable[TIn],
        *,
        checkpoint: Optional[CheckpointStore] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[SubjectResult[TOut]]:
        """
        Apply ``op`` across ``items`` with checkpoint-aware resume.

        Results stream out in COMPLETION order (successes immediately,
        terminal failures after their retry rounds are exhausted); each
        :class:`SubjectResult` names its subject so callers restore the
        canonical order, per the backend protocol.

        Args:
            op: The subject-level operation to run.
            items: Subject-scoped inputs.
            checkpoint: Optional store for resume and persistence.
            progress: Optional callback receiving ``(completed, total)``.

        Yields:
            One :class:`SubjectResult` per item, exactly once.

        Raises:
            BaseException: The first subject failure under
                ``on_subject_failure="fail_fast"``.
        """
        materialised: List[TIn] = list(items)
        total = len(materialised)
        completed = 0

        def _report() -> None:
            if progress is not None:
                progress(completed, total)

        policy = self._policy
        forced = set(policy.force_rerun_subjects)
        pending: List[_Task] = []
        had_failure = False

        for index, item in enumerate(materialised):
            subject_id = _subject_id_of(item, index)
            cache_key = _cache_key_of(op, item, subject_id)
            task = _Task(index, subject_id, cache_key, item)
            try:
                pickle.dumps(item)
            except Exception as exc:
                # Pre-flight the pickling boundary in the parent: an
                # unpicklable payload would otherwise kill a queue feeder
                # thread silently and look like a hung worker.
                error = HABITAPIError(
                    f"Payload for subject {subject_id!r} is not picklable "
                    f"and cannot cross the process boundary: {exc}"
                )
                if policy.on_subject_failure == "fail_fast":
                    raise error
                had_failure = True
                completed += 1
                _report()
                yield SubjectResult(
                    subject_id=subject_id,
                    value=None,
                    error=error,
                    from_cache=False,
                )
                continue
            if checkpoint is not None and policy.resume:
                if subject_id in forced:
                    checkpoint.discard_failure(cache_key)
                else:
                    cached = checkpoint.get(cache_key)
                    if cached is not None:
                        completed += 1
                        _report()
                        yield SubjectResult(
                            subject_id=subject_id,
                            value=cached,
                            error=None,
                            from_cache=True,
                        )
                        continue
                    failure_message = checkpoint.get_failure(cache_key)
                    if failure_message is not None and not policy.retry_failed_subjects:
                        completed += 1
                        _report()
                        yield SubjectResult(
                            subject_id=subject_id,
                            value=None,
                            error=HABITAPIError(
                                "Subject has a recorded checkpoint failure: "
                                f"{failure_message}"
                            ),
                            from_cache=True,
                        )
                        continue
            pending.append(task)

        attempts: Dict[int, int] = {task.task_id: 0 for task in pending}

        while pending:
            retry: List[_Task] = []
            round_iter = self._execute_round(op, pending)
            try:
                for task, status, payload in round_iter:
                    if status == _STATUS_OK:
                        if checkpoint is not None:
                            checkpoint.put(task.cache_key, payload)
                        completed += 1
                        _report()
                        yield SubjectResult(
                            subject_id=task.subject_id,
                            value=payload,
                            error=None,
                            from_cache=False,
                        )
                        continue
                    # Named ``failure`` rather than ``exc``: Python deletes an
                    # ``except ... as exc`` target at the end of its block, so
                    # reusing that name here confuses both readers and mypy's
                    # deleted-variable analysis.
                    failure = (
                        payload
                        if isinstance(payload, BaseException)
                        else HABITAPIError(str(payload))
                    )
                    if policy.on_subject_failure == "fail_fast":
                        raise failure
                    had_failure = True
                    attempts[task.task_id] += 1
                    if attempts[task.task_id] <= policy.auto_retry_rounds:
                        retry.append(task)
                        continue
                    if checkpoint is not None:
                        checkpoint.put_failure(
                            task.cache_key, f"{type(failure).__name__}: {failure}"
                        )
                    completed += 1
                    _report()
                    yield SubjectResult(
                        subject_id=task.subject_id,
                        value=None,
                        error=failure,
                        from_cache=False,
                    )
            finally:
                # Closing the round generator terminates any surviving
                # workers (its own finally block), including the fail-fast
                # path above.
                round_iter.close()
            pending = retry

        if (
            checkpoint is not None
            and policy.clear_checkpoint_on_success
            and not had_failure
        ):
            checkpoint.clear()

    # ------------------------------------------------------------------
    # Round schedulers
    # ------------------------------------------------------------------

    def _execute_round(
        self, op: SubjectOperator[TIn, TOut], tasks: List[_Task]
    ) -> Generator[Tuple[_Task, str, Any], None, None]:
        """
        Run one dispatch round over ``tasks``.

        Args:
            op: The subject-level operator.
            tasks: Tasks to compute in this round.

        Yields:
            ``(task, status, payload)`` triples in completion order.
        """
        if self._policy.parallel_mode == "isolated":
            yield from self._round_isolated(op, tasks)
        else:
            yield from self._round_persistent(op, tasks)

    def _terminate(self, proc: Any) -> None:
        """
        Stop a child process with the graceful-shutdown ladder.

        Args:
            proc: The ``multiprocessing.Process`` to stop.
        """
        if not proc.is_alive():
            proc.join(timeout=0)
            return
        proc.terminate()
        proc.join(timeout=self._policy.graceful_shutdown_sec)
        if proc.is_alive():
            proc.kill()
            proc.join()

    def _round_isolated(
        self, op: SubjectOperator[TIn, TOut], tasks: List[_Task]
    ) -> Iterator[Tuple[_Task, str, Any]]:
        """
        Dispatch one child process per subject, up to ``workers`` at once.

        Args:
            op: The subject-level operator.
            tasks: Tasks to compute.

        Yields:
            ``(task, status, payload)`` triples in completion order.
        """
        ctx = multiprocessing.get_context("spawn")
        max_concurrent = max(1, self.workers)
        running: List[Dict[str, Any]] = []
        task_iter = iter(tasks)
        exhausted = False
        try:
            while not exhausted or running:
                while not exhausted and len(running) < max_concurrent:
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        exhausted = True
                        continue
                    result_queue = ctx.Queue()
                    # Slot index ≈ concurrent index so TorchRadiomics can
                    # hash across a multi-GPU pool when one is configured.
                    slot_index = len(running)
                    proc = ctx.Process(
                        target=_isolated_worker,
                        args=(op, task.item, result_queue, slot_index),
                        daemon=True,
                    )
                    proc.start()
                    running.append(
                        {
                            "proc": proc,
                            "queue": result_queue,
                            "task": task,
                            "started": False,
                            "outcome": None,
                            "dispatched_at": time.monotonic(),
                            "started_at": None,
                        }
                    )
                time.sleep(_POLL_INTERVAL_SEC)
                for slot in list(running):
                    while True:
                        try:
                            message = slot["queue"].get_nowait()
                        except queue_module.Empty:
                            break
                        if message == _MSG_STARTED:
                            slot["started"] = True
                            slot["started_at"] = time.monotonic()
                        else:
                            slot["outcome"] = message
                    proc = slot["proc"]
                    if slot["outcome"] is not None:
                        running.remove(slot)
                        proc.join(timeout=self._policy.graceful_shutdown_sec)
                        if proc.is_alive():
                            self._terminate(proc)
                        status, payload = slot["outcome"]
                        if status == _STATUS_OOM:
                            max_concurrent = self._oom_reduced(max_concurrent)
                        yield slot["task"], status, payload
                        continue
                    if not proc.is_alive():
                        running.remove(slot)
                        yield (
                            slot["task"],
                            _STATUS_ERROR,
                            _WorkerDiedError(
                                "Isolated worker exited with code "
                                f"{proc.exitcode} without reporting an outcome."
                            ),
                        )
                        continue
                    now = time.monotonic()
                    spawn_timeout = self._policy.subject_spawn_timeout_sec
                    if (
                        not slot["started"]
                        and spawn_timeout is not None
                        and now - slot["dispatched_at"] > spawn_timeout
                    ):
                        running.remove(slot)
                        self._terminate(proc)
                        yield (
                            slot["task"],
                            _STATUS_ERROR,
                            SubjectTimeoutError(
                                "Worker process did not start within "
                                f"{spawn_timeout}s."
                            ),
                        )
                        continue
                    subject_timeout = self._policy.subject_timeout_sec
                    if (
                        slot["started"]
                        and subject_timeout is not None
                        and now - slot["started_at"] > subject_timeout
                    ):
                        running.remove(slot)
                        self._terminate(proc)
                        yield (
                            slot["task"],
                            _STATUS_ERROR,
                            SubjectTimeoutError(
                                f"Subject exceeded its {subject_timeout}s "
                                "wall-clock budget."
                            ),
                        )
                        continue
        finally:
            for slot in running:
                self._terminate(slot["proc"])

    def _oom_reduced(self, current: int) -> int:
        """
        Apply one OOM backoff step to a worker count.

        Args:
            current: Current effective worker count.

        Returns:
            The reduced count (never below one, unchanged when backoff is
            disabled).
        """
        if not self._policy.oom_backoff:
            return current
        return max(1, current - self._policy.oom_reduce_workers_by)

    def _shutdown_worker_session(self) -> None:
        """Stop every persistent worker retained by :meth:`reuse_workers`."""
        for slot in list(self._session_slots.values()):
            self._stop_persistent_slot(slot)
        self._session_slots.clear()
        self._session_ctx = None
        self._session_result_queue = None
        self._session_next_worker_index = 0
        self._session_bound_op_id = None

    def _stop_persistent_slot(self, slot: _PersistentSlot) -> None:
        """Poison one persistent slot and join its process."""
        try:
            slot.task_queue.put(None)
        except Exception:  # noqa: BLE001 - queue already broken
            pass
        slot.proc.join(timeout=self._policy.graceful_shutdown_sec)
        if slot.proc.is_alive():
            self._terminate(slot.proc)

    def _round_persistent(
        self, op: SubjectOperator[TIn, TOut], tasks: List[_Task]
    ) -> Iterator[Tuple[_Task, str, Any]]:
        """
        Serve tasks from long-lived workers, one private queue per slot.

        Dispatch is parent-driven: a slot holds at most one task and
        receives the next only after reporting the previous outcome. A
        worker that is terminated therefore never takes queued-but-
        unaccounted tasks down with it, and late messages from replaced
        workers are dropped instead of double-counted. OOM backoff is
        realised as retirements consumed by slots as they become free, so
        busy workers always finish their current subject.

        When :meth:`reuse_workers` is active, slots and the result queue
        survive across ``map`` calls; only the operator is rebound.

        Args:
            op: The subject-level operator.
            tasks: Tasks to compute.

        Yields:
            ``(task, status, payload)`` triples in completion order.
        """
        keep_alive = self._reuse_depth > 0
        recycle_after = self._policy.persistent_worker_recycle_after_tasks
        max_consec = self._policy.persistent_worker_max_consecutive_failures

        if keep_alive and self._session_ctx is None:
            self._session_ctx = multiprocessing.get_context("spawn")
            self._session_result_queue = self._session_ctx.Queue()

        ctx = self._session_ctx if keep_alive else multiprocessing.get_context("spawn")
        result_queue = self._session_result_queue if keep_alive else ctx.Queue()
        assert ctx is not None and result_queue is not None

        task_by_id = {task.task_id: task for task in tasks}
        pending: List[_Task] = list(tasks)
        slots: Dict[int, _PersistentSlot] = self._session_slots if keep_alive else {}
        slot_count = max(1, min(self.workers, len(tasks))) if tasks else 0
        next_worker_index = self._session_next_worker_index if keep_alive else 0
        completed = 0
        total = len(tasks)
        pending_retirements = 0
        op_id = id(op)

        def _spawn_slot() -> int:
            nonlocal next_worker_index
            task_queue = ctx.Queue()
            proc = ctx.Process(
                target=_persistent_worker,
                args=(
                    task_queue,
                    result_queue,
                    next_worker_index,
                    recycle_after,
                ),
                daemon=True,
            )
            proc.start()
            slots[next_worker_index] = _PersistentSlot(
                worker_index=next_worker_index,
                proc=proc,
                task_queue=task_queue,
                started=False,
                bound=False,
                in_flight_task=None,
                dispatched_at=time.monotonic(),
            )
            next_worker_index += 1
            if keep_alive:
                self._session_next_worker_index = next_worker_index
            return next_worker_index - 1

        def _bind_slot(slot: _PersistentSlot) -> None:
            # An unpicklable operator surfaces here, in the parent.
            slot.bound = False
            slot.dispatched_at = time.monotonic()
            slot.task_queue.put((_CMD_BIND, op))

        def _dispatch(slot: _PersistentSlot) -> None:
            task = pending.pop(0)
            slot.in_flight_task = task.task_id
            slot.dispatched_at = time.monotonic()
            slot.task_queue.put((_CMD_RUN, task))

        def _restart_slot(old: _PersistentSlot) -> _PersistentSlot:
            self._stop_persistent_slot(old)
            slots.pop(old.worker_index, None)
            new_index = _spawn_slot()
            new_slot = slots[new_index]
            _bind_slot(new_slot)
            return new_slot

        def _note_outcome(slot: _PersistentSlot, kind: str) -> bool:
            """
            Update consecutive-failure counters; return whether to restart.

            Args:
                slot: Slot that just finished a run.
                kind: Outcome kind (``ok`` / ``error`` / ``oom``).

            Returns:
                ``True`` when the slot should be restarted before reuse.
            """
            if kind == _STATUS_OK:
                slot.consecutive_failures = 0
                slot.successful_tasks += 1
                return False
            if kind in (_STATUS_ERROR, _STATUS_OOM):
                slot.consecutive_failures += 1
                return slot.consecutive_failures >= max_consec
            return False

        try:
            # Grow the pool up to slot_count (session reuse may already have
            # some warm workers).
            while len(slots) < slot_count:
                _spawn_slot()

            # Always (re)bind before dispatching. Session reuse across recipe
            # stages changes the operator; skipping bind when ``bound`` was
            # still True from the previous map would silently run the old op
            # (e.g. units results under label cache keys).
            for slot in list(slots.values()):
                if slot.in_flight_task is None:
                    _bind_slot(slot)
            if keep_alive:
                self._session_bound_op_id = op_id

            # Wait until every idle slot is bound, then dispatch.
            unbound = [s for s in slots.values() if not s.bound]
            while unbound:
                try:
                    message = result_queue.get(timeout=_POLL_INTERVAL_SEC)
                except queue_module.Empty:
                    message = None
                if message is not None:
                    worker_index, _task_id, kind, _payload = message
                    maybe_slot = slots.get(worker_index)
                    if maybe_slot is None:
                        continue
                    slot = maybe_slot
                    if kind == _MSG_STARTED:
                        slot.started = True
                        continue
                    if kind == _MSG_BOUND:
                        slot.bound = True
                        slot.started = True
                        continue
                now = time.monotonic()
                spawn_timeout = self._policy.subject_spawn_timeout_sec
                for index, slot in list(slots.items()):
                    if slot.bound or slot.in_flight_task is not None:
                        continue
                    elapsed = now - slot.dispatched_at
                    if (
                        spawn_timeout is not None
                        and elapsed > spawn_timeout
                        and not slot.started
                    ):
                        # Startup hang during bind: replace the slot.
                        _restart_slot(slot)
                unbound = [s for s in slots.values() if not s.bound]

            for slot in list(slots.values()):
                if pending and slot.in_flight_task is None and slot.bound:
                    _dispatch(slot)

            while completed < total:
                try:
                    message = result_queue.get(timeout=_POLL_INTERVAL_SEC)
                except queue_module.Empty:
                    message = None

                if message is not None:
                    worker_index, task_id, kind, payload = message
                    maybe_slot = slots.get(worker_index)
                    if maybe_slot is None:
                        continue
                    slot = maybe_slot
                    if kind == _MSG_STARTED:
                        slot.started = True
                        continue
                    if kind == _MSG_BOUND:
                        slot.bound = True
                        slot.started = True
                        if pending and slot.in_flight_task is None:
                            _dispatch(slot)
                        continue
                    if task_id != slot.in_flight_task:
                        continue
                    completed += 1
                    slot.in_flight_task = None
                    restart = _note_outcome(slot, kind)
                    if kind == _STATUS_OOM and self._policy.oom_backoff:
                        pending_retirements = min(
                            pending_retirements + self._policy.oom_reduce_workers_by,
                            max(0, len(slots) - 1),
                        )
                    yield task_by_id[task_id], kind, payload
                    if pending_retirements > 0 and len(slots) > 1:
                        pending_retirements -= 1
                        slots.pop(worker_index)
                        self._stop_persistent_slot(slot)
                    elif restart:
                        new_slot = _restart_slot(slot)
                        if pending:
                            # Wait for bind ack on the next loop iteration.
                            pass
                        del new_slot
                    elif pending and slot.bound:
                        _dispatch(slot)

                now = time.monotonic()
                spawn_timeout = self._policy.subject_spawn_timeout_sec
                subject_timeout = self._policy.subject_timeout_sec
                timed_out: List[Tuple[int, SubjectTimeoutError]] = []
                for index, slot in slots.items():
                    if slot.in_flight_task is None:
                        continue
                    elapsed = now - slot.dispatched_at
                    if (
                        not slot.started
                        and spawn_timeout is not None
                        and elapsed > spawn_timeout
                    ):
                        timed_out.append(
                            (
                                index,
                                SubjectTimeoutError(
                                    "Worker process did not start within "
                                    f"{spawn_timeout}s."
                                ),
                            )
                        )
                    elif (
                        slot.started
                        and slot.bound
                        and subject_timeout is not None
                        and elapsed > subject_timeout
                    ):
                        timed_out.append(
                            (
                                index,
                                SubjectTimeoutError(
                                    f"Subject exceeded its {subject_timeout}s "
                                    "wall-clock budget."
                                ),
                            )
                        )
                for index, error in timed_out:
                    slot = slots.pop(index)
                    task_id = slot.in_flight_task
                    assert task_id is not None
                    self._terminate(slot.proc)
                    completed += 1
                    slot.consecutive_failures += 1
                    yield task_by_id[task_id], _STATUS_ERROR, error
                    if pending and completed < total:
                        new_slot = slots[_spawn_slot()]
                        _bind_slot(new_slot)

                for index, slot in list(slots.items()):
                    if slot.proc.is_alive():
                        continue
                    slots.pop(index)
                    if slot.in_flight_task is not None:
                        completed += 1
                        yield (
                            task_by_id[slot.in_flight_task],
                            _STATUS_ERROR,
                            _WorkerDiedError(
                                "Persistent worker exited with code "
                                f"{slot.proc.exitcode} mid-subject."
                            ),
                        )
                    if pending and completed < total and len(slots) < slot_count:
                        new_slot = slots[_spawn_slot()]
                        _bind_slot(new_slot)
        finally:
            if not keep_alive:
                for slot in list(slots.values()):
                    self._stop_persistent_slot(slot)
                slots.clear()
