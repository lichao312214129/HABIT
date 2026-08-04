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
"""Execution backends: optional accelerators, never a precondition.

``SerialBackend`` is the reference implementation and the default used by
``Cohort.map`` when no backend is supplied. It runs one subject at a time in
the current process, which keeps notebook debugging trivial: a failure stops
(or is captured) at exactly the subject that caused it.

Checkpoint semantics mirror :class:`~habit.execution.process_pool.ProcessPoolBackend`
field-by-field for the policy subset that makes sense without a process
boundary: cached successes are reused, recorded failures are skipped unless
retried, and forced subjects are always recomputed. Reads are gated by
``resume``; writes are not -- a checkpoint store attached to a run always
records that run's outcomes, so a later resumed run can skip them (v0.1
behaviour). Process-boundary concerns (timeouts, OOM backoff, in-run retry
rounds) deliberately do not exist here.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, TypeVar

from habit.exceptions import HABITAPIError
from habit.contracts.ops import SubjectOperator, SubjectResult
from habit.execution.checkpoint import CheckpointStore

__all__ = ["SerialBackend"]

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


def _subject_id_of(item: Any, fallback_index: int) -> str:
    """
    Derive the subject identity used in result slots and progress reports.

    Domain payloads carry ``subject_id`` (``Subject``, ``VoxelFeatureField``,
    ``Supervoxelization``, ...). Foreign payloads fall back to their position
    so the backend still returns a well-formed :class:`SubjectResult`.

    Args:
        item: One subject-scoped payload.
        fallback_index: Position used when the payload has no identity.

    Returns:
        The subject identifier string.
    """
    subject_id = getattr(item, "subject_id", None)
    if isinstance(subject_id, str) and subject_id:
        return subject_id
    return f"item_{fallback_index}"


def _cache_key_of(op: Any, item: Any, subject_id: str) -> str:
    """
    Build the checkpoint key for one computation.

    Operators implementing ``cache_key`` (the ``SubjectOperator`` contract)
    control their own key; plain callables get a key combining the operator
    class and the subject identity.

    Args:
        op: The subject-level operator.
        item: The subject-scoped payload.
        subject_id: Identity derived by :func:`_subject_id_of`.

    Returns:
        A stable checkpoint key.
    """
    cache_key = getattr(op, "cache_key", None)
    if callable(cache_key):
        return str(cache_key(item))
    return f"{type(op).__module__}.{type(op).__qualname__}:{subject_id}"


class SerialBackend:
    """
    Run subject-level work one item at a time in the current process.

    This is the reference :class:`~habit.contracts.ops.ExecutionBackend`
    implementation: correct by construction, trivially debuggable, and the
    default behind ``Cohort.map(op)``.

    Args:
        on_subject_failure: ``"continue"`` captures a subject's exception in
            its :class:`SubjectResult` and proceeds; ``"fail_fast"`` re-raises
            the first failure immediately.
        resume: Reuse checkpointed successes and honour recorded failures.
            Writes to the store happen regardless, matching
            :class:`~habit.execution.process_pool.ProcessPoolBackend`.
        retry_failed_subjects: Re-run subjects whose checkpoint records a
            failure instead of skipping them.
        force_rerun_subjects: Subject ids reprocessed even when a
            checkpoint success exists.
        clear_checkpoint_on_success: Clear the checkpoint store after a
            run with zero failures.
    """

    def __init__(
        self,
        on_subject_failure: str = "continue",
        *,
        resume: bool = True,
        retry_failed_subjects: bool = False,
        force_rerun_subjects: Tuple[str, ...] = (),
        clear_checkpoint_on_success: bool = False,
    ) -> None:
        if on_subject_failure not in ("continue", "fail_fast"):
            raise ValueError(
                "on_subject_failure must be 'continue' or 'fail_fast'; got "
                f"{on_subject_failure!r}."
            )
        self.on_subject_failure = on_subject_failure
        self.resume = bool(resume)
        self.retry_failed_subjects = bool(retry_failed_subjects)
        self.force_rerun_subjects = tuple(force_rerun_subjects)
        self.clear_checkpoint_on_success = bool(clear_checkpoint_on_success)

    def map(
        self,
        op: SubjectOperator[TIn, TOut],
        items: Iterable[TIn],
        *,
        checkpoint: Optional[CheckpointStore] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[SubjectResult[TOut]]:
        """
        Apply ``op`` to each item in iteration order.

        Args:
            op: The subject-level operation to run.
            items: Subject-scoped inputs.
            checkpoint: Optional store used to skip already-computed subjects
                and to persist new results as they complete. Cached successes
                and recorded failures are honoured only when this backend's
                ``resume`` is set; outcomes are written regardless.
            progress: Optional callback receiving ``(completed, total)``.

        Yields:
            One :class:`SubjectResult` per item, in input order.

        Raises:
            BaseException: The first subject failure, when
                ``on_subject_failure`` is ``"fail_fast"``.
        """
        materialised: List[TIn] = list(items)
        total = len(materialised)
        completed = 0
        had_failure = False
        forced = set(self.force_rerun_subjects)
        for index, item in enumerate(materialised):
            subject_id = _subject_id_of(item, index)
            cache_key = _cache_key_of(op, item, subject_id)
            if checkpoint is not None and self.resume:
                if subject_id in forced:
                    # A forced rerun recomputes and overwrites; any recorded
                    # failure for the key is stale from this point on.
                    checkpoint.discard_failure(cache_key)
                else:
                    cached = checkpoint.get(cache_key)
                    if cached is not None:
                        completed += 1
                        if progress is not None:
                            progress(completed, total)
                        yield SubjectResult(
                            subject_id=subject_id,
                            value=cached,
                            error=None,
                            from_cache=True,
                        )
                        continue
                    failure_message = checkpoint.get_failure(cache_key)
                    if failure_message is not None and not self.retry_failed_subjects:
                        completed += 1
                        if progress is not None:
                            progress(completed, total)
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
            try:
                value = op(item)
            except BaseException as exc:  # noqa: BLE001 - isolation is the point
                if self.on_subject_failure == "fail_fast":
                    raise
                had_failure = True
                if checkpoint is not None:
                    # Serial has no in-run retry rounds, so a captured failure
                    # is terminal by definition and belongs in the store.
                    checkpoint.put_failure(cache_key, f"{type(exc).__name__}: {exc}")
                completed += 1
                if progress is not None:
                    progress(completed, total)
                yield SubjectResult(
                    subject_id=subject_id,
                    value=None,
                    error=exc,
                    from_cache=False,
                )
                continue
            if checkpoint is not None:
                checkpoint.put(cache_key, value)
            completed += 1
            if progress is not None:
                progress(completed, total)
            yield SubjectResult(
                subject_id=subject_id,
                value=value,
                error=None,
                from_cache=False,
            )
        if (
            checkpoint is not None
            and self.clear_checkpoint_on_success
            and not had_failure
        ):
            checkpoint.clear()
