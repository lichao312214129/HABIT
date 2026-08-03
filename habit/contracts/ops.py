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
"""Two-level operator and execution contracts (L2).

Every HABIT computation is either SUBJECT-LEVEL or COHORT-LEVEL. That single
distinction simultaneously defines the parallelism boundary, the checkpoint
boundary, the train/predict boundary, and -- in future -- the federation
boundary, where subject-level work runs inside the hospital and only
supervoxel features leave it.

ONE SUBJECT IS THE ATOMIC CALL: every subject-level operator is a plain
callable on one subject's payload, so ``op(subject)`` works with no cohort,
no backend, and no configuration. Cohorts, execution backends and checkpoints
are optional machinery layered on top, never a precondition for doing one
piece of work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    # Typing-only references: these modules sit in higher layers and are
    # therefore imported lazily at runtime, never at module import time.
    from habit.contracts.subject import Cohort
    from habit.contracts.habitat import HabitatMap, HabitatModel
    from habit.contracts.table import FeatureTable
    from habit.execution.checkpoint import CheckpointStore
    from habit.spec.specs import Spec

__all__ = [
    "SubjectOperator",
    "CohortOperator",
    "SubjectResult",
    "ExecutionBackend",
    "DataSource",
    "ResultWriter",
]

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@runtime_checkable
class SubjectOperator(Protocol, Generic[TIn, TOut]):
    """
    A computation that touches exactly one subject.

    Declaring this is a contract, not a hint: it tells the execution backend
    that the work may be parallelised, checkpointed, retried, isolated on
    failure, and -- in a federated deployment -- executed inside the hospital
    that owns the images.

    Note what this protocol does NOT introduce: a second method name. It is
    ``__call__`` plus two pieces of metadata, so every one of the
    subject-level domain protocols satisfies it automatically and no plugin
    author ever writes an adapter.

    Implementations must be free of shared mutable state so they can be sent
    to a worker process.
    """

    @property
    def spec(self) -> "Spec":
        """Return the algorithm specification, used as part of the cache key."""

    def __call__(self, item: TIn) -> TOut:
        """
        Process one subject's payload.

        Args:
            item: The subject-scoped input.

        Returns:
            The subject-scoped output.
        """

    def cache_key(self, item: TIn) -> str:
        """
        Return a stable key identifying this computation for checkpointing.

        Args:
            item: The subject-scoped input.

        Returns:
            A key combining the subject identity and the spec fingerprint, so
            that changing an algorithm parameter correctly invalidates a
            resumed run instead of silently reusing stale results.
        """


@runtime_checkable
class CohortOperator(Protocol, Generic[TIn, TOut]):
    """
    A computation that must observe the whole cohort at once.

    Cohort-level operations cannot be parallelised across subjects and cannot
    be resumed per subject. Habitat model fitting and population-level feature
    preprocessing are the two instances in HABIT.
    """

    @property
    def spec(self) -> "Spec":
        """Return the algorithm specification."""

    def fit(self, items: Sequence[TIn], **context: Any) -> TOut:
        """
        Aggregate across subjects to produce a shared artefact.

        Args:
            items: Subject-level payloads in a defined order.
            **context: Optional keyword context an implementation may accept,
                e.g. a habitat model fitter takes ``cohort=`` to record a
                non-identifiable fingerprint.

        Returns:
            The cohort-level artefact, e.g. a ``HabitatModel``.
        """


@dataclass(frozen=True)
class SubjectResult(Generic[TOut]):
    """
    Result slot for one subject, distinguishing success from isolated failure.

    Batch habitat analysis must be able to continue when a single subject
    fails, while still reporting that failure honestly. Returning an explicit
    result rather than raising keeps that policy in the backend instead of
    scattering try/except through the algorithms.

    Attributes:
        subject_id: Subject this result belongs to.
        value: Computed result when successful, otherwise ``None``.
        error: Captured exception when failed, otherwise ``None``.
        from_cache: Whether the value was restored from a checkpoint instead
            of being recomputed.
    """

    subject_id: str
    value: Optional[TOut]
    error: Optional[BaseException]
    from_cache: bool = False

    def result(self) -> TOut:
        """
        Return the value or re-raise the captured failure.

        Named after ``concurrent.futures.Future.result()``, the
        standard-library anchor for "give me the value or re-raise the error".

        Returns:
            The successful value.

        Raises:
            BaseException: The originally captured error, when this result
                represents a failure.
        """
        if self.error is not None:
            raise self.error
        return self.value  # type: ignore[return-value]


@runtime_checkable
class ExecutionBackend(Protocol):
    """
    Strategy for executing subject-level work.

    Every scheduling concern that v0.1 kept in the configuration schema --
    worker counts, per-subject timeouts, graceful shutdown, spawn timeouts,
    failure policy, OOM backoff, resume -- belongs here instead. Algorithms
    then contain no scheduling code at all, and adding a Dask or cluster
    backend requires no change to any algorithm.

    A backend is an OPTIONAL ACCELERATOR, never a precondition.
    ``op(subject)`` is always available directly, ``Cohort.map(op)`` runs the
    whole cohort with an implicit serial backend, and an explicit backend is
    constructed only when the user wants parallelism, timeouts or resume.
    """

    def map(
        self,
        op: SubjectOperator[TIn, TOut],
        items: Iterable[TIn],
        *,
        checkpoint: Optional["CheckpointStore"] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Iterator[SubjectResult[TOut]]:
        """
        Apply a subject-level operation across many subjects.

        Args:
            op: The subject-level operation to run.
            items: Subject-scoped inputs.
            checkpoint: Optional store used to skip already-computed subjects
                and to persist new results as they complete.
            progress: Optional callback receiving ``(completed, total)``.

        Returns:
            An iterator of per-subject results; each result names its subject
            so callers can restore the canonical order when a backend
            completes out of order.
        """


@runtime_checkable
class DataSource(Protocol):
    """
    Anything that can produce a cohort.

    This protocol is the concrete mechanism behind the goal of embedding
    HABIT into the wider ecosystem. The v0.1 directory convention becomes one
    implementation among several rather than the only way in, so data
    prepared by nnU-Net, MONAI, a DICOM export, or an in-memory notebook
    session are all equally valid entry points.
    """

    def load(self) -> "Cohort":
        """
        Build the cohort described by this source.

        Returns:
            A cohort with a defined, reproducible subject order.
        """


@runtime_checkable
class ResultWriter(Protocol):
    """
    Anything that can persist HABIT outputs.

    Separating the writer from the algorithms is what allows a caller to run
    a full habitat analysis entirely in memory, which is impossible in v0.1
    where every workflow writes to an output directory by construction.
    """

    def write_habitat_map(self, habitat_map: "HabitatMap") -> Optional[str]:
        """Persist one habitat map and return its location, when applicable."""

    def write_feature_table(
        self, table: "FeatureTable", name: str
    ) -> Optional[str]:
        """Persist one feature table and return its location, when applicable."""

    def write_habitat_model(self, model: "HabitatModel") -> Optional[str]:
        """Persist a fitted habitat model and return its location."""
