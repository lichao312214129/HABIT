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
"""RunPolicy: how a study executes, decoupled from what it computes.

Every scheduling concern (worker counts, per-subject timeouts, failure
policy, OOM backoff, resume) lives here so that algorithms contain no
scheduling code at all, and so the same ``HabitatSpec`` runs identically on
a laptop or a cluster -- only the policy changes.

The field set is the declarative snapshot of the execution parameters an
:class:`~habit.contracts.ops.ExecutionBackend` accepts; field names match
the backend keyword arguments verbatim so the YAML form and the Python form
stay one-to-one (developer/api_upgrade/07 §9.7). Note what is NOT here:
the random seed. Seeds change the scientific result, so they belong to
``HabitatSpec.random_seed`` and take part in the spec fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from habit.exceptions import HABITAPIError

__all__ = ["RunPolicy"]

#: Failure policies a backend may implement.
_FAILURE_POLICIES = ("continue", "fail_fast")

#: Execution backends selectable by name.
_BACKEND_NAMES = ("serial", "process")

#: Per-subject worker lifecycle strategies.
_PARALLEL_MODES = ("persistent", "isolated")


@dataclass(frozen=True)
class RunPolicy:
    """
    Execution policy for a study run.

    Attributes:
        workers: Parallel worker processes; ``1`` means serial execution.
        backend: Execution backend name; ``"serial"`` or ``"process"``.
        subject_timeout_sec: Wall-clock seconds allowed per subject before
            it is marked failed; ``None`` disables the per-subject timeout.
        subject_spawn_timeout_sec: Wall-clock seconds allowed for a worker
            process to start; ``None`` disables the spawn timeout.
        graceful_shutdown_sec: Seconds to wait after terminate() before
            kill() when a subject exceeds its timeout.
        on_subject_failure: ``"continue"`` isolates a subject failure in its
            result slot; ``"fail_fast"`` aborts the run on the first failure.
        oom_backoff: Reduce workers after a fatal memory error so pending
            subjects can still run.
        oom_reduce_workers_by: Workers subtracted per OOM backoff step; the
            effective worker count never drops below one.
        cap_workers_to_gpu_pool: Clamp worker count to the usable GPU pool
            for steps whose components require a GPU.
        resume: Reuse checkpointed subject results when a checkpoint
            directory is available.
        checkpoint_dir: Directory for resumable subject results; ``None``
            lets the runner pick its default location.
        parallel_mode: ``"persistent"`` keeps one long-lived worker per slot;
            ``"isolated"`` spawns one child process per subject.
        auto_retry_rounds: Extra dispatch rounds for checkpoint-failed
            subjects within one run; ``0`` disables.
        retry_failed_subjects: Re-queue checkpoint-failed subjects on the
            next resumed run.
        force_rerun_subjects: Subject IDs reprocessed even when a checkpoint
            exists.
        clear_checkpoint_on_success: Remove the checkpoint directory after a
            successful run.
        strict_checkpoint_hash: Raise instead of discarding checkpoints when
            the recorded spec fingerprint is incompatible.
    """

    workers: int = 1
    backend: str = "serial"
    subject_timeout_sec: Optional[float] = 900.0
    subject_spawn_timeout_sec: Optional[float] = 120.0
    graceful_shutdown_sec: float = 15.0
    on_subject_failure: str = "continue"
    oom_backoff: bool = True
    oom_reduce_workers_by: int = 1
    cap_workers_to_gpu_pool: bool = False
    resume: bool = True
    checkpoint_dir: Optional[str] = None
    parallel_mode: str = "persistent"
    auto_retry_rounds: int = 2
    retry_failed_subjects: bool = False
    force_rerun_subjects: Tuple[str, ...] = field(default_factory=tuple)
    clear_checkpoint_on_success: bool = False
    strict_checkpoint_hash: bool = False

    def __post_init__(self) -> None:
        """Validate policy values at the boundary."""
        if not isinstance(self.workers, int) or self.workers < 1:
            raise HABITAPIError(
                f"RunPolicy.workers must be a positive integer; got {self.workers!r}."
            )
        if self.backend not in _BACKEND_NAMES:
            raise HABITAPIError(
                f"RunPolicy.backend must be one of {_BACKEND_NAMES}; "
                f"got {self.backend!r}."
            )
        for name in ("subject_timeout_sec", "subject_spawn_timeout_sec"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise HABITAPIError(
                    f"RunPolicy.{name} must be positive when set; got {value!r}. "
                    "Use None to disable the timeout."
                )
        if self.graceful_shutdown_sec <= 0:
            raise HABITAPIError(
                "RunPolicy.graceful_shutdown_sec must be positive; "
                f"got {self.graceful_shutdown_sec!r}."
            )
        if self.on_subject_failure not in _FAILURE_POLICIES:
            raise HABITAPIError(
                f"RunPolicy.on_subject_failure must be one of {_FAILURE_POLICIES}; "
                f"got {self.on_subject_failure!r}."
            )
        if not isinstance(self.oom_reduce_workers_by, int) or self.oom_reduce_workers_by < 1:
            raise HABITAPIError(
                "RunPolicy.oom_reduce_workers_by must be a positive integer; "
                f"got {self.oom_reduce_workers_by!r}."
            )
        if self.parallel_mode not in _PARALLEL_MODES:
            raise HABITAPIError(
                f"RunPolicy.parallel_mode must be one of {_PARALLEL_MODES}; "
                f"got {self.parallel_mode!r}."
            )
        if not isinstance(self.auto_retry_rounds, int) or self.auto_retry_rounds < 0:
            raise HABITAPIError(
                "RunPolicy.auto_retry_rounds must be a non-negative integer; "
                f"got {self.auto_retry_rounds!r}."
            )
        if self.checkpoint_dir is not None:
            object.__setattr__(self, "checkpoint_dir", str(self.checkpoint_dir))
        object.__setattr__(
            self, "force_rerun_subjects", tuple(self.force_rerun_subjects)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (YAML isomorphic)."""
        return {
            "workers": self.workers,
            "backend": self.backend,
            "subject_timeout_sec": self.subject_timeout_sec,
            "subject_spawn_timeout_sec": self.subject_spawn_timeout_sec,
            "graceful_shutdown_sec": self.graceful_shutdown_sec,
            "on_subject_failure": self.on_subject_failure,
            "oom_backoff": self.oom_backoff,
            "oom_reduce_workers_by": self.oom_reduce_workers_by,
            "cap_workers_to_gpu_pool": self.cap_workers_to_gpu_pool,
            "resume": self.resume,
            "checkpoint_dir": self.checkpoint_dir,
            "parallel_mode": self.parallel_mode,
            "auto_retry_rounds": self.auto_retry_rounds,
            "retry_failed_subjects": self.retry_failed_subjects,
            "force_rerun_subjects": list(self.force_rerun_subjects),
            "clear_checkpoint_on_success": self.clear_checkpoint_on_success,
            "strict_checkpoint_hash": self.strict_checkpoint_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunPolicy":
        """
        Rebuild a run policy from its dict form, tolerating omissions.

        Unknown keys are rejected so a misspelled field surfaces at load
        time instead of being silently dropped.

        Args:
            payload: Mapping as produced by :meth:`to_dict`; every key is
                optional.

        Returns:
            The reconstructed policy.

        Raises:
            HABITAPIError: On unknown keys or invalid values.
        """
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise HABITAPIError(
                f"Unknown RunPolicy field(s): {', '.join(unknown)}. "
                f"Valid fields: {', '.join(sorted(known))}."
            )
        kwargs: Dict[str, Any] = {}
        for name, value in payload.items():
            if name == "force_rerun_subjects":
                value = tuple(value or ())
            kwargs[name] = value
        return cls(**kwargs)
