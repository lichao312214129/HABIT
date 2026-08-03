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

Every scheduling concern (worker counts, failure policy, resume) lives here
so that algorithms contain no scheduling code at all, and so the same
``HabitatSpec`` runs identically on a laptop or a cluster -- only the policy
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from habit.api.exceptions import HABITAPIError

__all__ = ["RunPolicy"]

#: Failure policies a backend may implement.
_FAILURE_POLICIES = ("continue", "fail_fast")

#: Execution backends selectable by name.
_BACKEND_NAMES = ("serial", "process")


@dataclass(frozen=True)
class RunPolicy:
    """
    Execution policy for a study run.

    Attributes:
        workers: Parallel worker processes; ``1`` means serial execution.
        on_failure: ``"continue"`` isolates a subject failure in its result
            slot; ``"fail_fast"`` aborts the run on the first failure.
        seed: Seed applied to every :class:`~habit.domain.protocols.Seedable`
            component of the pipeline.
        backend: Execution backend name; ``"serial"`` or ``"process"``.
        checkpoint_path: Directory for resumable subject results; ``None``
            disables resume.
    """

    workers: int = 1
    on_failure: str = "continue"
    seed: int = 0
    backend: str = "serial"
    checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate policy values at the boundary."""
        if not isinstance(self.workers, int) or self.workers < 1:
            raise HABITAPIError(
                f"RunPolicy.workers must be a positive integer; got {self.workers!r}."
            )
        if self.on_failure not in _FAILURE_POLICIES:
            raise HABITAPIError(
                f"RunPolicy.on_failure must be one of {_FAILURE_POLICIES}; "
                f"got {self.on_failure!r}."
            )
        if self.backend not in _BACKEND_NAMES:
            raise HABITAPIError(
                f"RunPolicy.backend must be one of {_BACKEND_NAMES}; "
                f"got {self.backend!r}."
            )
        if self.checkpoint_path is not None:
            object.__setattr__(
                self, "checkpoint_path", str(self.checkpoint_path)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (YAML isomorphic)."""
        return {
            "workers": self.workers,
            "on_failure": self.on_failure,
            "seed": self.seed,
            "backend": self.backend,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunPolicy":
        """Rebuild a run policy from its dict form, tolerating omissions."""
        return cls(
            workers=int(payload.get("workers", 1)),
            on_failure=str(payload.get("on_failure", "continue")),
            seed=int(payload.get("seed", 0)),
            backend=str(payload.get("backend", "serial")),
            checkpoint_path=payload.get("checkpoint_path"),
        )
