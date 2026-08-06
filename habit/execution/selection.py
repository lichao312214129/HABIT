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
"""Map a :class:`~habit.spec.RunPolicy` onto a concrete execution backend.

Shared by the habitat CLI and ``run_from_yaml`` so the v0.1 spawn rule —
workers > 1 **or** a positive per-subject timeout — stays in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from habit.execution.backends import SerialBackend
from habit.execution.process_pool import ProcessPoolBackend

if TYPE_CHECKING:
    from habit.spec.policy import RunPolicy

__all__ = ["should_use_process_pool", "backend_from_policy"]


def should_use_process_pool(policy: "RunPolicy") -> bool:
    """
    Return whether the policy requires a spawn process-pool backend.

    Mirrors v0.1 ``_should_use_spawn_workers`` at the policy gate: a positive
    ``subject_timeout_sec`` needs a child process even when ``workers == 1``,
    otherwise hung subjects cannot be terminated from the parent.

    Args:
        policy: Declarative execution policy.

    Returns:
        ``True`` when :class:`ProcessPoolBackend` must be used.
    """
    if policy.backend == "process":
        return True
    timeout = policy.subject_timeout_sec
    return timeout is not None and timeout > 0


def backend_from_policy(policy: "RunPolicy") -> Union[ProcessPoolBackend, SerialBackend]:
    """
    Build the execution backend a policy asks for.

    Args:
        policy: Translated run policy.

    Returns:
        A process-pool backend when the policy requires spawn isolation
        (``backend == "process"`` or a positive subject timeout); otherwise
        a serial backend carrying the policy's checkpoint / failure flags.
    """
    if should_use_process_pool(policy):
        # Ensure the backend snapshot records process semantics even when the
        # YAML left ``backend: serial`` but armed a per-subject timeout.
        if policy.backend != "process":
            from dataclasses import replace

            policy = replace(policy, backend="process")
        return ProcessPoolBackend.from_policy(policy)
    return SerialBackend(
        on_subject_failure=policy.on_subject_failure,
        resume=policy.resume,
        retry_failed_subjects=policy.retry_failed_subjects,
        force_rerun_subjects=policy.force_rerun_subjects,
        clear_checkpoint_on_success=policy.clear_checkpoint_on_success,
    )
