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

Shared by the habitat CLI and ``run_from_yaml`` so backend selection stays
in one place. Timeout is **not** coupled to spawn: the library default
``subject_timeout_sec=900`` must not force a process pool when the user
asked for serial ``workers=1``.
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

    Process pool is selected when any of the following hold:

    * ``backend == "process"`` (explicit request)
    * ``workers > 1`` (subject-level parallelism)
    * ``parallel_mode == "isolated"`` (fresh child per subject)

    A positive ``subject_timeout_sec`` alone does **not** force the process
    pool. The typed default is ``900.0``; coupling timeout to spawn would
    turn every ``workers=1`` / ``backend="serial"`` run into a child
    process and erase the serial path. Per-subject timeout isolation still
    requires ``backend="process"`` or ``parallel_mode="isolated"``.

    Args:
        policy: Declarative execution policy.

    Returns:
        ``True`` when :class:`ProcessPoolBackend` must be used.
    """
    if policy.backend == "process":
        return True
    if policy.parallel_mode == "isolated":
        return True
    if policy.workers > 1:
        return True
    return False


def backend_from_policy(policy: "RunPolicy") -> Union[ProcessPoolBackend, SerialBackend]:
    """
    Build the execution backend a policy asks for.

    Args:
        policy: Translated run policy.

    Returns:
        A process-pool backend when the policy requires spawn
        (``backend == "process"``, ``workers > 1``, or
        ``parallel_mode == "isolated"``); otherwise a serial backend
        carrying the policy's checkpoint / failure flags. Timeout knobs
        apply only under the process-pool backend.
    """
    if should_use_process_pool(policy):
        # Ensure the backend snapshot records process semantics even when
        # YAML left ``backend: serial`` but requested workers > 1 or
        # isolated mode.
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
