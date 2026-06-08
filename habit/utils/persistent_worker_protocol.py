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
IPC message types for persistent parallel worker pools.

All messages are plain picklable objects for ``multiprocessing.Queue`` under spawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from habit.utils.isolated_runner import ProcessingResult

WorkerCommandKind = Literal["RUN", "STOP"]
WorkerReplyKind = Literal["READY", "RESULT", "WORKER_EXIT"]


@dataclass(frozen=True)
class WorkerRunCommand:
    """Parent → worker: execute one item."""

    kind: WorkerCommandKind = "RUN"
    item: Any = None


@dataclass(frozen=True)
class WorkerStopCommand:
    """Parent → worker: exit the run loop."""

    kind: WorkerCommandKind = "STOP"


@dataclass
class WorkerReadyReply:
    """Worker → parent: initialization finished."""

    kind: WorkerReplyKind = "READY"
    worker_slot: int = 0


@dataclass
class WorkerResultReply:
    """Worker → parent: one item finished."""

    kind: WorkerReplyKind = "RESULT"
    worker_slot: int = 0
    proc_result: Optional[ProcessingResult] = None


@dataclass
class WorkerExitReply:
    """Worker → parent: run loop ended."""

    kind: WorkerReplyKind = "WORKER_EXIT"
    worker_slot: int = 0
