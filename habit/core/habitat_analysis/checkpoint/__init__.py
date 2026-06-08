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
Resumable training checkpoints for habitat analysis.

All checkpoint-related code lives in this package:

- :mod:`manager` — manifest and per-subject pickle I/O
- :mod:`step` — individual-level ``checkpoint_save`` pipeline step
- :mod:`stage` — resume filtering and parallel orchestration for Stage 1
- :mod:`recipe` — helpers to append ``checkpoint_save`` to pipeline recipes
"""

from __future__ import annotations

from typing import Any

from .manager import (
    CHECKPOINT_VERSION,
    MANIFEST_FILENAME,
    SUBJECTS_SUBDIR,
    CheckpointManifest,
    HabitatTrainCheckpoint,
    compute_config_hash,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "MANIFEST_FILENAME",
    "SUBJECTS_SUBDIR",
    "CheckpointManifest",
    "CheckpointSaveStep",
    "HabitatTrainCheckpoint",
    "IndividualCheckpointStage",
    "append_checkpoint_save_step",
    "compute_config_hash",
]


def __getattr__(name: str) -> Any:
    """Lazy imports to avoid circular dependency with ``base_pipeline``."""
    if name == "append_checkpoint_save_step":
        from .recipe import append_checkpoint_save_step

        return append_checkpoint_save_step
    if name == "CheckpointSaveStep":
        from .step import CheckpointSaveStep

        return CheckpointSaveStep
    if name == "IndividualCheckpointStage":
        from .stage import IndividualCheckpointStage

        return IndividualCheckpointStage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
