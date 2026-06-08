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
Helpers for wiring checkpoint steps into habitat pipeline recipes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from ..config_schemas import HabitatAnalysisConfig

if TYPE_CHECKING:
    from ..pipelines.base_pipeline import BasePipelineStep


def append_checkpoint_save_step(
    steps: List[Tuple[str, "BasePipelineStep"]],
    config: HabitatAnalysisConfig,
) -> None:
    """
    Append the individual-level checkpoint save step for train-mode pipelines.

    Args:
        steps: Mutable recipe step list (individual steps must already be present).
        config: Habitat analysis configuration.
    """
    if config.run_mode != "train":
        return
    from .step import CheckpointSaveStep

    steps.append(("checkpoint_save", CheckpointSaveStep(config)))
