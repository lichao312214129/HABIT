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
