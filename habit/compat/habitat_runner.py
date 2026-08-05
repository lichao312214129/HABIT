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
"""Habitat train/predict workflow runner (L1 compat).

Routes v1 ``.habitatmodel`` archives and train configs through the L4 habitat
recipes (same assembly as ``habit habitat`` / ``run_from_yaml``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from habit.utils.log_utils import get_module_logger

__all__ = ["run_habitat_analysis_from_config"]

_LOG = get_module_logger(__name__)
_V1_MODEL_NAME = "habitat_model.habitatmodel"
_LEGACY_PICKLE_MESSAGE = (
    "Legacy v0.1 pickle pipelines are not supported in HABIT v1.0. "
    "Train a model to produce {model_name!r}, then run predict with "
    "pipeline_path pointing at that archive or call "
    "habit.recipes.apply_habitat_model in Python."
)


def run_habitat_analysis_from_config(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run habitat segmentation in train or predict mode.

    v1 model archives and train configs execute through the L4 recipes.

    Args:
        config: Validated :class:`~habit.schemas.workflows.habitat.HabitatAnalysisConfig`.
        logger: Optional run logger.
        output_dir: Optional output directory override written onto ``config``.

    Returns:
        Habitat feature table as a pandas DataFrame.

    Raises:
        ValueError: When predict mode lacks ``pipeline_path`` or the artefact
            is a legacy v0.1 raw pickle.
        FileNotFoundError: When the pipeline file does not exist.
    """
    log = logger or _LOG
    if output_dir is not None:
        config.out_dir = output_dir

    if str(config.run_mode) == "predict":
        if not config.pipeline_path:
            raise ValueError(
                "In 'predict' mode, pipeline_path is required in the YAML "
                "or via CLI override."
            )
        pipeline_file = Path(config.pipeline_path)
        if not pipeline_file.is_file():
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
        if not _is_v1_model_archive(pipeline_file):
            raise ValueError(
                _LEGACY_PICKLE_MESSAGE.format(model_name=_V1_MODEL_NAME)
                + f" Got: {pipeline_file}"
            )

    from habit.recipes.yaml_runner import (
        _habitat_predict,
        _habitat_train,
        _save_habitat_result,
    )

    if str(config.run_mode) == "predict":
        log.info(
            "Running habitat predict through v1 recipes (pipeline=%s)",
            config.pipeline_path,
        )
        result = _habitat_predict(config, logger=log)
    else:
        log.info(
            "Running habitat train through v1 recipes (clustering_mode=%s)",
            config.habitat_segmentation.clustering_mode,
        )
        result = _habitat_train(config, logger=log)

    _save_habitat_result(result, config)
    if result.habitat_model is not None:
        log.info(
            "Saved fitted habitat model to %s",
            Path(config.out_dir) / _V1_MODEL_NAME,
        )

    return result.features.frame.copy()


def _is_v1_model_archive(path: Path) -> bool:
    """Return whether the artefact is a v1 zip archive vs a v0.1 pickle."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"PK\x03\x04"
    except OSError:
        return False
