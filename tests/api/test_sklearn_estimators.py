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
"""Contract tests for config-driven workflow runners still exposed publicly.

The old config/disk sklearn facades (``HabitClassifier``, ``HabitatClusterer``,
``ClinicalPreprocessor``, …) were removed; sklearn interop now lives under
``habit.domain.sklearn_interop`` and ``habit.compat.sklearn.as_estimator``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import habit


@pytest.mark.unit
def test_dicom_habitat_and_ml_runners_delegate_to_core() -> None:
    """Every remaining top-level workflow runner preserves its core delegation."""
    config = MagicMock()
    config.out_dir = "habitat-output"
    config.output_dir = "dicom-output"
    config.output = "ml-output"
    config.run_mode = "train"
    logger = MagicMock()
    with (
        patch("habit.api.dicom_sort.coerce_config", return_value=config),
        patch(
            "habit.api.habitat.coerce_config",
            return_value=config,
        ),
        patch(
            "habit.api.machine_learning.coerce_config",
            return_value=config,
        ),
        patch("habit.compat.dicom_sort_runner.run_dicom_sort") as dicom_run,
        patch(
            "habit.compat.habitat_runner.run_habitat_analysis_from_config",
            return_value=pd.DataFrame(),
        ) as habitat_run,
        patch(
            "habit.compat.ml_runner.run_ml_from_config",
            return_value=MagicMock(metrics={}),
        ) as ml_run,
        patch(
            "habit.compat.ml_runner.run_kfold_from_config",
            return_value=MagicMock(),
        ) as kfold_run,
        patch("habit.api.dicom_sort.create_run_manifest"),
        patch("habit.api.dicom_sort.write_run_manifest"),
        patch("habit.api.habitat.create_run_manifest"),
        patch("habit.api.habitat.write_run_manifest"),
        patch("habit.api.machine_learning.create_run_manifest"),
        patch("habit.api.machine_learning.write_run_manifest"),
    ):
        habit.run_dicom_sort(config)
        assert habit.run_habitat_analysis(config, logger=logger).data.empty
        habit.run_ml(config, logger=logger, output_dir="ml-output")
        habit.run_kfold(config, logger=logger, output_dir="kfold-output")

    dicom_run.assert_called_once_with(config, logger=None)
    habitat_run.assert_called_once_with(config, logger=logger)
    ml_run.assert_called_once_with(config, logger=logger, output_dir="ml-output")
    kfold_run.assert_called_once_with(
        config,
        logger=logger,
        output_dir="kfold-output",
    )
