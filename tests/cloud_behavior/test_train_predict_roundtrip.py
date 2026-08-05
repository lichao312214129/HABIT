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
"""Train/predict round-trip tests for habitat models."""

from __future__ import annotations

from pathlib import Path

import pytest

from habit.contracts.habitat import HabitatModel
from habit.recipes import apply_habitat_model
from habit.spec.specs import HabitatSpec
from tests.cloud_behavior.helpers import (
    count_label_mismatches,
    load_cohort_from_tree,
    run_two_step_on_tree,
)


@pytest.mark.integration
def test_train_save_load_predict_matches_training_labels(
    synthetic_tree: Path,
    habitat_spec: HabitatSpec,
    tmp_path: Path,
) -> None:
    """
    ``two_step`` → save → ``HabitatModel.load`` → ``apply_habitat_model`` round-trip.

    Re-applying the fitted model to the training cohort must reproduce the
    original habitat label maps with zero voxel mismatches.
    """
    train_result = run_two_step_on_tree(synthetic_tree, habitat_spec)
    assert train_result.habitat_model is not None

    artefact_dir = tmp_path / "train_artefacts"
    train_result.save(artefact_dir)

    loaded_model = HabitatModel.load(artefact_dir / "habitat_model.habitatmodel")
    cohort = load_cohort_from_tree(synthetic_tree)
    predict_result = apply_habitat_model(cohort, habitat_spec, loaded_model)

    mismatches = count_label_mismatches(
        train_result.habitat_maps,
        predict_result.habitat_maps,
    )
    assert mismatches == 0
