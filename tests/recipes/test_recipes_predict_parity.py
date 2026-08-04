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
"""
The published-model round trip: fit, save, load, apply.

``HabitatModel`` is the artefact HABIT expects to outlive the run that made
it -- someone reloads it years later to reproduce a published figure. The
danger is not that loading fails loudly; it is that a lossy round trip loads
successfully and returns a plausible but different habitat map. These tests
close that gap by requiring the reloaded model to relabel its own training
cohort *identically*, and by pinning that agreement against the frozen
``habitat_two_step_predict`` baseline the v0.1 CLI produced.

Run with::

    pytest tests/recipes -m slow
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.recipes.conftest import (
    demo_data_available,
    load_baseline,
    load_demo_cohort,
    spec_and_data_root,
)

#: The design that has a frozen predict baseline to compare against.
TRAIN_CONFIG = "config/habitat/config_habitat_two_step.yaml"

#: Golden case whose artefacts pin the CLI's predict-mode behaviour.
PREDICT_CASE = "habitat_two_step_predict"


def _fit_two_step() -> Any:
    """Fit the two-step study on the demo cohort."""
    import habit.recipes as recipes

    spec, root = spec_and_data_root(TRAIN_CONFIG)
    cohort = load_demo_cohort(spec, root)
    return recipes.two_step(cohort, spec), spec, cohort


@pytest.mark.slow
@pytest.mark.integration
def test_saved_model_relabels_its_training_cohort_identically(tmp_path: Path) -> None:
    """
    A model saved, reloaded and re-applied reproduces the training labels.

    Args:
        tmp_path: Scratch directory for the ``.habitatmodel`` archive.
    """
    if not demo_data_available():
        pytest.skip("demo_data/ is not present; the predict round trip needs imaging data")

    import habit.recipes as recipes
    from habit.contracts.habitat import HabitatModel

    result, spec, cohort = _fit_two_step()
    assert result.habitat_model is not None

    archive = result.habitat_model.save(tmp_path / "habitat_model.habitatmodel")
    reloaded = HabitatModel.load(archive)
    assert reloaded.model_id == result.habitat_model.model_id, (
        "reloading changed the model identity; the archive is not faithful"
    )

    predicted = recipes.apply_habitat_model(cohort, spec, reloaded)
    trained_maps = {m.subject_id: m.label_array for m in result.habitat_maps}
    for habitat_map in predicted.habitat_maps:
        expected = trained_maps[habitat_map.subject_id]
        actual = habitat_map.label_array
        mismatched = int(np.count_nonzero(np.asarray(expected) != np.asarray(actual)))
        assert mismatched == 0, (
            f"{habitat_map.subject_id}: {mismatched} voxels change label when the "
            "saved model relabels the cohort it was fitted on"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_predict_labels_match_the_frozen_cli_baseline(tmp_path: Path) -> None:
    """
    Predict-mode labels reproduce the frozen v0.1 predict baseline.

    The baseline was produced by ``habit get-habitat`` in ``run_mode:
    predict`` from the pipeline the training case wrote, so agreeing with it
    proves the in-memory model application is the same operation the CLI
    performs -- not merely self-consistent.

    Args:
        tmp_path: Scratch directory the writer persists into.
    """
    if not demo_data_available():
        pytest.skip("demo_data/ is not present; the predict round trip needs imaging data")

    import hashlib

    import habit.recipes as recipes
    from habit.contracts.habitat import HabitatModel

    baseline = load_baseline(PREDICT_CASE)
    result, spec, cohort = _fit_two_step()
    reloaded = HabitatModel.load(
        result.habitat_model.save(tmp_path / "habitat_model.habitatmodel")
    )
    predicted = recipes.apply_habitat_model(cohort, spec, reloaded)

    for habitat_map in predicted.habitat_maps:
        key = f"{habitat_map.subject_id}_habitats.nrrd"
        expected = baseline["fingerprints"][key]
        # Cast to the width v0.1 stored, for the reason documented in
        # tests/recipes/test_recipes_golden_parity.py: v1 unifies the label
        # dtype, and a width change must not read as a label change.
        array = np.ascontiguousarray(
            np.asarray(habitat_map.label_array).astype(np.dtype(expected["dtype"]))
        )
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        assert digest == expected["sha256"], f"{key}: predicted labels differ from baseline"
        assert list(array.shape) == expected["shape"], f"{key}: shape differs"


@pytest.mark.slow
@pytest.mark.integration
def test_cli_predict_baseline_agrees_with_its_training_baseline() -> None:
    """
    The frozen baselines themselves record train/predict agreement.

    This is a property of the recorded data rather than of a fresh run, and
    it is what makes the predict case worth freezing: applying the trained
    pipeline back onto the training cohort must be a no-op, otherwise every
    downstream claim about model portability is unfounded.
    """
    train = load_baseline("habitat_two_step")["fingerprints"]
    predict = load_baseline(PREDICT_CASE)["fingerprints"]
    shared = [
        key
        for key in train
        if key.endswith("_habitats.nrrd") and key in predict
    ]
    assert shared, "no habitat maps common to the train and predict baselines"
    for key in shared:
        assert train[key]["sha256"] == predict[key]["sha256"], (
            f"{key}: the frozen predict labels differ from the frozen train labels"
        )
