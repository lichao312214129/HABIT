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
Fast synthetic parity checks for the L4 habitat recipes.

These tests mirror ``tests/recipes/test_recipes_golden_parity.py`` but run on
the in-memory synthetic cohort from :mod:`habit.datasets`, so they stay in the
default ``pytest -m "not slow"`` selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest

from tests.golden.fast._runner import (
    FAST_N_HABITATS,
    _light_habitat_spec,
    compare_fast_records,
    scrub_record,
    synthetic_cohort,
)

HABITAT_MAP_SUFFIX = "_habitats.nrrd"


class RecipeCase(NamedTuple):
    """One habitat design executed through its recipe."""

    name: str
    recipe: str


RECIPE_CASES = [
    RecipeCase(name="habitat_two_step", recipe="two_step"),
    RecipeCase(name="habitat_one_step", recipe="one_step"),
    RecipeCase(name="habitat_direct_pooling", recipe="direct_pooling"),
]


def _run_study(design: str, cohort, spec):
    """Run a habitat design through the object-style Study API."""
    from habit.recipes.study import Study

    return Study(spec=spec, design=design).fit_predict(cohort)


def _restrict(record: dict, suffix: str) -> dict:
    """Keep only artefacts whose relative path ends with ``suffix``."""
    kept = [name for name in record["artefacts"] if name.endswith(suffix)]
    return {
        "artefacts": sorted(kept),
        "fingerprints": {name: record["fingerprints"][name] for name in kept},
    }


@pytest.mark.integration
@pytest.mark.parametrize("case", RECIPE_CASES, ids=lambda case: case.name)
def test_fast_recipe_writes_expected_habitat_maps(case: RecipeCase, tmp_path: Path) -> None:
    """
    A recipe run on the synthetic cohort writes stable habitat label maps.

    Args:
        case: Habitat design under test.
        tmp_path: Scratch directory for the writer.
    """
    from scripts.make_golden_baseline import fingerprint_output_dir

    from tests.golden.fast.conftest import load_fast_baseline

    spec = _light_habitat_spec(two_step=case.recipe == "two_step")
    result = _run_study(case.recipe, synthetic_cohort(), spec)
    result.save(tmp_path)
    baseline = load_fast_baseline(case.name)
    expected = _restrict(baseline, HABITAT_MAP_SUFFIX)
    actual = _restrict(scrub_record(fingerprint_output_dir(tmp_path)), HABITAT_MAP_SUFFIX)
    problems = compare_fast_records(expected, actual)
    assert not problems, (
        f"{case.name}: synthetic recipe maps diverged:\n" + "\n".join(problems[:20])
    )


@pytest.mark.integration
@pytest.mark.parametrize("case", RECIPE_CASES, ids=lambda case: case.name)
def test_fast_recipe_settles_on_three_habitats(case: RecipeCase) -> None:
    """Cluster count stays pinned at three on the synthetic cohort."""
    spec = _light_habitat_spec(two_step=case.recipe == "two_step")
    result = _run_study(case.recipe, synthetic_cohort(), spec)
    counts = {int(habitat_map.label_array.max()) for habitat_map in result.habitat_maps}
    assert counts == {FAST_N_HABITATS}


@pytest.mark.integration
def test_fast_predict_relabels_training_cohort_identically(tmp_path: Path) -> None:
    """Saved model round trip reproduces training labels on synthetic data."""
    from habit.contracts.habitat import HabitatModel
    from habit.recipes.study import Study

    cohort = synthetic_cohort()
    spec = _light_habitat_spec(two_step=True)
    trained = Study(spec=spec, design="two_step").fit_predict(cohort)
    assert trained.habitat_model is not None
    archive = trained.habitat_model.save(tmp_path / "model.habitatmodel")
    reloaded = HabitatModel.load(archive)
    predicted = Study.from_model(reloaded, spec).predict(cohort)
    expected = {item.subject_id: item.label_array for item in trained.habitat_maps}
    for habitat_map in predicted.habitat_maps:
        assert np.array_equal(
            np.asarray(expected[habitat_map.subject_id]),
            np.asarray(habitat_map.label_array),
        )
