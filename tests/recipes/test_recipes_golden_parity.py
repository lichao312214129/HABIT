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
Numerical parity between the L4 recipes and the frozen v0.1 baseline.

The recipes are the first callers of the v1 stack that produce a complete
study, so they are also the first place the refactor can be proved right: the
same legacy YAML, translated by :class:`~habit.spec.legacy.LegacyConfigAdapter`
and executed in memory, must yield the habitat maps the v0.1 CLI wrote --
voxel for voxel, on the same grid.

The comparison deliberately goes through
:class:`~habit.adapters.writers.DirectoryResultWriter` and the golden
fingerprint helpers rather than comparing arrays directly. Passing therefore
means both halves of the contract hold: the numbers are unchanged *and* the L1
writer still lays them out the way v0.1 users' downstream scripts expect.

Run with::

    pytest tests/recipes -m slow
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple

import pytest

from tests.recipes.conftest import (
    demo_data_available,
    load_baseline,
    load_demo_cohort,
    spec_and_data_root,
)

from scripts.make_golden_baseline import compare_records, fingerprint_output_dir

#: Suffix of the artefact both paths must agree on. The v0.1 run also emits
#: supervoxel maps, plots and a parquet table; those belong to the CLI's
#: reporting layer (stage 4), not to the recipe, so they are out of scope here
#: and stay pinned by ``tests/golden/test_golden_baseline.py``.
HABITAT_MAP_SUFFIX = "_habitats.nrrd"


class RecipeCase(NamedTuple):
    """One golden habitat case rerun through its v1 recipe."""

    #: Golden baseline / case name.
    name: str
    #: Legacy YAML the baseline was generated from.
    config: str
    #: Dotted attribute of ``habit.recipes`` implementing the design.
    recipe: str
    #: Habitat count the frozen baseline settled on.
    n_habitats: int


RECIPE_CASES: List[RecipeCase] = [
    RecipeCase(
        name="habitat_two_step",
        config="config/habitat/config_habitat_two_step.yaml",
        recipe="two_step",
        n_habitats=4,
    ),
    RecipeCase(
        name="habitat_one_step",
        config="config/habitat/config_habitat_one_step_raw_concat_train.yaml",
        recipe="one_step",
        n_habitats=4,
    ),
    RecipeCase(
        name="habitat_direct_pooling",
        config="config/habitat/config_habitat_direct_pooling.yaml",
        recipe="direct_pooling",
        n_habitats=4,
    ),
]


def _recipe(name: str) -> Callable[..., Any]:
    """Resolve a recipe function by name."""
    import habit.recipes as recipes

    return getattr(recipes, name)


#: Study results keyed by case name. A habitat run costs tens of seconds, and
#: the assertions below inspect different facets of the same run rather than
#: needing independent ones, so the run is shared across them.
_RESULTS: Dict[str, Any] = {}


def _study_result(case: RecipeCase) -> Any:
    """
    Run one case through its recipe, reusing an earlier run when possible.

    Args:
        case: Habitat design to execute.

    Returns:
        The resulting :class:`~habit.recipes.result.StudyResult`.
    """
    if case.name not in _RESULTS:
        spec, root = spec_and_data_root(case.config)
        cohort = load_demo_cohort(spec, root)
        _RESULTS[case.name] = _recipe(case.recipe)(cohort, spec)
    return _RESULTS[case.name]


def _restrict(record: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """
    Keep only the artefacts whose name ends with ``suffix``.

    Args:
        record: A baseline or freshly captured fingerprint record.
        suffix: Artefact filename suffix to retain.

    Returns:
        A record of the same shape holding just the selected artefacts.
    """
    kept = [name for name in record["artefacts"] if name.endswith(suffix)]
    return {
        "artefacts": sorted(kept),
        "fingerprints": {name: record["fingerprints"][name] for name in kept},
    }


def _match_storage_dtype(
    current: Dict[str, Any], baseline: Dict[str, Any], out_dir: Path
) -> Dict[str, Any]:
    """
    Recompute array digests in the baseline's on-disk integer type.

    v0.1 had no single label dtype: the two-step writer emitted ``uint16``
    while the one-step and direct-pooling writers emitted ``int32``, purely
    because the maps travelled through different code paths. The v1 writer
    unifies them on ``int32``, so a byte-level digest would report a
    difference that is a storage-width change, not a change in any label.
    Casting to the recorded width keeps the comparison exactly as strict
    about the labels themselves while letting the unification through.

    Args:
        current: Freshly captured record for ``out_dir``.
        baseline: The frozen record supplying the storage dtypes.
        out_dir: Directory the artefacts were written to.

    Returns:
        A copy of ``current`` whose array digests use the baseline dtypes.
    """
    import hashlib

    import numpy as np
    import SimpleITK as sitk

    adjusted = {"artefacts": list(current["artefacts"]), "fingerprints": {}}
    for name, record in current["fingerprints"].items():
        expected = baseline["fingerprints"].get(name)
        if record.get("kind") != "array" or expected is None:
            adjusted["fingerprints"][name] = record
            continue
        array = sitk.GetArrayFromImage(sitk.ReadImage(str(out_dir / name)))
        cast = np.ascontiguousarray(array.astype(np.dtype(expected["dtype"])))
        assert np.array_equal(cast, array), (
            f"{name}: labels do not survive the cast to {expected['dtype']}; "
            "the storage type is losing information, not just changing width"
        )
        adjusted["fingerprints"][name] = {
            **record,
            "dtype": expected["dtype"],
            "sha256": hashlib.sha256(cast.tobytes()).hexdigest(),
        }
    return adjusted


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("case", RECIPE_CASES, ids=lambda case: case.name)
def test_recipe_reproduces_baseline_habitat_maps(case: RecipeCase, tmp_path: Path) -> None:
    """
    A recipe run in memory reproduces the CLI's habitat maps exactly.

    Args:
        case: Habitat design under test.
        tmp_path: Scratch directory the writer persists into.
    """
    if not demo_data_available():
        pytest.skip("demo_data/ is not present; recipe parity needs local imaging data")

    baseline = load_baseline(case.name)
    result = _study_result(case)
    result.save(tmp_path)

    expected = _restrict(baseline, HABITAT_MAP_SUFFIX)
    current = _restrict(fingerprint_output_dir(tmp_path), HABITAT_MAP_SUFFIX)
    problems = compare_records(expected, _match_storage_dtype(current, expected, tmp_path))
    assert not problems, (
        f"{case.name}: recipe output diverges from the frozen baseline:\n"
        + "\n".join(problems[:40])
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("case", RECIPE_CASES, ids=lambda case: case.name)
def test_recipe_settles_on_the_baseline_habitat_count(case: RecipeCase) -> None:
    """
    Cluster-count selection lands where the baseline did.

    Checked separately from the voxel digests because a habitat-count drift is
    the failure the digests report least legibly: every map differs, with no
    hint that the cause was selection rather than clustering.

    Args:
        case: Habitat design under test.
    """
    if not demo_data_available():
        pytest.skip("demo_data/ is not present; recipe parity needs local imaging data")

    result = _study_result(case)
    counts = {int(habitat_map.label_array.max()) for habitat_map in result.habitat_maps}
    assert counts == {case.n_habitats}, (
        f"{case.name}: habitat counts {sorted(counts)} != baseline {case.n_habitats}"
    )
