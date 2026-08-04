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
Synthetic fast golden gate for the v1 stack.

These tests replace the slow demo-data reproduction tier in ordinary CI.
Every case uses :mod:`habit.datasets` synthetic inputs, fixed ``n_habitats=3``,
and the v1 recipes plus :class:`~habit.adapters.writers.DirectoryResultWriter`
layout. Manifest timestamps and software fingerprints are scrubbed before
comparison.

Generate or refresh the committed baselines with::

    python scripts/make_fast_golden_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.golden.fast._runner import FAST_GOLDEN_CASES, FAST_N_HABITATS
from tests.golden.fast.conftest import compare_fast_records, load_fast_baseline, run_case


@pytest.mark.unit
@pytest.mark.parametrize("case", FAST_GOLDEN_CASES, ids=lambda case: case.name)
def test_fast_baseline_exists(case) -> None:
    """Every declared fast case has a committed baseline on disk."""
    baseline = load_fast_baseline(case.name)
    assert baseline["case"] == case.name
    assert baseline["artefacts"]
    assert set(baseline["artefacts"]) == set(baseline["fingerprints"])


@pytest.mark.integration
@pytest.mark.parametrize("case", FAST_GOLDEN_CASES, ids=lambda case: case.name)
def test_fast_case_reproduces_baseline(case, tmp_path: Path) -> None:
    """
    Re-running a fast case reproduces its committed fingerprint record.

    Args:
        case: Synthetic golden case under test.
        tmp_path: Isolated scratch directory for the run.
    """
    baseline = load_fast_baseline(case.name)
    current = run_case(case, tmp_path / case.name)
    problems = compare_fast_records(baseline, current)
    assert not problems, (
        f"{case.name}: fast golden drift:\n" + "\n".join(problems[:40])
    )


@pytest.mark.integration
@pytest.mark.parametrize("case", FAST_GOLDEN_CASES, ids=lambda case: case.name)
def test_fast_case_is_deterministic(case, tmp_path: Path) -> None:
    """
    Two consecutive runs of the same fast case match exactly.

    Args:
        case: Synthetic golden case under test.
        tmp_path: Isolated scratch directory for both runs.
    """
    first = run_case(case, tmp_path / "run_a")
    second = run_case(case, tmp_path / "run_b")
    problems = compare_fast_records(first, second)
    assert not problems, (
        f"{case.name}: repeated fast runs differ:\n" + "\n".join(problems[:40])
    )


@pytest.mark.integration
def test_fast_two_step_settles_on_three_habitats(tmp_path: Path) -> None:
    """The canonical two-step case always yields three habitat labels."""
    case = next(item for item in FAST_GOLDEN_CASES if item.name == "habitat_two_step")
    result = case.runner(tmp_path / "two_step")
    counts = {int(habitat_map.label_array.max()) for habitat_map in result.habitat_maps}
    assert counts == {FAST_N_HABITATS}
