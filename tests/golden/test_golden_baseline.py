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
Regression tests against the frozen v1.0 golden baseline.

These tests are the numerical acceptance gate for the v1.0 release. They come
in two tiers so that the cheap half runs in ordinary CI:

* Contract tests (fast, no imaging): assert that the committed baseline still
  describes the artefacts every HABIT run has to produce -- habitat label maps,
  supervoxel label maps, the habitats table, the fitted ``.habitatmodel``,
  ``run_manifest.json``, and the visualisation tree. A refactor that quietly
  stops writing the ``.nrrd`` maps or the cluster plots breaks these without
  needing to run anything.

* Reproduction tests (slow, needs ``demo_data/``): re-run each CLI case and
  compare voxel-by-voxel and value-by-value against the baseline. Run manifest
  timestamps and run ids are ignored; label maps, tables, and model archives
  are compared strictly.

Regenerate the baseline with::

    python scripts/make_golden_baseline.py

Run the slow tier with::

    pytest tests/golden -m slow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_golden_baseline import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    GOLDEN_CASES,
    GoldenCase,
    baseline_path,
    compare_records,
    run_case,
)

#: Artefact families v1.0 produces for a two-step habitat train run. Written as
#: suffix/segment patterns rather than exact filenames so the assertion states
#: the contract ("label maps and plots are produced") instead of restating the
#: baseline file it is checking. v1 no longer writes ``habitat_pipeline.pkl`` or
#: the v0.1 supervoxel-clustering plot tree; predict uses
#: ``habitat_model.habitatmodel`` instead.
REQUIRED_TWO_STEP_ARTEFACTS: Dict[str, str] = {
    "_habitats.nrrd": "per-subject habitat label map",
    "_supervoxel.nrrd": "per-subject supervoxel label map",
    "habitats.parquet": "population habitats table",
    "habitat_model.habitatmodel": "fitted v1 habitat model",
    "run_manifest.json": "run provenance manifest",
    "visualizations/habitat_clustering": "habitat clustering plots",
}


def _load_baseline(case_name: str) -> Dict[str, Any]:
    """Load one committed baseline record, skipping when absent."""
    path = baseline_path(case_name)
    if not path.is_file():
        pytest.skip(
            f"No golden baseline for '{case_name}'. "
            "Generate it locally with: python scripts/make_golden_baseline.py"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _demo_data_available() -> bool:
    """Return whether the untracked demo dataset is present locally."""
    # The shipped habitat configs consume data_dir = demo_data/preprocessed
    # with images/ (and masks/) underneath; probe the same layout.
    return (REPO_ROOT / "demo_data" / "preprocessed" / "images").is_dir()


@pytest.mark.unit
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.name)
def test_baseline_exists_and_is_well_formed(case: GoldenCase) -> None:
    """Every declared case has a committed, structurally valid baseline."""
    baseline = _load_baseline(case.name)
    assert baseline["case"] == case.name
    assert baseline["config"] == case.config
    assert baseline["artefacts"], "baseline records no artefacts"
    assert set(baseline["artefacts"]) == set(baseline["fingerprints"])


@pytest.mark.unit
def test_two_step_baseline_covers_the_artefact_contract() -> None:
    """
    The baseline pins the full v1.0 output contract, not just the numbers.

    This is the guard for regressions where a implementation computes correct
    habitats but returns them only in memory: the label maps, model archive,
    manifest and plots CLI users rely on would silently disappear.
    """
    baseline = _load_baseline("habitat_two_step")
    artefacts: List[str] = baseline["artefacts"]

    for pattern, description in REQUIRED_TWO_STEP_ARTEFACTS.items():
        assert any(pattern in artefact for artefact in artefacts), (
            f"Baseline is missing the {description} ({pattern!r}); "
            "the artefact contract cannot be enforced without it."
        )


@pytest.mark.unit
def test_label_maps_are_pinned_voxelwise() -> None:
    """Habitat and supervoxel maps carry a digest plus their geometry."""
    baseline = _load_baseline("habitat_two_step")
    fingerprints: Dict[str, Any] = baseline["fingerprints"]
    label_maps = {
        key: value for key, value in fingerprints.items() if value.get("kind") == "array"
    }
    assert label_maps, "no label maps captured in the baseline"

    for key, record in label_maps.items():
        assert record["sha256"], f"{key}: missing voxel digest"
        assert record["shape"], f"{key}: missing shape"
        for field in ("spacing", "origin", "direction"):
            assert record[field], f"{key}: missing {field}; geometry drift would go unnoticed"
        assert record["label_values"], f"{key}: no label values recorded"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.name)
def test_case_reproduces_baseline(case: GoldenCase, tmp_path: Path) -> None:
    """
    Re-running a case reproduces the baseline artefact-for-artefact.

    Args:
        case: The CLI case under test.
        tmp_path: Pytest-provided scratch root, keeping the run isolated from
            the directory the baseline was generated in.
    """
    if not _demo_data_available():
        pytest.skip("demo_data/ is not present; golden reproduction needs local imaging data")

    baseline = _load_baseline(case.name)
    current = run_case(case, tmp_path)
    problems = compare_records(baseline, current)
    assert not problems, "Golden baseline drift:\n" + "\n".join(problems[:40])


@pytest.mark.slow
@pytest.mark.integration
def test_repeated_run_is_deterministic() -> None:
    """
    The same case run twice in a row produces identical artefacts.

    Without this, a baseline mismatch after the refactor could always be
    blamed on inherent run-to-run randomness rather than on the refactor.
    """
    if not _demo_data_available():
        pytest.skip("demo_data/ is not present; determinism check needs local imaging data")

    case = next(item for item in GOLDEN_CASES if item.name == "habitat_two_step")
    first = run_case(case, DEFAULT_OUT_ROOT / "determinism_a")
    second = run_case(case, DEFAULT_OUT_ROOT / "determinism_b")
    problems = compare_records(first, second)
    assert not problems, "Repeated runs differ:\n" + "\n".join(problems[:40])
