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
"""Shared helpers for cloud behavior tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from click.testing import Result

from habit.adapters.directory import DirectoryDataSource
from habit.contracts.habitat import HabitatMap
from habit.recipes.result import StudyResult
from habit.spec.legacy import LegacyConfigAdapter
from habit.spec.specs import HabitatSpec

from tests.cloud_behavior.synthetic_data import MODALITIES, ROI_NAME, SYNTHETIC_SEED

#: JSON leaves that may differ between otherwise identical runs.
_VOLATILE_MANIFEST_KEYS: Tuple[str, ...] = (
    "started_at",
    "finished_at",
    "created_at",
    "run_id",
)


def spec_from_v0_payload(payload: Mapping[str, object]) -> HabitatSpec:
    """
    Translate one v0 habitat YAML mapping into a :class:`HabitatSpec`.

    Args:
        payload: Parsed v0 habitat configuration mapping.

    Returns:
        Translated habitat specification.
    """
    translation = LegacyConfigAdapter().translate(payload, "habitat")
    spec_payload = translation.document["spec"]
    assert isinstance(spec_payload, dict)
    return HabitatSpec.from_dict(spec_payload)


def load_cohort_from_tree(data_root: Path) -> object:
    """
    Load the synthetic directory tree as a :class:`~habit.contracts.subject.Cohort`.

    Args:
        data_root: Cohort root produced by :func:`build_synthetic_tree`.

    Returns:
        Loaded cohort with modalities declared by the minimal config.
    """
    return DirectoryDataSource(
        data_root,
        modalities=list(MODALITIES),
        roi=ROI_NAME,
    ).load()


def habitat_label_digest(maps: Sequence[HabitatMap]) -> str:
    """
    Hash per-subject habitat label arrays for cross-path comparisons.

    Args:
        maps: Habitat label maps in cohort order.

    Returns:
        Hex digest stable for identical label images.
    """
    digest = hashlib.sha256()
    for habitat_map in maps:
        digest.update(habitat_map.subject_id.encode("utf-8"))
        digest.update(np.ascontiguousarray(habitat_map.label_array).tobytes())
    return digest.hexdigest()


def assert_habitat_maps_equal(
    left: Sequence[HabitatMap],
    right: Sequence[HabitatMap],
) -> None:
    """
    Assert two habitat-map sequences are exactly identical voxel-wise.

    Args:
        left: Reference habitat maps.
        right: Candidate habitat maps.

    Raises:
        AssertionError: When subject ids or label arrays differ.
    """
    assert len(left) == len(right)
    for reference, candidate in zip(left, right):
        assert reference.subject_id == candidate.subject_id
        assert np.array_equal(reference.label_array, candidate.label_array)


def assert_parquet_frames_equal(left: Path, right: Path) -> None:
    """
    Assert two ``habitats.parquet`` files contain identical values.

    Args:
        left: Reference parquet path.
        right: Candidate parquet path.

    Raises:
        AssertionError: When frames differ.
    """
    frame_left = pd.read_parquet(left)
    frame_right = pd.read_parquet(right)
    pd.testing.assert_frame_equal(
        frame_left.sort_values(list(frame_left.columns)).reset_index(drop=True),
        frame_right.sort_values(list(frame_right.columns)).reset_index(drop=True),
        check_dtype=True,
    )


def scrub_manifest(payload: Mapping[str, object]) -> dict:
    """
    Remove volatile manifest keys before comparing JSON artefacts.

    Args:
        payload: Parsed ``run_manifest.json`` document.

    Returns:
        Scrubbed copy safe for structural comparison.
    """
    scrubbed = json.loads(json.dumps(payload))

    def _walk(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for key in list(node.keys()):
                child_path = f"{path}.{key}" if path else key
                if key in _VOLATILE_MANIFEST_KEYS:
                    node.pop(key)
                    continue
                _walk(node[key], child_path)
        elif isinstance(node, list):
            for item in node:
                _walk(item, path)

    _walk(scrubbed)
    return scrubbed


def assert_manifests_equal_except_volatile(left: Path, right: Path) -> None:
    """
    Compare two run manifests ignoring timestamps and run identifiers.

    Args:
        left: Reference manifest path.
        right: Candidate manifest path.
    """
    left_payload = scrub_manifest(json.loads(left.read_text(encoding="utf-8")))
    right_payload = scrub_manifest(json.loads(right.read_text(encoding="utf-8")))
    assert left_payload == right_payload


def cli_output_text(result: Result) -> str:
    """
    Combine Click stdout and stderr for message assertions.

    Args:
        result: Click CLI invocation result.

    Returns:
        Combined textual output.
    """
    stderr = result.stderr_bytes.decode("utf-8", errors="replace") if result.stderr_bytes else ""
    return f"{result.output}\n{stderr}"


def assert_cli_user_error(
    result: Result,
    *,
    must_mention: Iterable[str],
    forbid_traceback: bool = True,
) -> None:
    """
    Assert a CLI invocation failed with an actionable, non-programmer message.

    Args:
        result: Click CLI invocation result.
        must_mention: Substrings expected in the combined output (case-insensitive).
        forbid_traceback: When ``True``, reject raw Python tracebacks.

    Raises:
        AssertionError: When exit code, messaging, or traceback checks fail.
    """
    assert result.exit_code != 0, result.output
    combined = cli_output_text(result)
    if forbid_traceback:
        assert "Traceback (most recent call last)" not in combined, combined
    lowered = combined.lower()
    for fragment in must_mention:
        assert fragment.lower() in lowered, (
            f"Expected {fragment!r} in CLI output; got:\n{combined}"
        )


def parse_yaml(path: Path) -> dict:
    """
    Load one YAML file as a mapping.

    Args:
        path: YAML path.

    Returns:
        Parsed top-level mapping.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def count_label_mismatches(
    reference: Sequence[HabitatMap],
    candidate: Sequence[HabitatMap],
) -> int:
    """
    Count voxel-level disagreements between two habitat-map sequences.

    Args:
        reference: Expected habitat maps.
        candidate: Maps to compare.

    Returns:
        Total number of differing voxels across all subjects.
    """
    total = 0
    for ref_map, cand_map in zip(reference, candidate):
        total += int(np.sum(ref_map.label_array != cand_map.label_array))
    return total


def run_two_step_on_tree(data_root: Path, spec: HabitatSpec) -> StudyResult:
    """
    Execute the two-step recipe on a directory-backed cohort.

    Args:
        data_root: Synthetic cohort root.
        spec: Habitat analysis specification.

    Returns:
        Completed study result.
    """
    import habit.recipes as recipes

    cohort = load_cohort_from_tree(data_root)
    return recipes.two_step(cohort, spec, seed=SYNTHETIC_SEED)
