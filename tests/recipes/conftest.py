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
Shared plumbing for the recipe parity tests.

The recipe tests answer one question: does the in-memory L4 path reproduce the
numbers the v0.1 CLI produced? They therefore need the same three inputs the
CLI had -- the legacy YAML, the demo cohort, and the frozen baseline -- without
any of the CLI's directory or run-mode machinery. This module holds exactly
that plumbing so each test file states only the comparison it makes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = REPO_ROOT / "tests" / "golden" / "baseline"


def demo_data_available() -> bool:
    """Return whether the untracked demo dataset is present locally."""
    return (REPO_ROOT / "demo_data" / "preprocessed" / "processed_images").is_dir()


def load_baseline(case_name: str) -> Dict[str, Any]:
    """
    Load one frozen baseline record, skipping the test when it is absent.

    Args:
        case_name: Golden case name, e.g. ``habitat_two_step``.

    Returns:
        The parsed baseline document.
    """
    path = BASELINE_DIR / f"{case_name}.json"
    if not path.is_file():
        pytest.skip(
            f"No golden baseline for '{case_name}'. "
            "Generate it locally with: python scripts/make_golden_baseline.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def spec_and_data_root(config_rel: str) -> Tuple[Any, Path]:
    """
    Translate a v0.1 YAML config into a spec plus the cohort's data root.

    Args:
        config_rel: Config path relative to the repository root.

    Returns:
        Tuple of the translated :class:`~habit.spec.specs.HabitatSpec` and the
        directory the demo images live in.
    """
    from habit.spec.legacy import LegacyConfigAdapter
    from habit.spec.specs import HabitatSpec

    config_path = REPO_ROOT / config_rel
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    translation = LegacyConfigAdapter().translate(payload, "habitat")
    spec = HabitatSpec.from_dict(translation.document["spec"])

    # v0.1 resolves relative data paths against the config file's directory.
    source = (config_path.parent / translation.document["data"]["source"]).resolve()
    return spec, source


def load_demo_cohort(spec: Any, root: Path) -> Any:
    """
    Load the demo cohort with the modalities a spec asks for.

    Args:
        spec: Translated habitat spec; its voxel feature extractor declares
            the modality order the legacy run used.
        root: Cohort root directory.

    Returns:
        The loaded :class:`~habit.contracts.subject.Cohort`.
    """
    from habit.adapters import DirectoryDataSource

    modalities: List[str] = list(spec.voxel_feature_extractor.params.get("modalities") or [])
    if root.is_file() and root.suffix.lower() in (".yaml", ".yml"):
        # The direct-pooling baseline was generated through the v0.1 manifest
        # loader. Do not infer a directory root from one manifest entry: that
        # silently includes every complete subject on disk rather than the
        # exact manifest-selected cohort used for the frozen result.
        import logging

        from habit.commands.cmd_habitat import _cohort_from_manifest

        return _cohort_from_manifest(
            root,
            modalities=tuple(modalities),
            roi=modalities[0],
            logger=logging.getLogger(__name__),
        )
    return DirectoryDataSource(root, modalities=modalities, roi=modalities[0]).load()
