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
Coverage matrix: habitat feature extraction through ``habit extract``,
chained on a two-step habitat train run produced inside this module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree


@pytest.fixture(scope="module")
def extract_train_out(synthetic_tree: SyntheticTree, render_config) -> Path:
    """
    Run a compact two-step train whose maps feed feature extraction.

    Args:
        synthetic_tree: Session synthetic dataset.
        render_config: Session config renderer.

    Returns:
        Habitat train output directory holding ``*_habitats.nrrd`` maps.
    """
    rendered: RenderedConfig = render_config(
        "habitat_two_step_train.yaml", "extract_habitat_train", synthetic_tree
    )
    run_cli(CliRunner(), ["get-habitat", "-c", str(rendered.path)])
    return rendered.out_dir


@pytest.fixture(scope="module")
def extract_out(
    extract_train_out: Path, synthetic_tree: SyntheticTree, render_config
) -> Path:
    """
    Run ``habit extract`` once per module and return its output dir.

    Args:
        extract_train_out: Habitat maps from the module train run.
        synthetic_tree: Session synthetic dataset.
        render_config: Session config renderer.

    Returns:
        Feature extraction output directory.
    """
    rendered: RenderedConfig = render_config(
        "extract_features.yaml",
        "extract_features",
        synthetic_tree,
        {"@HABITAT_MAP_DIR@": extract_train_out.as_posix()},
    )
    run_cli(CliRunner(), ["extract", "-c", str(rendered.path)])
    return rendered.out_dir


@pytest.mark.integration
def test_extract_writes_feature_tables(extract_out: Path, synthetic_tree: SyntheticTree) -> None:
    """Extraction writes per-subject feature rows for every subject."""
    tables = [
        p for p in extract_out.glob("**/*")
        if p.suffix.lower() in (".csv", ".xlsx", ".parquet") and p.is_file()
    ]
    assert tables, f"no feature tables under {extract_out}"
    widest = max(tables, key=lambda p: p.stat().st_size)
    frame = (
        pd.read_parquet(widest)
        if widest.suffix == ".parquet"
        else pd.read_csv(widest)
    )
    # The writer exports the subject index as the first (unnamed) column.
    first_col = frame.columns[0]
    subjects = set(frame[first_col].astype(str))
    assert set(synthetic_tree.subjects) <= subjects, (
        f"{widest.name} subjects: {subjects}"
    )


@pytest.mark.integration
def test_extract_feature_families(extract_out: Path) -> None:
    """Multiple requested feature families appear in the exports."""
    names = " ".join(p.name.lower() for p in extract_out.glob("**/*") if p.is_file())
    hits = [
        family
        for family in ("traditional", "msi", "ith", "habitat")
        if family in names
    ]
    assert hits, f"no recognisable feature-family exports under {extract_out}"
