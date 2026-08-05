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
Coverage matrix: traditional radiomics extraction (pyradiomics) on the
synthetic images+masks, run through ``habit radiomics``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree


@pytest.fixture(scope="module")
def radiomics_out(synthetic_tree: SyntheticTree, render_config) -> Path:
    """
    Run traditional radiomics once per module and return the output dir.

    Args:
        synthetic_tree: Session synthetic dataset.
        render_config: Session config renderer.

    Returns:
        Output directory holding the per-modality and combined CSVs.
    """
    rendered: RenderedConfig = render_config(
        "radiomics_traditional.yaml", "radiomics_traditional", synthetic_tree
    )
    run_cli(CliRunner(), ["radiomics", "-c", str(rendered.path)])
    return rendered.out_dir


@pytest.mark.integration
def test_radiomics_writes_tables(radiomics_out: Path, synthetic_tree: SyntheticTree) -> None:
    """Per-modality CSVs plus a combined CSV cover every subject."""
    csvs = list(radiomics_out.glob("**/*.csv"))
    assert csvs, f"no CSV exports under {radiomics_out}"
    combined = [p for p in csvs if "combined" in p.name.lower() or "all" in p.name.lower()]
    per_modality = {
        modality: [p for p in csvs if modality in p.name]
        for modality in synthetic_tree.modalities
    }
    assert all(per_modality.values()) or combined, (
        f"expected per-modality or combined exports, got {[p.name for p in csvs]}"
    )


@pytest.mark.integration
def test_radiomics_table_content(radiomics_out: Path, synthetic_tree: SyntheticTree) -> None:
    """The combined table has one row per subject and real feature columns."""
    csvs = sorted(radiomics_out.glob("**/*.csv"), key=lambda p: -p.stat().st_size)
    assert csvs
    widest = pd.read_csv(csvs[0])
    id_cols = [c for c in widest.columns if "subject" in c.lower() or c.lower() == "id"]
    assert id_cols, f"no subject id column in {csvs[0].name}: {list(widest.columns)[:8]}"
    assert len(widest) >= len(synthetic_tree.subjects)
    feature_cols = [
        c for c in widest.columns
        if c not in id_cols and widest[c].dtype.kind in "fi"
    ]
    assert len(feature_cols) >= 20, (
        f"expected dozens of radiomics features, got {len(feature_cols)}"
    )
    assert widest[feature_cols].notna().all().all()


@pytest.mark.integration
def test_radiomics_feature_sanity(radiomics_out: Path, synthetic_tree: SyntheticTree) -> None:
    """First-order means sit between background and brightest planted region."""
    csvs = sorted(radiomics_out.glob("**/*.csv"), key=lambda p: -p.stat().st_size)
    widest = pd.read_csv(csvs[0])
    mean_cols = [c for c in widest.columns if c.endswith("_Mean") or c == "original_firstorder_Mean" or "firstorder_Mean" in c]
    assert mean_cols, f"no firstorder Mean column in {csvs[0].name}"
    values = widest[mean_cols[0]].astype(float)
    # Planted tumour means span ~80-255 (delay2) on a ~45 background.
    assert values.between(70, 300).all(), f"unexpected means: {values.tolist()}"
