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
Coverage matrix: auxiliary statistics workflows.

- ICC analysis through ``habit icc`` on the synthetic paired measurement
  tables (retest = test + small noise -> ICC must be high);
- test-retest habitat label mapping through ``habit retest``, chained on a
  two-step habitat train run of this module (identity mapping smoke test,
  mirroring the reference demo config).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree


@pytest.mark.integration
def test_icc_analysis_cli(synthetic_tree: SyntheticTree, render_config) -> None:
    """ICC(2,1)/ICC(3,1) on the paired tables are high and well-formed."""
    rendered: RenderedConfig = render_config(
        "icc_analysis.yaml",
        "icc_analysis",
        synthetic_tree,
        {
            "@ICC_TEST_CSV@": synthetic_tree.icc_test_csv.as_posix(),
            "@ICC_RETEST_CSV@": synthetic_tree.icc_retest_csv.as_posix(),
        },
    )
    run_cli(CliRunner(), ["icc", "-c", str(rendered.path)])
    result_path = rendered.out_dir / "icc_results.json"
    assert result_path.is_file(), f"missing {result_path}"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report, "empty ICC report"
    # Flatten whatever grouping the writer used and collect ICC values.
    text = result_path.read_text(encoding="utf-8").lower()
    assert "icc2" in text and "icc3" in text

    def _values(node):
        """Yield every numeric leaf of the nested report."""
        if isinstance(node, dict):
            for value in node.values():
                yield from _values(value)
        elif isinstance(node, (int, float)):
            yield float(node)

    icc_values = [v for v in _values(report) if -1.0 <= v <= 1.0]
    assert icc_values, "no ICC values in report"
    assert sum(1 for v in icc_values if v > 0.9) >= len(icc_values) / 2, (
        f"expected mostly-high ICC on test+noise pairs, got {icc_values[:8]}"
    )


@pytest.fixture(scope="module")
def retest_train_out(synthetic_tree: SyntheticTree, render_config) -> Path:
    """
    Run a compact two-step train whose outputs feed the retest mapping.

    Args:
        synthetic_tree: Session synthetic dataset.
        render_config: Session config renderer.

    Returns:
        Habitat train output directory.
    """
    rendered: RenderedConfig = render_config(
        "habitat_two_step_train.yaml", "retest_habitat_train", synthetic_tree
    )
    run_cli(CliRunner(), ["get-habitat", "-c", str(rendered.path)])
    return rendered.out_dir


@pytest.mark.integration
def test_test_retest_mapping_cli(
    retest_train_out: Path, synthetic_tree: SyntheticTree, render_config
) -> None:
    """Label mapping relabels every habitat map (identity mapping demo)."""
    habitat_table = retest_train_out / "habitats.parquet"
    if not habitat_table.is_file():
        habitat_table = next(retest_train_out.glob("habitats.*"))
    rendered: RenderedConfig = render_config(
        "test_retest.yaml",
        "test_retest",
        synthetic_tree,
        {
            "@HABITAT_TABLE@": habitat_table.as_posix(),
            "@HABITAT_MAP_DIR@": retest_train_out.as_posix(),
        },
    )
    run_cli(CliRunner(), ["retest", "-c", str(rendered.path)])
    relabelled = list(rendered.out_dir.glob("**/*.nrrd"))
    assert relabelled, (
        f"no relabelled maps under {rendered.out_dir}: "
        f"{[p.name for p in rendered.out_dir.glob('**/*')]}"
    )
