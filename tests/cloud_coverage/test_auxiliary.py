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
  tables (retest = test + small noise -> ICC must be high).
"""

from __future__ import annotations

import json

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
