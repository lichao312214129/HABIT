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
CLI-level tests for `habit cv` command (K-Fold cross-validation).

Heavy K-fold demo run: ``tests/machine_learning/ml_kfold_cross_validation.py``.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from habit.cli import cli


class TestKFoldCLI:
    # ------------------------------------------------------------------
    # Help / meta
    # ------------------------------------------------------------------

    def test_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["cv", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_cv_or_crossvalidation(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["cv", "--help"])
        assert any(
            kw in result.output.lower()
            for kw in ("cross-validation", "k-fold", "cv", "fold")
        )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_missing_config_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["cv", "-c", "nonexistent.yaml"])
        assert result.exit_code != 0
