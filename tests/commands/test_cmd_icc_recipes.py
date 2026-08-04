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
"""Fast CLI wiring tests for the ``habit icc`` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

import habit.commands.cmd_icc as cmd_icc
from habit.api.analysis import ICCConfig
from habit.commands.cmd_icc import run_icc
from habit.recipes.icc import icc_analysis


def _write_session_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write one measurement-session table with the subject id as index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _icc_config_yaml(test_csv: Path, retest_csv: Path, output_json: Path) -> str:
    """Render a minimal v0.1 ICC config for two synthetic session CSVs."""
    return f"""input:
  type: files
  file_groups:
    - ["{test_csv.as_posix()}", "{retest_csv.as_posix()}"]
output:
  path: "{output_json.as_posix()}"
metrics:
  - icc2
  - icc3
debug: false
"""


@pytest.fixture
def synthetic_sessions(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Build two aligned session CSVs and return (test, retest, output_json)."""
    rng = np.random.default_rng(7)
    subject_ids = [f"S{i:03d}" for i in range(12)]
    index = pd.Index(subject_ids, name="subject")
    test_frame = pd.DataFrame(
        {
            "stable_a": rng.normal(10.0, 1.0, size=12),
            "stable_b": rng.normal(5.0, 1.0, size=12),
            "noisy": rng.normal(0.0, 1.0, size=12),
        },
        index=index,
    )
    retest_frame = pd.DataFrame(
        {
            # Stable features keep the subject signal under tiny remeasure noise.
            "stable_a": test_frame["stable_a"] + rng.normal(0.0, 0.01, size=12),
            "stable_b": test_frame["stable_b"] + rng.normal(0.0, 0.01, size=12),
            # The unstable feature is an independent draw on the retest session.
            "noisy": rng.normal(0.0, 1.0, size=12),
        },
        index=index,
    )
    test_csv = tmp_path / "test.csv"
    retest_csv = tmp_path / "retest.csv"
    _write_session_csv(test_csv, test_frame)
    _write_session_csv(retest_csv, retest_frame)
    return test_csv, retest_csv, tmp_path / "out_icc" / "icc_results.json"


def _write_config(root: Path, content: str) -> Path:
    """Write one YAML config under ``root`` and return its path."""
    path = root / "config_icc.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.cli
def test_icc_cli_dispatches_to_recipe(
    synthetic_sessions: Tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """icc loads YAML then calls the L4 recipe, not habit.core runners."""
    test_csv, retest_csv, output_json = synthetic_sessions
    config_path = _write_config(
        output_json.parent.parent,
        _icc_config_yaml(test_csv, retest_csv, output_json),
    )

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> None:
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(cmd_icc, "icc_analysis", _spy)

    run_icc(str(config_path))

    assert len(calls) == 1
    config_arg = calls[0]["args"][0]
    assert isinstance(config_arg, ICCConfig)
    assert config_arg.output.path == str(output_json)


@pytest.mark.cli
def test_icc_recipe_delegates_to_api(
    synthetic_sessions: Tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The L4 recipe forwards to habit.api.analysis.run_icc_analysis."""
    test_csv, retest_csv, output_json = synthetic_sessions
    config_path = _write_config(
        output_json.parent.parent,
        _icc_config_yaml(test_csv, retest_csv, output_json),
    )
    config = ICCConfig.from_file(str(config_path))

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        return object()

    monkeypatch.setattr("habit.api.analysis.run_icc_analysis", _spy)

    icc_analysis(config)

    assert len(calls) == 1
    assert calls[0]["args"][0] is config


@pytest.mark.cli
def test_icc_analysis_writes_result_artifacts(
    synthetic_sessions: Tuple[Path, Path, Path],
) -> None:
    """Synthetic sessions run through the recipe and emit ICC JSON + manifest."""
    test_csv, retest_csv, output_json = synthetic_sessions
    config_path = _write_config(
        output_json.parent.parent,
        _icc_config_yaml(test_csv, retest_csv, output_json),
    )
    config = ICCConfig.from_file(str(config_path))

    result = icc_analysis(config)

    assert output_json.is_file()
    assert (output_json.parent / "habit_run_manifest.json").is_file()
    assert result.artifacts["icc_result"] == output_json

    results = json.loads(output_json.read_text(encoding="utf-8"))
    group_results = results["test_vs_retest"]
    assert set(group_results) == {"stable_a", "stable_b", "noisy"}
    assert group_results["stable_a"]["ICC3"]["value"] > 0.9
    assert group_results["stable_b"]["ICC2"]["value"] > 0.9
    assert group_results["noisy"]["ICC3"]["value"] < 0.9
