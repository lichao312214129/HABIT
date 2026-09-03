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
"""CLI versus public API parity tests."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from habit.api.preprocessing import PreprocessingConfig, run_preprocess
from tests.api.helpers import assert_output_trees_equal
from habit.utils.subprocess_utils import run_capture_text


@pytest.mark.integration
def test_preprocess_cli_api_parity(
    synthetic_preprocess_dataset: tuple[Path, Dict[str, Any]],
    project_root: Path,
    tmp_path: Path,
) -> None:
    """``habit preprocess`` CLI and ``run_preprocess`` must produce identical outputs."""
    _, base_config = synthetic_preprocess_dataset

    api_config_dict = deepcopy(base_config)
    api_config_dict["out_dir"] = str(tmp_path / "out_api")
    cli_config_dict = deepcopy(base_config)
    cli_config_dict["out_dir"] = str(tmp_path / "out_cli")

    api_config_path = tmp_path / "api_preprocess.yaml"
    cli_config_path = tmp_path / "cli_preprocess.yaml"
    api_config_path.write_text(
        yaml.safe_dump(api_config_dict, sort_keys=False),
        encoding="utf-8",
    )
    cli_config_path.write_text(
        yaml.safe_dump(cli_config_dict, sort_keys=False),
        encoding="utf-8",
    )

    api_config = PreprocessingConfig.model_validate(api_config_dict)
    run_preprocess(api_config)

    result = run_capture_text(
        [
            sys.executable,
            "-m",
            "habit.cli",
            "preprocess",
            "-c",
            str(cli_config_path),
        ],
        cwd=str(project_root),
        check=False,
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")

    assert_output_trees_equal(
        Path(api_config_dict["out_dir"]),
        Path(cli_config_dict["out_dir"]),
        ignore_globs=("*.log",),
    )
