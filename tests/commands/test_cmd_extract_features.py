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

"""Fast CLI wiring tests for extract-features and radiomics commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from habit.commands.cmd_extract_features import run_extract_features
from habit.commands.cmd_radiomics import run_radiomics


def _minimal_extract_config_yaml(out_dir: Path) -> str:
    """Render a minimal v0.1 extract-features config."""
    return f"""out_dir: "{out_dir.as_posix()}"
debug: false
feature_types:
  - non_radiomics
raw_img_folder: "images"
habitats_map_folder: "habitats"
n_habitats: 2
"""


def _minimal_radiomics_config_yaml(out_dir: Path) -> str:
    """Render a minimal v0.1 radiomics config."""
    return f"""paths:
  images_folder: "{(out_dir / 'images').as_posix()}"
  out_dir: "{out_dir.as_posix()}"
"""


@pytest.mark.cli
def test_extract_features_cli_dispatches_to_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """extract-features loads YAML then calls the L4 recipe, not habit.core.run."""
    out_dir = tmp_path / "out_extract"
    config_path = tmp_path / "config_extract.yaml"
    config_path.write_text(_minimal_extract_config_yaml(out_dir), encoding="utf-8")

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock()

    monkeypatch.setattr(
        "habit.commands.cmd_extract_features.extract_habitat_features", _spy
    )

    run_extract_features(str(config_path))

    assert len(calls) == 1
    assert calls[0]["kwargs"]["plugin_configs"] == {}
    assert calls[0]["args"][0].out_dir == str(out_dir)


@pytest.mark.cli
def test_radiomics_cli_dispatches_to_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """radiomics validates YAML then calls the L4 recipe, not habit.core.run."""
    out_dir = tmp_path / "out_radiomics"
    config_path = tmp_path / "config_radiomics.yaml"
    config_path.write_text(_minimal_radiomics_config_yaml(out_dir), encoding="utf-8")

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock()

    monkeypatch.setattr("habit.commands.cmd_radiomics.traditional_radiomics", _spy)

    run_radiomics(str(config_path))

    assert len(calls) == 1
    assert calls[0]["args"][0].paths.out_dir == str(out_dir)
