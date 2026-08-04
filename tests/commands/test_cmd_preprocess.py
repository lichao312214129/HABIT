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

"""Fast CLI wiring tests for the preprocess command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from habit.commands.cmd_preprocess import run_preprocess


def _minimal_preprocess_config_yaml(out_dir: Path) -> str:
    """Render a minimal v0.1 preprocess config."""
    return f"""data_dir: "{(out_dir / 'input').as_posix()}"
out_dir: "{out_dir.as_posix()}"
preprocessing:
  resample:
    images: [T1]
    target_spacing: [1.0, 1.0, 1.0]
    img_mode: bilinear
"""


@pytest.mark.cli
def test_preprocess_cli_dispatches_to_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preprocess validates YAML then calls the L4 recipe, not habit.core.run."""
    out_dir = tmp_path / "out_preprocess"
    config_path = tmp_path / "config_preprocess.yaml"
    config_path.write_text(_minimal_preprocess_config_yaml(out_dir), encoding="utf-8")

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock()

    monkeypatch.setattr("habit.commands.cmd_preprocess.preprocess_images", _spy)

    run_preprocess(str(config_path))

    assert len(calls) == 1
    assert calls[0]["args"][0].out_dir == str(out_dir)
    assert calls[0]["kwargs"]["logger"] is not None
