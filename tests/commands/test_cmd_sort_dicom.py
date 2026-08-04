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

"""Fast CLI wiring tests for the sort-dicom command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from habit.commands.cmd_sort_dicom import run_sort_dicom


def _minimal_sort_dicom_config_yaml(data_dir: Path, out_dir: Path) -> str:
    """Render a minimal v0.1 DICOM sort config."""
    return f"""data_dir: "{data_dir.as_posix()}"
out_dir: "{out_dir.as_posix()}"
f: "subj_%n_%g_%x/%s_%d/%r_%o.dcm"
"""


@pytest.mark.cli
def test_sort_dicom_cli_dispatches_to_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sort-dicom validates YAML then calls the L4 recipe, not habit.core.run."""
    data_dir = tmp_path / "input_dicom"
    data_dir.mkdir()
    out_dir = tmp_path / "out_sort_dicom"
    config_path = tmp_path / "config_sort_dicom.yaml"
    config_path.write_text(
        _minimal_sort_dicom_config_yaml(data_dir, out_dir), encoding="utf-8"
    )

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> MagicMock:
        calls.append({"args": args, "kwargs": kwargs})
        return MagicMock()

    monkeypatch.setattr("habit.commands.cmd_sort_dicom.sort_dicom", _spy)

    run_sort_dicom(str(config_path))

    assert len(calls) == 1
    assert calls[0]["args"][0].out_dir == str(out_dir)
    assert calls[0]["kwargs"]["logger"] is not None
