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
"""Fast CLI wiring tests for the ``habit retest`` command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

import habit.commands.cmd_test_retest as cmd_test_retest
from habit.api.analysis import TestRetestConfig
from habit.commands.cmd_test_retest import run_test_retest

# Aliased on import: pytest would otherwise collect the ``test_``-prefixed
# recipe function re-exported in this module as a test case of its own.
from habit.recipes.test_retest import (
    test_retest_analysis as retest_analysis_recipe,
)

_FEATURES = ["feat_a", "feat_b"]


def _habitat_table(rows: List[Tuple[str, int, float, float]]) -> pd.DataFrame:
    """Build a long-format habitat table: one row per subject per habitat."""
    return pd.DataFrame(rows, columns=["subject_id", "habitats", *_FEATURES])


def _write_habitat_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write one habitat feature table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _swapped_session_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build test/retest tables whose physical habitats carry swapped labels.

    Physical habitat A is bright in ``feat_a`` and dark in ``feat_b``; B is
    the mirror image. The test session labels A as 1 and B as 2, the retest
    session labels them the other way round, so the correct mapping is
    ``{1: 2, 2: 1}``.
    """
    rng = np.random.default_rng(11)
    test_rows: List[Tuple[str, int, float, float]] = []
    retest_rows: List[Tuple[str, int, float, float]] = []
    for i in range(6):
        subject = f"S{i:03d}"
        jitter = rng.normal(0.0, 0.05, size=4)
        # Physical habitat A: label 1 in the test session, 2 in the retest.
        test_rows.append((subject, 1, 10.0 + jitter[0], 1.0 + jitter[1]))
        retest_rows.append((subject, 2, 10.0 + jitter[0], 1.0 + jitter[1]))
        # Physical habitat B: label 2 in the test session, 1 in the retest.
        test_rows.append((subject, 2, 1.0 + jitter[2], 10.0 + jitter[3]))
        retest_rows.append((subject, 1, 1.0 + jitter[2], 10.0 + jitter[3]))
    return _habitat_table(test_rows), _habitat_table(retest_rows)


def _write_habitat_nrrd(path: Path) -> np.ndarray:
    """Write one tiny habitat label map and return its array (labels 0/1/2)."""
    import SimpleITK as sitk

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((2, 4, 4), dtype=np.int16)
    array[0, :2, :] = 1
    array[0, 2:, :] = 2
    sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
    return array


def _retest_config_yaml(
    test_csv: Path,
    retest_csv: Path,
    input_dir: Path,
    out_dir: Path,
) -> str:
    """Render a minimal v0.1 test-retest config for the synthetic tables."""
    return f"""test_habitat_table: "{test_csv.as_posix()}"
retest_habitat_table: "{retest_csv.as_posix()}"
features: [feat_a, feat_b]
similarity_method: pearson
input_dir: "{input_dir.as_posix()}"
out_dir: "{out_dir.as_posix()}"
processes: 1
debug: false
"""


@pytest.fixture
def synthetic_retest(tmp_path: Path) -> Dict[str, Path]:
    """Build swapped-label tables plus one habitat NRRD; return all paths."""
    test_frame, retest_frame = _swapped_session_tables()
    test_csv = tmp_path / "test_habitats.csv"
    retest_csv = tmp_path / "retest_habitats.csv"
    _write_habitat_csv(test_csv, test_frame)
    _write_habitat_csv(retest_csv, retest_frame)
    input_dir = tmp_path / "retest_maps"
    out_dir = tmp_path / "remapped"
    out_dir.mkdir(parents=True)
    _write_habitat_nrrd(input_dir / "subj001_habitats.nrrd")
    return {
        "test_csv": test_csv,
        "retest_csv": retest_csv,
        "input_dir": input_dir,
        "out_dir": out_dir,
    }


def _write_config(root: Path, content: str) -> Path:
    """Write one YAML config under ``root`` and return its path."""
    path = root / "config_test_retest.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.cli
def test_retest_cli_dispatches_to_recipe(
    synthetic_retest: Dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retest loads YAML then calls the L4 recipe, not habit.core mappers."""
    config_path = _write_config(
        synthetic_retest["out_dir"].parent,
        _retest_config_yaml(
            synthetic_retest["test_csv"],
            synthetic_retest["retest_csv"],
            synthetic_retest["input_dir"],
            synthetic_retest["out_dir"],
        ),
    )

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(data={1: 2, 2: 1})

    monkeypatch.setattr(cmd_test_retest, "test_retest_analysis", _spy)

    run_test_retest(str(config_path))

    assert len(calls) == 1
    config_arg = calls[0]["args"][0]
    assert isinstance(config_arg, TestRetestConfig)
    assert config_arg.out_dir == str(synthetic_retest["out_dir"])


@pytest.mark.cli
def test_retest_recipe_delegates_to_api(
    synthetic_retest: Dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The L4 recipe forwards to habit.api.analysis.run_test_retest_analysis."""
    config_path = _write_config(
        synthetic_retest["out_dir"].parent,
        _retest_config_yaml(
            synthetic_retest["test_csv"],
            synthetic_retest["retest_csv"],
            synthetic_retest["input_dir"],
            synthetic_retest["out_dir"],
        ),
    )
    config = TestRetestConfig.from_file(str(config_path))

    calls: List[Dict[str, Any]] = []

    def _spy(*args: Any, **kwargs: Any) -> object:
        calls.append({"args": args, "kwargs": kwargs})
        return object()

    monkeypatch.setattr("habit.api.analysis.run_test_retest_analysis", _spy)

    retest_analysis_recipe(config)

    assert len(calls) == 1
    assert calls[0]["args"][0] is config
    assert calls[0]["kwargs"]["logger"] is None


@pytest.mark.cli
def test_retest_analysis_relabels_habitat_map(
    synthetic_retest: Dict[str, Path],
) -> None:
    """The recipe recovers the swapped labels and rewrites the NRRD map."""
    import SimpleITK as sitk

    config_path = _write_config(
        synthetic_retest["out_dir"].parent,
        _retest_config_yaml(
            synthetic_retest["test_csv"],
            synthetic_retest["retest_csv"],
            synthetic_retest["input_dir"],
            synthetic_retest["out_dir"],
        ),
    )
    config = TestRetestConfig.from_file(str(config_path))

    result = retest_analysis_recipe(config)

    assert result.data == {1: 2, 2: 1}
    remapped_path = synthetic_retest["out_dir"] / "subj001_habitats_remapped.nrrd"
    assert remapped_path.is_file()
    assert (synthetic_retest["out_dir"] / "habit_run_manifest.json").is_file()

    remapped = sitk.GetArrayFromImage(sitk.ReadImage(str(remapped_path)))
    original = sitk.GetArrayFromImage(
        sitk.ReadImage(str(synthetic_retest["input_dir"] / "subj001_habitats.nrrd"))
    )
    # Every voxel keeps its geometry; labels 1 and 2 trade places.
    assert (remapped[original == 0] == 0).all()
    assert (remapped[original == 1] == 2).all()
    assert (remapped[original == 2] == 1).all()


#: YAML payloads written in each encoding the retest config reader supports,
#: paired with the object a correct read must produce. Non-ASCII content is the
#: whole point: an ASCII-only fixture cannot distinguish a right codec from a
#: wrong one.
_ENCODING_FIXTURES: Tuple[Tuple[str, str, Dict[str, Any]], ...] = (
    ("plain_ascii.yaml", "ascii", {"name": "plain", "values": [1]}),
    ("utf8.yaml", "utf-8", {"name": "测试", "note": "病灶区域"}),
    ("utf8_bom.yaml", "utf-8-sig", {"name": "测试", "note": "病灶区域"}),
    ("gbk.yaml", "gbk", {"name": "测试", "note": "病灶区域"}),
)


def _write_encoded_yaml(directory: Path, name: str, encoding: str, obj: Any) -> Path:
    """
    Write ``obj`` as YAML using an explicit byte encoding.

    ``yaml.safe_dump`` is deliberately not used: it escapes non-ASCII by
    default, which would produce a pure-ASCII file and defeat the test.

    Args:
        directory: Directory to write into.
        name: File name.
        encoding: Codec used to encode the text.
        obj: Mapping whose keys/values are rendered as ``key: value`` lines.

    Returns:
        Path: The written file.
    """
    lines = []
    for key, value in obj.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    path = directory / name
    path.write_bytes(("\n".join(lines) + "\n").encode(encoding))
    return path


@pytest.mark.unit
def test_retest_config_reader_decodes_every_supported_encoding(
    tmp_path: Path,
) -> None:
    """
    The encoding probe must decode correctly without ``chardet`` installed.

    ``chardet`` was a REQUIRED dependency purely for this probe until 1.1.0 and
    is now declared nowhere, so the no-chardet path is the one users get. It
    must still round-trip every codec the reader's candidate list covers, and
    it must not depend on the machine's locale: a locale-derived first guess
    reports ``cp936`` on a Chinese Windows install, and cp936 decodes most
    UTF-8 byte sequences without raising, which would silently yield mojibake
    for the commonest case instead of an error the retry loop could catch.
    """
    import locale
    import sys
    from importlib.abc import MetaPathFinder

    class _NoChardet(MetaPathFinder):
        """Hide ``chardet`` so the stdlib-only code path is what runs."""

        def find_spec(
            self,
            fullname: str,
            path: Any = None,
            target: Any = None,
        ) -> None:
            if fullname.split(".")[0] == "chardet":
                raise ModuleNotFoundError("hidden by test", name=fullname)
            return None

    fixtures = [
        (_write_encoded_yaml(tmp_path, name, encoding, expected), expected)
        for name, encoding, expected in _ENCODING_FIXTURES
    ]

    finder = _NoChardet()
    sys.meta_path.insert(0, finder)
    real_getpreferredencoding = locale.getpreferredencoding
    try:
        from habit.compat.test_retest_mapper import read_file_with_encoding

        # Both a UTF-8 and a legacy Chinese locale must give the same answers.
        for reported_locale in ("UTF-8", "cp936"):
            locale.getpreferredencoding = (  # type: ignore[assignment]
                lambda do_setlocale=True, _enc=reported_locale: _enc
            )
            for path, expected in fixtures:
                assert (
                    read_file_with_encoding(str(path)) == expected
                ), f"{path.name} misdecoded with locale {reported_locale}"
    finally:
        locale.getpreferredencoding = real_getpreferredencoding  # type: ignore[assignment]
        sys.meta_path.remove(finder)
