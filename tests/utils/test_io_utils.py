"""
Unit tests for habit.utils.io_utils.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from habit.utils.io_utils import load_json, save_json


class TestSaveAndLoadJson:
    def test_round_trip_dict(self, tmp_path: Path) -> None:
        data = {"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}
        path = tmp_path / "test.json"
        save_json(data, str(path))
        loaded = load_json(str(path))
        assert loaded == data

    def test_save_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        save_json({"x": 1}, str(path))
        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, IOError)):
            load_json(str(tmp_path / "nonexistent.json"))

    def test_save_list(self, tmp_path: Path) -> None:
        data = [1, 2, 3, {"a": "b"}]
        path = tmp_path / "list.json"
        save_json(data, str(path))
        assert load_json(str(path)) == data

    def test_valid_json_on_disk(self, tmp_path: Path) -> None:
        """Saved file must be valid JSON parseable by the standard library."""
        data = {"a": 1}
        path = tmp_path / "check.json"
        save_json(data, str(path))
        with open(path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
        assert parsed == data
