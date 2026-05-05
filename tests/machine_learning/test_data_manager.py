"""
Unit tests for DataManager, FeatureTableAssembler, TabularLoader, and SplitStrategy.

Uses tmp_path to create real CSV files so the full load/split path
is exercised without touching external resources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from habit.core.machine_learning.data_manager import (
    DataManager,
    FeatureTableAssembler,
    SplitStrategy,
    TabularLoader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def _make_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "subject_id": [f"S{i:03d}" for i in range(n)],
            "feature_0": rng.randn(n),
            "feature_1": rng.randn(n),
            "label": rng.randint(0, 2, n),
        }
    )


class _FakeFileConfig:
    """Minimal stand-in for InputFileConfig used by FeatureTableAssembler."""

    def __init__(self, path: str, subject_id_col: str = "subject_id", label_col: str = "label",
                 name: str = "", features: list = None):
        self.path = path
        self.subject_id_col = subject_id_col
        self.label_col = label_col
        self.name = name
        self.features = features or []
        self.add_prefix = False


class _FakeConfig:
    """Minimal stand-in for MLConfig passed to DataManager."""

    def __init__(self, csv_path: str):
        self.input = [_FakeFileConfig(path=csv_path)]
        self.split_method = "stratified"
        self.test_size = 0.3
        self.random_state = 42
        self.train_ids_file = None
        self.test_ids_file = None


# ---------------------------------------------------------------------------
# TabularLoader
# ---------------------------------------------------------------------------


class TestTabularLoader:
    def test_load_csv(self, tmp_path: Path) -> None:
        df = _make_df(20)
        p = tmp_path / "data.csv"
        _write_csv(p, df)

        loader = TabularLoader()
        result = loader.load(str(p), subject_id_col="subject_id")
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 20

    def test_load_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.parquet"
        p.write_bytes(b"fake")
        loader = TabularLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(str(p))

    def test_subject_id_stays_string_dtype(self, tmp_path: Path) -> None:
        df = _make_df(10)
        p = tmp_path / "data.csv"
        _write_csv(p, df)
        loader = TabularLoader()
        result = loader.load(str(p), subject_id_col="subject_id")
        assert result["subject_id"].dtype == object  # string columns are object dtype


# ---------------------------------------------------------------------------
# FeatureTableAssembler
# ---------------------------------------------------------------------------


class TestFeatureTableAssembler:
    def _assembler(self, tmp_path: Path):
        import logging
        loader = TabularLoader()
        logger = logging.getLogger("test")
        return FeatureTableAssembler(loader=loader, logger=logger)

    def test_assemble_single_file(self, tmp_path: Path) -> None:
        df = _make_df(30)
        p = tmp_path / "data.csv"
        _write_csv(p, df)

        assembler = self._assembler(tmp_path)
        merged = assembler.assemble([_FakeFileConfig(str(p))])
        assert merged.data.shape[0] == 30
        assert "label" in merged.data.columns
        assert merged.label_col == "label"

    def test_assemble_empty_config_raises(self, tmp_path: Path) -> None:
        assembler = self._assembler(tmp_path)
        with pytest.raises(ValueError, match="cannot be empty"):
            assembler.assemble([])

    def test_assemble_non_list_raises(self, tmp_path: Path) -> None:
        assembler = self._assembler(tmp_path)
        with pytest.raises(TypeError, match="list"):
            assembler.assemble("not_a_list")  # type: ignore[arg-type]

    def test_collision_resolution_renames_columns(self, tmp_path: Path) -> None:
        """When two files share a column name, the second file's column gets renamed."""
        df1 = pd.DataFrame({
            "subject_id": ["A", "B", "C"],
            "feature_x": [1.0, 2.0, 3.0],
            "label": [0, 1, 0],
        })
        df2 = pd.DataFrame({
            "subject_id": ["A", "B", "C"],
            "feature_x": [4.0, 5.0, 6.0],  # same column name → collision
            "label": [0, 1, 0],
        })
        p1 = tmp_path / "f1.csv"
        p2 = tmp_path / "f2.csv"
        _write_csv(p1, df1)
        _write_csv(p2, df2)

        assembler = self._assembler(tmp_path)
        merged = assembler.assemble([
            _FakeFileConfig(str(p1), name="src1"),
            _FakeFileConfig(str(p2), name="src2"),
        ])
        # feature_x should appear only once under the original name (the other is renamed)
        assert "feature_x" in merged.data.columns
        # The renamed version should also appear
        renamed_cols = [c for c in merged.data.columns if "feature_x" in c and c != "feature_x"]
        assert len(renamed_cols) >= 1


# ---------------------------------------------------------------------------
# SplitStrategy
# ---------------------------------------------------------------------------


class TestSplitStrategy:
    def _splitter(self):
        import logging
        return SplitStrategy(logger=logging.getLogger("test"))

    def _make_X_y(self, n: int = 60) -> tuple:
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(n, 4), columns=list("ABCD"))
        y = pd.Series(rng.randint(0, 2, n), name="label")
        return X, y

    def test_stratified_split_sizes(self) -> None:
        splitter = self._splitter()
        X, y = self._make_X_y(60)
        X_tr, X_te, y_tr, y_te = splitter.split(X, y, "stratified", 0.3, 42)
        assert len(X_tr) + len(X_te) == 60
        assert abs(len(X_te) / 60 - 0.3) < 0.05  # roughly 30 %

    def test_random_split_reproducible(self) -> None:
        splitter = self._splitter()
        X, y = self._make_X_y(60)
        r1 = splitter.split(X, y, "random", 0.3, 0)
        r2 = splitter.split(X, y, "random", 0.3, 0)
        assert list(r1[0].index) == list(r2[0].index)

    def test_custom_split_from_json_file(self, tmp_path: Path) -> None:
        splitter = self._splitter()
        X, y = self._make_X_y(60)

        train_ids = list(X.index[:40].astype(str))
        test_ids = list(X.index[40:].astype(str))
        train_file = tmp_path / "train.json"
        test_file = tmp_path / "test.json"
        train_file.write_text(json.dumps(train_ids))
        test_file.write_text(json.dumps(test_ids))

        X_tr, X_te, y_tr, y_te = splitter.split(
            X, y, "custom", 0.3, 0,
            train_ids_file=str(train_file),
            test_ids_file=str(test_file),
        )
        assert len(X_tr) == 40
        assert len(X_te) == 20

    def test_custom_split_missing_files_raises(self) -> None:
        splitter = self._splitter()
        X, y = self._make_X_y(20)
        with pytest.raises(ValueError, match="requires"):
            splitter.split(X, y, "custom", 0.3, 0)


# ---------------------------------------------------------------------------
# DataManager end-to-end
# ---------------------------------------------------------------------------


class TestDataManager:
    def test_load_data_populates_attributes(self, tmp_path: Path) -> None:
        df = _make_df(40)
        p = tmp_path / "data.csv"
        _write_csv(p, df)

        dm = DataManager(config=_FakeConfig(str(p)))
        dm.load_data()
        assert dm.data is not None
        assert dm.label_col == "label"
        assert dm.subject_id_col == "subject_id"

    def test_split_data_requires_load_first(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        _write_csv(p, _make_df(20))
        dm = DataManager(config=_FakeConfig(str(p)))
        with pytest.raises(ValueError, match="load_data"):
            dm.split_data()

    def test_split_data_returns_four_objects(self, tmp_path: Path) -> None:
        df = _make_df(60)
        p = tmp_path / "data.csv"
        _write_csv(p, df)

        dm = DataManager(config=_FakeConfig(str(p)))
        dm.load_data()
        X_tr, X_te, y_tr, y_te = dm.split_data()
        assert isinstance(X_tr, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert len(X_tr) + len(X_te) == len(dm.data)

    def test_load_inference_data_file_not_found(self, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        _write_csv(p, _make_df(10))
        dm = DataManager(config=_FakeConfig(str(p)))
        with pytest.raises(FileNotFoundError):
            dm.load_inference_data("nonexistent_file.csv")

    def test_load_inference_data_returns_dataframe(self, tmp_path: Path) -> None:
        df = _make_df(20)
        p = tmp_path / "data.csv"
        _write_csv(p, df)
        dm = DataManager(config=_FakeConfig(str(p)))
        result = dm.load_inference_data(str(p))
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 20
