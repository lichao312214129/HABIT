"""
Tests for habitats results table I/O (parquet / csv).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from types import SimpleNamespace
from habit.core.habitat_analysis.services.result_publisher import (
    HabitatResultPublisher,
    canonical_csv_column_order,
)
from habit.utils.habitats_results_io import (
    find_habitats_results_file,
    habitats_results_filename,
    habitats_results_path,
    load_habitats_results,
    normalize_habitats_results_format,
    save_habitats_results,
)

pytest.importorskip("pyarrow")


@pytest.fixture
def sample_results_df() -> pd.DataFrame:
    """Small habitats-like table for round-trip tests."""
    return pd.DataFrame(
        {
            "Subject": ["sub1", "sub1", "sub2"],
            "Supervoxel": [1, 2, 1],
            "Habitats": [1, 2, 1],
            "Count": [10, 12, 8],
            "feature_a": [0.1, 0.2, 0.3],
            "feature_b": [1.1, 1.2, 1.3],
        }
    )


@pytest.mark.unit
@pytest.mark.habitat
class TestHabitatsResultsIo:
    """Round-trip and discovery helpers for habitats results files."""

    def test_default_config_format_is_parquet(self) -> None:
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

        assert (
            HabitatAnalysisConfig.model_fields["habitats_results_format"].default
            == "parquet"
        )

    def test_normalize_habitats_results_format(self) -> None:
        assert normalize_habitats_results_format("Parquet") == "parquet"
        assert normalize_habitats_results_format("csv") == "csv"
        with pytest.raises(ValueError):
            normalize_habitats_results_format("xlsx")

    def test_habitats_results_filename(self) -> None:
        assert habitats_results_filename("parquet") == "habitats.parquet"
        assert habitats_results_filename("csv") == "habitats.csv"

    def test_parquet_round_trip(
        self,
        tmp_path: Path,
        sample_results_df: pd.DataFrame,
    ) -> None:
        out_path = save_habitats_results(
            sample_results_df,
            tmp_path,
            "parquet",
        )
        assert out_path == habitats_results_path(tmp_path, "parquet")
        loaded = load_habitats_results(out_path)
        pd.testing.assert_frame_equal(loaded, sample_results_df)

    def test_csv_round_trip(
        self,
        tmp_path: Path,
        sample_results_df: pd.DataFrame,
    ) -> None:
        out_path = save_habitats_results(
            sample_results_df,
            tmp_path,
            "csv",
        )
        loaded = load_habitats_results(out_path)
        pd.testing.assert_frame_equal(loaded, sample_results_df)

    def test_find_prefers_parquet_then_csv(
        self,
        tmp_path: Path,
        sample_results_df: pd.DataFrame,
    ) -> None:
        save_habitats_results(sample_results_df, tmp_path, "parquet")
        csv_path = tmp_path / habitats_results_filename("csv")
        csv_path.write_text("Subject,Habitats\nsub1,1\n", encoding="utf-8")

        assert (
            find_habitats_results_file(tmp_path)
            == tmp_path / habitats_results_filename("parquet")
        )

        (tmp_path / habitats_results_filename("parquet")).unlink()
        assert find_habitats_results_file(tmp_path) == csv_path

    def test_load_from_directory(
        self,
        tmp_path: Path,
        sample_results_df: pd.DataFrame,
    ) -> None:
        save_habitats_results(sample_results_df, tmp_path, "parquet")
        loaded = load_habitats_results(tmp_path)
        pd.testing.assert_frame_equal(loaded, sample_results_df)


@pytest.mark.unit
@pytest.mark.habitat
class TestHabitatResultPublisherFormats:
    """Publisher should honor habitats_results_format."""

    def test_publisher_writes_parquet_by_default(
        self,
        tmp_path: Path,
        sample_results_df: pd.DataFrame,
    ) -> None:
        class _DummyWriter:
            mask_info_cache = {}

        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True)
        cfg = SimpleNamespace(
            out_dir=str(out_dir),
            habitats_results_format="parquet",
            save_images=False,
            verbose=True,
        )

        publisher = HabitatResultPublisher(
            config=cfg,
            habitat_image_writer=_DummyWriter(),  # type: ignore[arg-type]
            logger=__import__("logging").getLogger("test"),
            pipeline_getter=lambda: None,
        )
        ordered = sample_results_df.loc[:, canonical_csv_column_order(sample_results_df)]
        publisher._write_csv(ordered)

        out_file = tmp_path / "out" / habitats_results_filename("parquet")
        assert out_file.is_file()
        loaded = load_habitats_results(out_file)
        pd.testing.assert_frame_equal(loaded, ordered)
