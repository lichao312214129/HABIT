"""
Unit tests for habitat-analysis architecture contracts.

These are lightweight tests for the refactored seams that do not require image
I/O or a full end-to-end habitat run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from habit.core.common.configs.base import ConfigValidationError
from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
from habit.core.habitat_analysis.pipelines.habitat_subject_data import HabitatSubjectData
from habit.core.habitat_analysis.services.result_publisher import (
    HabitatResultPublisher,
    canonical_csv_column_order,
)


def _minimal_feature_construction() -> dict:
    return {
        "voxel_level": {"method": "firstorder()", "params": {}},
        "supervoxel_level": {"method": "mean_voxel_features()", "params": {}},
    }


def test_habitat_subject_data_require_methods_report_missing_field() -> None:
    state = HabitatSubjectData.empty()

    with pytest.raises(ValueError, match="ExampleStep requires 'features'"):
        state.require_features("ExampleStep")


def test_habitat_subject_data_require_methods_return_present_field() -> None:
    features = pd.DataFrame({"SUV": [1.0, 2.0]})
    state = HabitatSubjectData(features=features)

    assert state.require_features("ExampleStep") is features


def test_habitat_segmentation_new_field_is_required_in_train_mode(tmp_path: Path) -> None:
    cfg = HabitatAnalysisConfig(
        data_dir="data",
        out_dir=str(tmp_path),
        FeatureConstruction=_minimal_feature_construction(),
        HabitatSegmentation={"clustering_mode": "two_step"},
    )

    assert cfg.HabitatSegmentation.clustering_mode == "two_step"


def test_old_habitats_segmention_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError):
        HabitatAnalysisConfig(
            data_dir="data",
            out_dir=str(tmp_path),
            FeatureConstruction=_minimal_feature_construction(),
            **{"Habitats" + "Segmention": {"clustering_mode": "two_step"}},
        )


def test_canonical_csv_column_order_puts_metadata_first() -> None:
    df = pd.DataFrame(
        columns=["texture", "Habitats", "Subject", "Count", "Supervoxel", "SUV"]
    )

    assert canonical_csv_column_order(df) == [
        "Subject",
        "Supervoxel",
        "Habitats",
        "Count",
        "texture",
        "SUV",
    ]


def test_publisher_does_not_mutate_dataframe_index_or_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "texture": [0.2],
            "Subject": ["S1"],
            "Supervoxel": [1],
            "Habitats": [2],
            "Count": [10],
        }
    )
    original_index = df.index.copy()
    original_columns = list(df.columns)

    writer = MagicMock()
    writer.mask_info_cache = {}
    cfg = SimpleNamespace(
        out_dir=str(tmp_path),
        verbose=False,
        save_images=False,
        HabitatSegmentation=SimpleNamespace(clustering_mode="two_step"),
    )
    publisher = HabitatResultPublisher(
        config=cfg,
        habitat_image_writer=writer,
        logger=logging.getLogger(__name__),
        pipeline_getter=lambda: None,
    )

    publisher.publish(df, mode="train")

    assert df.index.equals(original_index)
    assert list(df.columns) == original_columns
    assert (tmp_path / "habitats.csv").exists()
    writer.save_all_habitat_images.assert_not_called()
