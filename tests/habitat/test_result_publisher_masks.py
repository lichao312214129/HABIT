"""Tests for on-demand mask loading in HabitatResultPublisher."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from habit.core.habitat_analysis.config_schemas import ResultColumns
from habit.core.habitat_analysis.services.result_publisher import HabitatResultPublisher


def test_populate_mask_cache_loads_from_feature_service(tmp_path) -> None:
    mask_array = np.ones((2, 2, 2), dtype=np.uint8)
    feature_service = MagicMock()
    feature_service.load_mask_info.return_value = {
        "mask_array": mask_array,
        "spacing": (1.0, 1.0, 1.0),
        "origin": (0.0, 0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }

    writer = MagicMock()
    writer.mask_info_cache = {}
    cfg = SimpleNamespace(
        out_dir=str(tmp_path),
        verbose=False,
        save_images=True,
        HabitatSegmentation=SimpleNamespace(clustering_mode="direct_pooling"),
    )
    pipeline = MagicMock()
    pipeline.mask_info_cache = {}

    publisher = HabitatResultPublisher(
        config=cfg,
        habitat_image_writer=writer,
        logger=logging.getLogger(__name__),
        pipeline_getter=lambda: pipeline,
        feature_service_getter=lambda: feature_service,
    )

    results_df = pd.DataFrame(
        {
            ResultColumns.SUBJECT: ["sub1"],
            ResultColumns.HABITATS: [1],
            ResultColumns.COUNT: [8],
        }
    )

    publisher._populate_mask_cache_for_results(results_df)

    feature_service.load_mask_info.assert_called_once_with("sub1")
    assert np.array_equal(writer.mask_info_cache["sub1"]["mask_array"], mask_array)


def test_populate_mask_cache_keeps_legacy_pipeline_cache() -> None:
    cached_array = np.zeros((2, 2, 2), dtype=np.uint8)
    feature_service = MagicMock()

    writer = MagicMock()
    writer.mask_info_cache = {}
    cfg = SimpleNamespace(
        out_dir=".",
        verbose=False,
        save_images=True,
        HabitatSegmentation=SimpleNamespace(clustering_mode="direct_pooling"),
    )
    pipeline = MagicMock()
    pipeline.mask_info_cache = {
        "sub1": {"mask_array": cached_array, "spacing": (1.0, 1.0, 1.0)},
    }

    publisher = HabitatResultPublisher(
        config=cfg,
        habitat_image_writer=writer,
        logger=logging.getLogger(__name__),
        pipeline_getter=lambda: pipeline,
        feature_service_getter=lambda: feature_service,
    )

    results_df = pd.DataFrame(
        {
            ResultColumns.SUBJECT: ["sub1"],
            ResultColumns.HABITATS: [1],
            ResultColumns.COUNT: [8],
        }
    )

    publisher._populate_mask_cache_for_results(results_df)

    feature_service.load_mask_info.assert_not_called()
    assert np.array_equal(writer.mask_info_cache["sub1"]["mask_array"], cached_array)
