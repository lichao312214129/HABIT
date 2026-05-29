"""Tests for unified habitat image batch export."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from habit.core.habitat_analysis.config_schemas import ResultColumns
from habit.core.habitat_analysis.services.habitat_image_writer import (
    HabitatImageWriter,
    _save_habitat_from_supervoxel_mapping_worker,
)
from habit.core.habitat_analysis.services.result_publisher import HabitatResultPublisher
from habit.utils.parallel_utils import default_thread_worker_count


def _make_writer(tmp_path) -> HabitatImageWriter:
    cfg = SimpleNamespace(
        out_dir=str(tmp_path),
        verbose=False,
        processes=2,
        oom_backoff=False,
        HabitatSegmentation=SimpleNamespace(
            postprocess_habitat=SimpleNamespace(
                model_dump=lambda: {"enabled": False},
            ),
        ),
    )
    return HabitatImageWriter(config=cfg, logger=logging.getLogger(__name__))


def test_save_all_habitat_images_uses_thread_map_for_two_step(tmp_path) -> None:
    writer = _make_writer(tmp_path)
    writer.results_df = pd.DataFrame(
        {
            ResultColumns.SUBJECT: ["sub1", "sub1"],
            ResultColumns.SUPERVOXEL: [1, 2],
            ResultColumns.HABITATS: [1, 2],
            ResultColumns.COUNT: [5, 6],
        }
    )

    with patch(
        "habit.core.habitat_analysis.services.habitat_image_writer.thread_map"
    ) as thread_map_mock:
        thread_map_mock.return_value = ([], [])
        writer.save_all_habitat_images(failed_subjects=[])

    thread_map_mock.assert_called_once()
    assert (
        thread_map_mock.call_args.kwargs["func"].__name__
        == "_save_habitat_from_supervoxel_mapping_worker"
    )
    assert thread_map_mock.call_args.kwargs["max_workers"] == default_thread_worker_count()


def test_save_all_habitat_images_uses_voxel_worker_for_pooling(tmp_path) -> None:
    writer = _make_writer(tmp_path)
    writer.results_df = pd.DataFrame(
        {
            ResultColumns.SUBJECT: ["sub1", "sub1"],
            ResultColumns.HABITATS: [1, 2],
        }
    )
    writer.mask_info_cache = {
        "sub1": {"mask_array": np.ones((4, 4, 2), dtype=np.uint8)},
    }

    with patch(
        "habit.core.habitat_analysis.services.habitat_image_writer.thread_map"
    ) as thread_map_mock:
        thread_map_mock.return_value = ([], [])
        writer.save_all_habitat_images(failed_subjects=[])

    thread_map_mock.assert_called_once()
    assert (
        thread_map_mock.call_args.kwargs["func"].__name__
        == "_save_habitat_from_voxel_labels_worker"
    )


def test_two_step_publish_skips_mask_cache_load(tmp_path) -> None:
    feature_service = MagicMock()
    writer = MagicMock()
    writer.mask_info_cache = {}
    cfg = SimpleNamespace(
        out_dir=str(tmp_path),
        verbose=False,
        save_images=True,
        HabitatSegmentation=SimpleNamespace(clustering_mode="two_step"),
    )

    publisher = HabitatResultPublisher(
        config=cfg,
        habitat_image_writer=writer,
        logger=logging.getLogger(__name__),
        pipeline_getter=lambda: MagicMock(mask_info_cache={}),
        feature_service_getter=lambda: feature_service,
    )

    results_df = pd.DataFrame(
        {
            ResultColumns.SUBJECT: ["sub1"],
            ResultColumns.SUPERVOXEL: [1],
            ResultColumns.HABITATS: [1],
            ResultColumns.COUNT: [8],
        }
    )

    with patch.object(publisher, "_populate_mask_cache_for_results") as populate_mock:
        publisher._write_images_if_enabled(results_df)

    populate_mock.assert_not_called()
    feature_service.load_mask_info.assert_not_called()
    assert writer.mask_info_cache == {}
    writer.save_all_habitat_images.assert_called_once()
