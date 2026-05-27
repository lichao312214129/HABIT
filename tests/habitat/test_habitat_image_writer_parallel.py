"""Tests for lightweight spawn payloads when saving habitat images."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from habit.core.habitat_analysis.config_schemas import ResultColumns
from habit.core.habitat_analysis.services.habitat_image_writer import (
    HabitatImageWriter,
    estimate_habitat_image_worker_pickle_bytes,
)
from habit.core.habitat_analysis.services.result_publisher import HabitatResultPublisher


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


def test_habitat_image_worker_pickle_stays_small() -> None:
    """Spawn items must not embed cohort-scale tables or mask caches."""
    subject_df = pd.DataFrame(
        {
            ResultColumns.SUBJECT: ["sub1"] * 3,
            ResultColumns.SUPERVOXEL: [1, 2, 3],
            ResultColumns.HABITATS: [1, 1, 2],
            ResultColumns.COUNT: [10, 20, 30],
        }
    ).set_index(ResultColumns.SUBJECT, drop=False)
    item = (
        "sub1",
        subject_df,
        "/tmp/out",
        {"enabled": False},
        None,
        logging.INFO,
    )
    assert estimate_habitat_image_worker_pickle_bytes(item) < 16_384


def test_save_all_habitat_images_uses_module_worker(tmp_path) -> None:
    writer = _make_writer(tmp_path)
    writer.results_df = pd.DataFrame(
        {
            ResultColumns.SUBJECT: ["sub1", "sub1"],
            ResultColumns.SUPERVOXEL: [1, 2],
            ResultColumns.HABITATS: [1, 2],
            ResultColumns.COUNT: [5, 6],
        }
    )
    writer.mask_info_cache = {
        "sub1": {"mask_array": np.ones((512, 512, 200), dtype=np.uint8)},
    }

    with patch(
        "habit.core.habitat_analysis.services.habitat_image_writer.parallel_map"
    ) as parallel_map_mock:
        parallel_map_mock.return_value = ([], [])
        writer.save_all_habitat_images(failed_subjects=[])

    parallel_map_mock.assert_called_once()
    assert (
        parallel_map_mock.call_args.kwargs["func"].__name__
        == "_save_habitat_image_worker"
    )
    pickled = estimate_habitat_image_worker_pickle_bytes(
        parallel_map_mock.call_args.kwargs["items"][0]
    )
    assert pickled < 16_384


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
