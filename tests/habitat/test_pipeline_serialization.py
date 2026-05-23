"""Tests for habitat pipeline save slimming helpers."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np

from habit.core.habitat_analysis.pipelines.pipeline_serialization import (
    apply_mask_metadata_to_sitk_image,
    prepare_pipeline_for_save,
    slim_mask_info_for_storage,
    strip_clustering_training_artifacts,
)


class _FakeClusterer:
    def __init__(self, n_labels: int) -> None:
        self.labels_ = np.arange(n_labels, dtype=np.int32)
        self.model = MagicMock()
        self.model.labels_ = self.labels_.copy()


def test_slim_mask_info_drops_sitk_object() -> None:
    mask_array = np.zeros((4, 5, 6), dtype=np.uint8)
    mask_img = MagicMock()
    mask_img.GetSpacing.return_value = (1.0, 1.0, 2.0)
    mask_img.GetOrigin.return_value = (0.0, 0.0, 0.0)
    mask_img.GetDirection.return_value = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    slim = slim_mask_info_for_storage({"mask": mask_img, "mask_array": mask_array})

    assert "mask" not in slim
    assert np.array_equal(slim["mask_array"], mask_array)
    assert slim["spacing"] == (1.0, 1.0, 2.0)


def test_apply_mask_metadata_uses_slim_fields() -> None:
    image = MagicMock()
    mask_info: Dict[str, Any] = {
        "mask_array": np.zeros((2, 2, 2), dtype=np.uint8),
        "spacing": (0.5, 0.5, 1.0),
        "origin": (1.0, 2.0, 3.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    }

    apply_mask_metadata_to_sitk_image(image, mask_info)

    image.SetSpacing.assert_called_once_with((0.5, 0.5, 1.0))
    image.SetOrigin.assert_called_once_with((1.0, 2.0, 3.0))
    image.SetDirection.assert_called_once_with(
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


def test_strip_clustering_training_artifacts_clears_labels() -> None:
    clusterer = _FakeClusterer(n_labels=1_000_000)
    strip_clustering_training_artifacts(clusterer)

    assert clusterer.labels_ is None
    assert clusterer.model.labels_ is None


def test_prepare_pipeline_for_save_clears_masks_and_strips_labels() -> None:
    mask_img = MagicMock()
    mask_img.GetSpacing.return_value = (1.0, 1.0, 1.0)
    mask_img.GetOrigin.return_value = (0.0, 0.0, 0.0)
    mask_img.GetDirection.return_value = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    mask_array = np.ones((3, 4, 5), dtype=np.uint8)

    clustering_service = MagicMock()
    clustering_service.supervoxel2habitat_clustering = _FakeClusterer(n_labels=500)

    group_step = MagicMock()
    group_step.clustering_model = clustering_service.supervoxel2habitat_clustering
    group_step.clustering_service = clustering_service

    pipeline = MagicMock()
    pipeline._train_checkpoint = object()
    pipeline.mask_info_cache = {
        "sub1": {"mask": mask_img, "mask_array": mask_array},
    }
    pipeline.steps = [("group_clustering", group_step)]

    prepare_pipeline_for_save(pipeline)

    assert pipeline._train_checkpoint is None
    assert pipeline.mask_info_cache == {}
    assert group_step.clustering_model.labels_ is None
