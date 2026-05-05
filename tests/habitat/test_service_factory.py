"""
Contract tests for the train/predict factories on the habitat analysis services.

These pin down the PR-4 refactor that split the previously-implicit dual
``run_mode`` behaviour of :class:`FeatureService` and :class:`ClusteringService`
into explicit ``for_train`` / ``for_predict`` factories on top of the same
single-class implementation.

Both services still accept the legacy ``__init__(config, logger)`` signature
(used by ``HabitatConfigurator``); the factories merely make caller intent
explicit and refuse to silently produce a malformed instance when the run
mode does not match.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

# These imports purposely avoid pulling in heavy optional deps (radiomics,
# SimpleITK, sklearn) — the tests below only exercise factory dispatch and
# attribute defaults, never actual feature extraction or clustering.
from habit.core.habitat_analysis.services.clustering_service import ClusteringService
from habit.core.habitat_analysis.services.feature_service import FeatureService


def _predict_config() -> Any:
    """
    Build a minimal stand-in for ``HabitatAnalysisConfig`` that the predict-mode
    constructors are happy with. Predict mode does NOT touch
    ``FeatureConstruction`` / ``HabitatsSegmention``, so we can use a
    ``SimpleNamespace`` and avoid importing the full pydantic schema.
    """
    return SimpleNamespace(
        run_mode='predict',
        verbose=False,
        FeatureConstruction=None,
        HabitatsSegmention=None,
    )


def test_feature_service_predict_mode_defaults() -> None:
    """
    Predict-mode FeatureService must declare the attribute set used elsewhere
    in the pipeline, so attribute access does not blow up before the pipeline
    pkl is injected.
    """
    cfg = _predict_config()
    service = FeatureService(cfg, logger=logging.getLogger("test"))

    assert service.voxel_method is None
    assert service.voxel_params == {}
    assert service.voxel_processing_steps == []
    assert service.has_supervoxel_config is False
    assert service.images_paths is None
    assert service.mask_paths is None
    assert service.supervoxel_file_dict is None


def test_feature_service_for_predict_factory_works() -> None:
    cfg = _predict_config()
    service = FeatureService.for_predict(cfg, logger=logging.getLogger("test"))
    assert isinstance(service, FeatureService)
    assert service.voxel_method is None


def test_feature_service_for_predict_rejects_train_config() -> None:
    cfg = _predict_config()
    cfg.run_mode = 'train'
    with pytest.raises(ValueError, match="for_predict"):
        FeatureService.for_predict(cfg, logger=logging.getLogger("test"))


def test_feature_service_for_train_rejects_predict_config() -> None:
    cfg = _predict_config()  # run_mode='predict'
    with pytest.raises(ValueError, match="for_train"):
        FeatureService.for_train(cfg, logger=logging.getLogger("test"))


def test_clustering_service_predict_mode_defaults() -> None:
    """
    Predict-mode ClusteringService must declare its public model attributes
    as ``None`` so the manager-injection whitelist can overwrite them
    cleanly when the pipeline is loaded.
    """
    cfg = _predict_config()
    service = ClusteringService(cfg, logger=logging.getLogger("test"))

    assert service.voxel2supervoxel_clustering is None
    assert service.supervoxel2habitat_clustering is None
    assert service.selection_methods is None


def test_clustering_service_for_predict_factory_works() -> None:
    cfg = _predict_config()
    service = ClusteringService.for_predict(cfg, logger=logging.getLogger("test"))
    assert isinstance(service, ClusteringService)


def test_clustering_service_for_predict_rejects_train_config() -> None:
    cfg = _predict_config()
    cfg.run_mode = 'train'
    with pytest.raises(ValueError, match="for_predict"):
        ClusteringService.for_predict(cfg, logger=logging.getLogger("test"))


def test_clustering_service_for_train_rejects_predict_config() -> None:
    cfg = _predict_config()  # run_mode='predict'
    with pytest.raises(ValueError, match="for_train"):
        ClusteringService.for_train(cfg, logger=logging.getLogger("test"))
