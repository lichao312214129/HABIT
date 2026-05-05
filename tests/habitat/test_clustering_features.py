"""
Unit tests for clustering_features extractors.

Uses synthetic per-voxel dicts so no image I/O or heavy packages are needed.
Extractors that require optional packages (pyradiomics) are guarded.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest


def _make_subject_voxels(n_voxels: int = 50, n_features: int = 4, seed: int = 0) -> Dict[str, Any]:
    """Simulate a subject's voxel-level data as a dict of arrays."""
    rng = np.random.RandomState(seed)
    return {
        "voxel_features": rng.randn(n_voxels, n_features).astype(np.float32),
        "cluster_labels": np.repeat([0, 1], [n_voxels // 2, n_voxels - n_voxels // 2]),
    }


# ---------------------------------------------------------------------------
# MeanVoxelFeaturesExtractor
# ---------------------------------------------------------------------------


class TestMeanVoxelFeaturesExtractor:
    def test_extract_returns_dict_per_cluster(self) -> None:
        from habit.core.habitat_analysis.clustering_features.mean_voxel_features_extractor import (
            MeanVoxelFeaturesExtractor,
        )

        extractor = MeanVoxelFeaturesExtractor(params={})
        subject_data = _make_subject_voxels()
        result = extractor.extract(subject_data)
        # Result must be a dict (cluster_id -> feature vector or dict)
        assert isinstance(result, dict)

    def test_extract_handles_single_cluster(self) -> None:
        from habit.core.habitat_analysis.clustering_features.mean_voxel_features_extractor import (
            MeanVoxelFeaturesExtractor,
        )

        extractor = MeanVoxelFeaturesExtractor(params={})
        data = {
            "voxel_features": np.ones((30, 4), dtype=np.float32),
            "cluster_labels": np.zeros(30, dtype=int),
        }
        result = extractor.extract(data)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# RawFeatureExtractor
# ---------------------------------------------------------------------------


class TestRawFeatureExtractor:
    def test_instantiation(self) -> None:
        from habit.core.habitat_analysis.clustering_features.raw_feature_extractor import (
            RawFeatureExtractor,
        )

        ext = RawFeatureExtractor(params={})
        assert ext is not None


# ---------------------------------------------------------------------------
# ConcatFeatureExtractor
# ---------------------------------------------------------------------------


class TestConcatFeatureExtractor:
    def test_instantiation(self) -> None:
        from habit.core.habitat_analysis.clustering_features.concat_feature_extractor import (
            ConcatFeatureExtractor,
        )

        ext = ConcatFeatureExtractor(params={})
        assert ext is not None


# ---------------------------------------------------------------------------
# FeatureExpressionParser
# ---------------------------------------------------------------------------


class TestFeatureExpressionParser:
    def test_parse_mean_expression(self) -> None:
        from habit.core.habitat_analysis.clustering_features.feature_expression_parser import (
            FeatureExpressionParser,
        )

        parser = FeatureExpressionParser()
        parsed = parser.parse("mean_voxel_features()")
        assert parsed is not None

    def test_parse_supervoxel_radiomics(self) -> None:
        from habit.core.habitat_analysis.clustering_features.feature_expression_parser import (
            FeatureExpressionParser,
        )

        parser = FeatureExpressionParser()
        parsed = parser.parse("supervoxel_radiomics()")
        assert parsed is not None

    def test_parse_unknown_raises(self) -> None:
        from habit.core.habitat_analysis.clustering_features.feature_expression_parser import (
            FeatureExpressionParser,
        )

        parser = FeatureExpressionParser()
        with pytest.raises((ValueError, KeyError)):
            parser.parse("totally_unknown_method()")


# ---------------------------------------------------------------------------
# FeatureExtractorFactory
# ---------------------------------------------------------------------------


class TestFeatureExtractorFactory:
    def test_factory_creates_mean_extractor(self) -> None:
        from habit.core.habitat_analysis.clustering_features.feature_extractor_factory import (
            FeatureExtractorFactory,
        )

        ext = FeatureExtractorFactory.create("mean_voxel_features()", params={})
        assert ext is not None

    def test_factory_raises_for_unknown_method(self) -> None:
        from habit.core.habitat_analysis.clustering_features.feature_extractor_factory import (
            FeatureExtractorFactory,
        )

        with pytest.raises((ValueError, KeyError)):
            FeatureExtractorFactory.create("unknown_xyz()", params={})
