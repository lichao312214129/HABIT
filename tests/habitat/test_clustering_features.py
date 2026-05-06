"""
Unit tests for clustering_features extractors.

Uses synthetic per-voxel dicts so no image I/O or heavy packages are needed.
Extractors that require optional packages (pyradiomics) are guarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover
    sitk = None


# ---------------------------------------------------------------------------
# calculate_supervoxel_means (in-memory path)
# ---------------------------------------------------------------------------


class TestCalculateSupervoxelMeans:
    def test_returns_dataframe_per_supervoxel(self) -> None:
        from habit.core.habitat_analysis.clustering_features.mean_voxel_features_extractor import (
            calculate_supervoxel_means,
        )

        n_voxels = 8
        rng = np.random.RandomState(0)
        feature_df = pd.DataFrame(rng.randn(n_voxels, 2), columns=["f0", "f1"])
        raw_df = pd.DataFrame(rng.randn(n_voxels, 1), columns=["raw0"])
        labels = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        out = calculate_supervoxel_means("S1", feature_df, raw_df, labels, n_clusters_supervoxel=2)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 2

    def test_handles_single_supervoxel(self) -> None:
        from habit.core.habitat_analysis.clustering_features.mean_voxel_features_extractor import (
            calculate_supervoxel_means,
        )

        feature_df = pd.DataFrame(np.ones((5, 2), dtype=np.float32), columns=["a", "b"])
        raw_df = pd.DataFrame(np.ones((5, 1), dtype=np.float32), columns=["r"])
        labels = np.ones(5, dtype=np.int64)
        out = calculate_supervoxel_means("S2", feature_df, raw_df, labels, n_clusters_supervoxel=1)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# MeanVoxelFeaturesExtractor.extract_features (DataFrame + supervoxel map)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sitk is None, reason="SimpleITK not installed")
class TestMeanVoxelFeaturesExtractor:
    def test_extract_features_dataframe_and_map(self) -> None:
        from habit.core.habitat_analysis.clustering_features.mean_voxel_features_extractor import (
            MeanVoxelFeaturesExtractor,
        )

        shape = (2, 2, 2)
        flat_n = int(np.prod(shape))
        sv_arr = np.zeros(shape, dtype=np.int32)
        sv_arr.ravel()[:] = np.repeat([1, 2], flat_n // 2)
        sv_img = sitk.GetImageFromArray(sv_arr)
        voxel_df = pd.DataFrame(
            {
                "supervoxel": sv_arr.ravel(),
                "f1": np.linspace(0, 1, flat_n, dtype=np.float32),
            }
        )
        extractor = MeanVoxelFeaturesExtractor()
        result = extractor.extract_features(voxel_df, sv_img)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# RawFeatureExtractor
# ---------------------------------------------------------------------------


class TestRawFeatureExtractor:
    def test_instantiation(self) -> None:
        from habit.core.habitat_analysis.clustering_features.raw_feature_extractor import (
            RawFeatureExtractor,
        )

        ext = RawFeatureExtractor()
        assert ext is not None


# ---------------------------------------------------------------------------
# ConcatImageFeatureExtractor
# ---------------------------------------------------------------------------


class TestConcatImageFeatureExtractor:
    def test_instantiation(self) -> None:
        from habit.core.habitat_analysis.clustering_features.concat_feature_extractor import (
            ConcatImageFeatureExtractor,
        )

        ext = ConcatImageFeatureExtractor()
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

    def test_parse_unknown_method_still_returns_structure(self) -> None:
        """
        Parser does not validate extractor names; unknown methods still yield a tuple.
        """
        from habit.core.habitat_analysis.clustering_features.feature_expression_parser import (
            FeatureExpressionParser,
        )

        parser = FeatureExpressionParser()
        cross_method, cross_params, steps = parser.parse("totally_unknown_method()")
        assert cross_method == "totally_unknown_method"
        assert isinstance(cross_params, dict)
        assert isinstance(steps, list)


# ---------------------------------------------------------------------------
# FeatureExtractorFactory
# ---------------------------------------------------------------------------


class TestFeatureExtractorFactory:
    def test_factory_creates_mean_extractor(self) -> None:
        from habit.core.habitat_analysis.clustering_features.feature_extractor_factory import (
            FeatureExtractorFactory,
        )

        ext = FeatureExtractorFactory.create_from_name("mean_voxel_features")
        assert ext is not None

    def test_factory_raises_for_unknown_method(self) -> None:
        from habit.core.habitat_analysis.clustering_features.feature_extractor_factory import (
            FeatureExtractorFactory,
        )

        with pytest.raises(ValueError, match="Unknown feature extractor"):
            FeatureExtractorFactory.create_from_name("unknown_xyz")

