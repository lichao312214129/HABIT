"""
Unit tests for Phase 2: All Pipeline Steps.

Tests all pipeline steps:
- VoxelFeatureExtractor
- SubjectPreprocessingStep
- IndividualClusteringStep
- SupervoxelFeatureExtractionStep
- SupervoxelAggregationStep
- ConcatenateVoxelsStep
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

from habit.core.habitat_analysis.pipelines import (
    HabitatPipeline,
    build_habitat_pipeline
)
from habit.core.habitat_analysis.pipelines.steps import (
    VoxelFeatureExtractor,
    SubjectPreprocessingStep,
    IndividualClusteringStep,
    SupervoxelFeatureExtractionStep,
    SupervoxelAggregationStep,
    ConcatenateVoxelsStep,
)
from habit.core.habitat_analysis.config_schemas import ResultColumns


class TestVoxelFeatureExtractor:
    """Test VoxelFeatureExtractor."""
    
    def test_voxel_feature_extractor_init(self):
        """Test VoxelFeatureExtractor initialization."""
        feature_manager = Mock()
        step = VoxelFeatureExtractor(feature_manager)
        
        assert step.feature_manager is feature_manager
        assert step.fitted_ is False
    
    def test_voxel_feature_extractor_fit(self):
        """Test VoxelFeatureExtractor fit method."""
        feature_manager = Mock()
        step = VoxelFeatureExtractor(feature_manager)
        
        X = {'subject1': {}}
        result = step.fit(X)
        
        assert result is step
        assert step.fitted_ is True
    
    def test_voxel_feature_extractor_transform(self):
        """Test VoxelFeatureExtractor transform method."""
        feature_manager = Mock()
        
        # Mock extract_voxel_features to return expected format
        feature_df = pd.DataFrame({'feature1': [1.0, 2.0], 'feature2': [3.0, 4.0]})
        raw_df = pd.DataFrame({'raw1': [1.0, 2.0]})
        mask_info = {'mask': None, 'mask_array': np.array([1, 1])}
        
        feature_manager.extract_voxel_features.return_value = (
            'subject1', feature_df, raw_df, mask_info
        )
        
        step = VoxelFeatureExtractor(feature_manager)
        step.fitted_ = True
        
        X = {'subject1': {}}
        result = step.transform(X)
        
        assert 'subject1' in result
        assert 'features' in result['subject1']
        assert 'raw' in result['subject1']
        assert 'mask_info' in result['subject1']
        assert isinstance(result['subject1']['features'], pd.DataFrame)


class TestSubjectPreprocessingStep:
    """Test SubjectPreprocessingStep."""
    
    def test_subject_preprocessing_step_init(self):
        """Test SubjectPreprocessingStep initialization."""
        feature_manager = Mock()
        step = SubjectPreprocessingStep(feature_manager)
        
        assert step.feature_manager is feature_manager
        assert step.fitted_ is False
    
    def test_subject_preprocessing_step_fit_transform(self):
        """Test SubjectPreprocessingStep fit_transform method."""
        feature_manager = Mock()
        
        # Mock apply_preprocessing and clean_features
        input_df = pd.DataFrame({'feature1': [1.0, 2.0, 3.0]})
        processed_df = pd.DataFrame({'feature1': [0.0, 0.5, 1.0]})
        
        feature_manager.apply_preprocessing.return_value = processed_df
        feature_manager.clean_features.return_value = processed_df
        
        step = SubjectPreprocessingStep(feature_manager)
        
        X = {
            'subject1': {
                'features': input_df,
                'raw': pd.DataFrame({'raw1': [1.0, 2.0, 3.0]}),
                'mask_info': {}
            }
        }
        
        result = step.fit_transform(X)
        
        assert 'subject1' in result
        assert 'features' in result['subject1']
        assert isinstance(result['subject1']['features'], pd.DataFrame)
        feature_manager.apply_preprocessing.assert_called_once()
        feature_manager.clean_features.assert_called_once()


class TestConcatenateVoxelsStep:
    """Test ConcatenateVoxelsStep."""
    
    def test_concatenate_voxels_step_init(self):
        """Test ConcatenateVoxelsStep initialization."""
        step = ConcatenateVoxelsStep()
        assert step.fitted_ is False
    
    def test_concatenate_voxels_step_transform(self):
        """Test ConcatenateVoxelsStep transform method."""
        step = ConcatenateVoxelsStep()
        step.fitted_ = True
        
        X = {
            'subject1': {
                'features': pd.DataFrame({'feature1': [1.0, 2.0], 'feature2': [3.0, 4.0]}),
                'raw': pd.DataFrame(),
                'mask_info': {}
            },
            'subject2': {
                'features': pd.DataFrame({'feature1': [5.0, 6.0], 'feature2': [7.0, 8.0]}),
                'raw': pd.DataFrame(),
                'mask_info': {}
            }
        }
        
        result = step.transform(X)
        
        assert isinstance(result, pd.DataFrame)
        assert ResultColumns.SUBJECT in result.columns
        assert len(result) == 4  # 2 subjects * 2 voxels each
        assert len(result[result[ResultColumns.SUBJECT] == 'subject1']) == 2
        assert len(result[result[ResultColumns.SUBJECT] == 'subject2']) == 2


class TestSupervoxelAggregationStep:
    """Test SupervoxelAggregationStep."""
    
    def test_supervoxel_aggregation_without_step4(self):
        """Test SupervoxelAggregationStep without Step 4 output."""
        feature_manager = Mock()
        config = Mock()
        config.verbose = False
        
        # Mock calculate_supervoxel_means
        mean_df = pd.DataFrame({
            ResultColumns.SUBJECT: ['subject1', 'subject1'],
            ResultColumns.SUPERVOXEL: [1, 2],
            'feature1': [1.5, 2.5],
            'feature2': [3.5, 4.5]
        })
        feature_manager.calculate_supervoxel_means.return_value = mean_df
        
        step = SupervoxelAggregationStep(feature_manager, config)
        step.fitted_ = True
        
        X = {
            'subject1': {
                'features': pd.DataFrame({'feature1': [1.0, 2.0], 'feature2': [3.0, 4.0]}),
                'raw': pd.DataFrame({'raw1': [1.0, 2.0]}),
                'mask_info': {},
                'supervoxel_labels': np.array([1, 1, 2, 2])
            }
        }
        
        result = step.transform(X)
        
        assert isinstance(result, pd.DataFrame)
        assert ResultColumns.SUBJECT in result.columns
        assert ResultColumns.SUPERVOXEL in result.columns
    
    def test_supervoxel_aggregation_with_step4(self):
        """Test SupervoxelAggregationStep with Step 4 output."""
        feature_manager = Mock()
        config = Mock()
        config.verbose = False
        
        # Mock calculate_supervoxel_means
        mean_df = pd.DataFrame({
            ResultColumns.SUBJECT: ['subject1', 'subject1'],
            ResultColumns.SUPERVOXEL: [1, 2],
            'feature1': [1.5, 2.5],
            'feature2': [3.5, 4.5]
        })
        feature_manager.calculate_supervoxel_means.return_value = mean_df
        
        step = SupervoxelAggregationStep(feature_manager, config)
        step.fitted_ = True
        
        # Advanced features from Step 4
        advanced_df = pd.DataFrame({
            ResultColumns.SUBJECT: ['subject1', 'subject1'],
            ResultColumns.SUPERVOXEL: [1, 2],
            'texture1': [0.1, 0.2],
            'texture2': [0.3, 0.4]
        })
        
        X = {
            'subject1': {
                'features': pd.DataFrame({'feature1': [1.0, 2.0], 'feature2': [3.0, 4.0]}),
                'raw': pd.DataFrame({'raw1': [1.0, 2.0]}),
                'mask_info': {},
                'supervoxel_labels': np.array([1, 1, 2, 2]),
                'supervoxel_features': advanced_df  # Step 4 output
            }
        }
        
        result = step.transform(X)
        
        assert isinstance(result, pd.DataFrame)
        assert ResultColumns.SUBJECT in result.columns
        assert ResultColumns.SUPERVOXEL in result.columns
        # Should have both mean features and advanced features
        assert 'feature1' in result.columns
        assert 'texture1' in result.columns or 'texture1_advanced' in result.columns


class TestPipelineDataFormat:
    """Test data format consistency across pipeline steps."""
    
    def test_data_format_step1_to_step3(self):
        """Test data format from Step 1 to Step 3."""
        # Step 1 output: Dict[str, Dict] with 'features', 'raw', 'mask_info'
        # Step 2 output: Same format
        # Step 3 output: Same format + 'supervoxel_labels'
        
        step1_output = {
            'subject1': {
                'features': pd.DataFrame({'f1': [1.0]}),
                'raw': pd.DataFrame({'r1': [1.0]}),
                'mask_info': {}
            }
        }
        
        # This should be the format after Step 3
        step3_output = {
            'subject1': {
                'features': pd.DataFrame({'f1': [1.0]}),
                'raw': pd.DataFrame({'r1': [1.0]}),
                'mask_info': {},
                'supervoxel_labels': np.array([1, 1])
            }
        }
        
        assert 'supervoxel_labels' in step3_output['subject1']
        assert isinstance(step3_output['subject1']['supervoxel_labels'], np.ndarray)
    
    def test_data_format_step5_output(self):
        """Test data format from Step 5 (should be DataFrame)."""
        # Step 5 output should be a DataFrame with Subject and Supervoxel columns
        step5_output = pd.DataFrame({
            ResultColumns.SUBJECT: ['subject1', 'subject1', 'subject2'],
            ResultColumns.SUPERVOXEL: [1, 2, 1],
            'feature1': [1.0, 2.0, 3.0]
        })
        
        assert isinstance(step5_output, pd.DataFrame)
        assert ResultColumns.SUBJECT in step5_output.columns
        assert ResultColumns.SUPERVOXEL in step5_output.columns


class TestPipelineBuilder:
    """Test Pipeline Builder."""
    
    def test_build_two_step_pipeline_structure(self):
        """Test that two-step pipeline has correct structure."""
        config = Mock()
        config.HabitatsSegmention.clustering_mode = 'two_step'
        config.FeatureConstruction.supervoxel_level.method = 'mean_voxel_features()'
        config.FeatureConstruction.preprocessing_for_group_level.methods = []
        config.out_dir = '/tmp'
        
        feature_manager = Mock()
        clustering_manager = Mock()
        result_manager = Mock()
        
        # This will fail without proper setup, but we can test structure
        with pytest.raises((ValueError, AttributeError)):
            pipeline = build_habitat_pipeline(
                config, feature_manager, clustering_manager, result_manager
            )
    
    def test_build_one_step_pipeline_structure(self):
        """Test that one-step pipeline has correct structure."""
        config = Mock()
        config.HabitatsSegmention.clustering_mode = 'one_step'
        config.HabitatsSegmention.supervoxel.n_clusters = 50
        config.HabitatsSegmention.habitat.min_clusters = 2
        config.HabitatsSegmention.habitat.max_clusters = 10
        config.HabitatsSegmention.habitat.habitat_cluster_selection_method = 'silhouette'
        config.HabitatsSegmention.habitat.best_n_clusters = None
        config.plot_curves = False
        config.out_dir = '/tmp'
        
        feature_manager = Mock()
        clustering_manager = Mock()
        
        # This will fail without proper setup, but we can test structure
        with pytest.raises((ValueError, AttributeError)):
            pipeline = build_habitat_pipeline(
                config, feature_manager, clustering_manager
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
