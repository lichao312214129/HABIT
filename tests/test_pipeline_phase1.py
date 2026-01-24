"""
Unit tests for Phase 1: Basic Pipeline Framework.

Tests the core pipeline infrastructure:
- BasePipelineStep
- HabitatPipeline
- GroupPreprocessingStep
- PopulationClusteringStep
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

from habit.core.habitat_analysis.pipelines import (
    BasePipelineStep,
    HabitatPipeline,
    build_habitat_pipeline
)
from habit.core.habitat_analysis.pipelines.steps import (
    GroupPreprocessingStep,
    PopulationClusteringStep
)


class TestBasePipelineStep:
    """Test BasePipelineStep abstract class."""
    
    def test_base_pipeline_step_is_abstract(self):
        """Test that BasePipelineStep cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePipelineStep()


class TestGroupPreprocessingStep:
    """Test GroupPreprocessingStep."""
    
    def test_group_preprocessing_step_init(self):
        """Test GroupPreprocessingStep initialization."""
        methods = [{'method': 'zscore', 'global_normalize': False}]
        step = GroupPreprocessingStep(methods=methods, out_dir='/tmp')
        
        assert step.methods == methods
        assert step.out_dir == '/tmp'
        assert step.fitted_ is False
        assert step.preprocessing_state is not None
    
    def test_group_preprocessing_step_fit(self):
        """Test GroupPreprocessingStep fit method."""
        methods = [{'method': 'zscore', 'global_normalize': False}]
        step = GroupPreprocessingStep(methods=methods, out_dir='/tmp')
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        result = step.fit(X)
        
        assert result is step
        assert step.fitted_ is True
        assert step.preprocessing_state.means is not None
    
    def test_group_preprocessing_step_transform_without_fit(self):
        """Test that transform raises error if not fitted."""
        methods = [{'method': 'zscore', 'global_normalize': False}]
        step = GroupPreprocessingStep(methods=methods, out_dir='/tmp')
        
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0]
        })
        
        with pytest.raises(ValueError, match="Must fit before transform"):
            step.transform(X)
    
    def test_group_preprocessing_step_fit_transform(self):
        """Test GroupPreprocessingStep fit_transform method."""
        methods = [{'method': 'zscore', 'global_normalize': False}]
        step = GroupPreprocessingStep(methods=methods, out_dir='/tmp')
        
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        result = step.fit_transform(X)
        
        assert isinstance(result, pd.DataFrame)
        assert step.fitted_ is True
        assert result.shape == X.shape


class TestHabitatPipeline:
    """Test HabitatPipeline."""
    
    def test_habitat_pipeline_init(self):
        """Test HabitatPipeline initialization."""
        steps = []
        pipeline = HabitatPipeline(steps=steps, config=None)
        
        assert pipeline.steps == steps
        assert pipeline.config is None
        assert pipeline.fitted_ is False
    
    def test_habitat_pipeline_init_empty_steps(self):
        """Test that empty steps raises error."""
        with pytest.raises(ValueError, match="steps cannot be empty"):
            HabitatPipeline(steps=[], config=None)
    
    def test_habitat_pipeline_fit(self):
        """Test HabitatPipeline fit method."""
        methods = [{'method': 'zscore', 'global_normalize': False}]
        step = GroupPreprocessingStep(methods=methods, out_dir='/tmp')
        
        steps = [('preprocessing', step)]
        pipeline = HabitatPipeline(steps=steps, config=None)
        
        # Create sample data (DataFrame for group preprocessing)
        # Note: In full pipeline, earlier steps would convert Dict to DataFrame
        # For Phase 1 testing, we directly use DataFrame as input
        X_train = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        # Note: This test assumes GroupPreprocessingStep accepts DataFrame
        # In full implementation, earlier steps would handle Dict -> DataFrame conversion
        result = pipeline.fit(X_train)
        
        assert result is pipeline
        assert pipeline.fitted_ is True
    
    def test_habitat_pipeline_transform_without_fit(self):
        """Test that transform raises error if not fitted."""
        methods = [{'method': 'zscore', 'global_normalize': False}]
        step = GroupPreprocessingStep(methods=methods, out_dir='/tmp')
        
        steps = [('preprocessing', step)]
        pipeline = HabitatPipeline(steps=steps, config=None)
        
        # Note: In full pipeline, this would be Dict[str, Any]
        # For Phase 1 testing, we use DataFrame directly
        X_test = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0]
        })
        
        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.transform(X_test)
    
    def test_habitat_pipeline_fit_twice(self):
        """Test that fitting twice raises error."""
        methods = [{'method': 'zscore', 'global_normalize': False}]
        step = GroupPreprocessingStep(methods=methods, out_dir='/tmp')
        
        steps = [('preprocessing', step)]
        pipeline = HabitatPipeline(steps=steps, config=None)
        
        # Note: In full pipeline, this would be Dict[str, Any]
        X_train = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        pipeline.fit(X_train)
        
        with pytest.raises(ValueError, match="Pipeline already fitted"):
            pipeline.fit(X_train)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
