"""
Base pipeline classes for habitat analysis.

This module provides the core pipeline infrastructure following sklearn design patterns.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple, Dict
import pandas as pd
import joblib
import logging


class BasePipelineStep(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for all pipeline steps.
    
    Follows sklearn interface: fit() and transform()
    All steps should inherit from this class and implement the abstract methods.
    
    Attributes:
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(self):
        """Initialize the pipeline step."""
        self.fitted_ = False
    
    @abstractmethod
    def fit(self, X: Any, y: Optional[Any] = None, **fit_params) -> 'BasePipelineStep':
        """
        Fit the step on training data.
        
        Args:
            X: Input data (can be DataFrame, dict, or other types depending on step)
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, X: Any) -> Any:
        """
        Transform data using fitted parameters.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, X: Any, y: Optional[Any] = None, **fit_params) -> Any:
        """
        Fit and transform in one call.
        
        Args:
            X: Input data
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            Transformed data
        """
        return self.fit(X, y, **fit_params).transform(X)


class HabitatPipeline:
    """
    Main pipeline for habitat analysis.
    
    Similar to sklearn Pipeline but adapted for habitat-specific workflow.
    
    Follows sklearn design philosophy:
    - fit() for training: learn parameters and save state
    - transform() for testing: use saved state to transform data
    - No mode parameter needed: state is managed via fitted_ attribute
    
    Attributes:
        steps: List of (name, step) tuples
        config: Configuration object
        fitted_: bool indicating whether the pipeline has been fitted
    """
    
    def __init__(
        self,
        steps: List[Tuple[str, BasePipelineStep]],
        config: Optional[Any] = None,
        load_from: Optional[str] = None
    ):
        """
        Initialize pipeline with steps.
        
        Args:
            steps: List of (name, step) tuples (ignored if load_from is provided)
            config: Configuration object
            load_from: Optional path to load saved pipeline state
            
        Raises:
            ValueError: If load_from is provided but file doesn't exist
        """
        if load_from:
            # Load saved pipeline
            loaded = self.load(load_from)
            self.steps = loaded.steps
            self.config = loaded.config
            self.fitted_ = loaded.fitted_
        else:
            if not steps:
                raise ValueError("steps cannot be empty if load_from is not provided")
            self.steps = steps
            self.config = config
            self.fitted_ = False
    
    def fit(
        self,
        X_train: Dict[str, Any],  # Dict of subject_id -> data
        y: Optional[Any] = None,
        **fit_params
    ) -> 'HabitatPipeline':
        """
        Fit pipeline on training data.
        
        Sequentially fits each step, passing the output of one step as input to the next.
        
        Args:
            X_train: Training data (dict of subject_id -> image/mask paths or features)
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            self
            
        Raises:
            ValueError: If pipeline is already fitted
        """
        if self.fitted_:
            raise ValueError(
                "Pipeline already fitted. Use transform() for new data, "
                "or create a new pipeline instance for training."
            )
        
        # Fit each step sequentially
        X = X_train
        for name, step in self.steps:
            X = step.fit_transform(X, y, **fit_params)
        
        self.fitted_ = True
        return self
    
    def transform(
        self,
        X_test: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Transform test data using fitted pipeline.
        
        Sequentially transforms data through each step using the fitted state.
        
        Args:
            X_test: Test data (dict of subject_id -> image/mask paths or features)
            
        Returns:
            Results DataFrame with habitat labels
            
        Raises:
            ValueError: If pipeline has not been fitted
        """
        if not self.fitted_:
            raise ValueError(
                "Pipeline must be fitted before transform(). "
                "Either call fit() first, or load a saved pipeline using "
                "HabitatPipeline.load(path) or HabitatPipeline(load_from=path)"
            )
        
        # Transform each step sequentially
        X = X_test
        for name, step in self.steps:
            X = step.transform(X)
        
        return X  # Final output should be DataFrame with habitat labels
    
    def fit_transform(
        self,
        X: Dict[str, Any],
        y: Optional[Any] = None,
        **fit_params
    ) -> pd.DataFrame:
        """
        Fit and transform in one call.
        
        Args:
            X: Input data
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            Transformed data
        """
        return self.fit(X, y, **fit_params).transform(X)
    
    def save(self, filepath: str) -> None:
        """
        Save fitted pipeline to disk.
        
        Uses joblib to serialize the entire pipeline including all fitted steps.
        
        Args:
            filepath: Path to save pipeline
            
        Raises:
            ValueError: If pipeline has not been fitted
        """
        if not self.fitted_:
            raise ValueError("Cannot save unfitted pipeline. Call fit() first.")
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'HabitatPipeline':
        """
        Load pipeline from disk.
        
        Args:
            filepath: Path to saved pipeline
            
        Returns:
            Loaded HabitatPipeline instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        return joblib.load(filepath)
