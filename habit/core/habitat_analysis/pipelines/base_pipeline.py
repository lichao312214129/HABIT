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
        
        # Fit each step sequentially (fit then transform separately to avoid double execution)
        X = X_train
        for name, step in self.steps:
            step.fit(X, y, **fit_params)
            X = step.transform(X)
        
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
        
        Efficiently executes fit and transform in a single pass through the pipeline,
        avoiding redundant computations.
        
        Args:
            X: Input data
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            Transformed data
        """
        if self.fitted_:
            raise ValueError(
                "Pipeline already fitted. Use transform() for new data, "
                "or create a new pipeline instance for training."
            )
        
        # Fit and transform each step in one pass (more efficient than fit() + transform())
        X_out = X
        for name, step in self.steps:
            X_out = step.fit_transform(X_out, y, **fit_params)
        
        self.fitted_ = True
        return X_out
    
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


class StreamingHabitatPipeline(HabitatPipeline):
    """
    Memory-efficient streaming pipeline for individual-level processing.
    
    This pipeline processes subjects one-by-one through individual-level steps
    (voxel extraction → preprocessing → clustering → supervoxel features → aggregation)
    to minimize memory usage. Only the final supervoxel-level features are kept in memory.
    
    **Memory Comparison**:
    - Standard Pipeline: All subjects' voxel features in memory (e.g., 16GB for 100 subjects)
    - Streaming Pipeline: One subject at a time (e.g., 160MB for 1 subject)
    
    **Usage**:
    This pipeline is designed for individual-level steps (Steps 1-5 in two-step strategy).
    Population-level clustering (Step 6) should be done separately with the standard pipeline.
    
    Attributes:
        steps: List of (name, step) tuples
        batch_size: Number of subjects to process in parallel (default: 1 for minimum memory)
        fitted_: bool indicating whether the pipeline has been fitted
    """
    
    def __init__(
        self, 
        steps: List[Tuple[str, BasePipelineStep]], 
        batch_size: int = 1
    ):
        """
        Initialize streaming pipeline.
        
        Args:
            steps: List of (name, step) tuples
            batch_size: Number of subjects to process in parallel
                       - batch_size=1: Minimum memory, slower
                       - batch_size=10: Balanced memory/speed
                       - batch_size=0: Process all (same as standard pipeline)
        """
        super().__init__(steps)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def fit_transform(
        self, 
        X: Dict[str, Any], 
        y: Optional[Any] = None, 
        **fit_params
    ) -> pd.DataFrame:
        """
        Fit and transform data with hybrid streaming processing.
        
        **Strategy**:
        1. Individual-level steps (Dict input/output): Process in batches to save memory
        2. Population-level steps (DataFrame input/output): Process all data at once
        
        This approach optimizes memory for expensive individual-level operations
        (voxel extraction, clustering) while allowing population-level operations
        (group preprocessing, population clustering) to access all data.
        
        Args:
            X: Dict of subject_id -> initial data
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            Combined DataFrame with results
        """
        if self.fitted_:
            raise ValueError(
                "Pipeline already fitted. Use transform() for new data, "
                "or create a new pipeline instance for training."
            )
        
        subject_ids = list(X.keys())
        n_subjects = len(subject_ids)
        
        # Determine batch size
        if self.batch_size == 0:
            batch_size = n_subjects  # Process all at once
        else:
            batch_size = self.batch_size
        
        self.logger.info(
            f"Streaming processing: {n_subjects} subjects in batches of {batch_size}"
        )
        
        # Find the transition point from individual-level to population-level steps
        # Individual steps work with Dict[str, Any], population steps with DataFrame
        transition_idx = len(self.steps)
        for idx, (name, step) in enumerate(self.steps):
            # Check if this step expects DataFrame input (population-level)
            # Heuristic: Steps with "group" or "population" in name are population-level
            if 'group' in name.lower() or 'population' in name.lower():
                transition_idx = idx
                break
        
        # Phase 1: Process individual-level steps in batches
        all_results = []
        for batch_start in range(0, n_subjects, batch_size):
            batch_end = min(batch_start + batch_size, n_subjects)
            batch_ids = subject_ids[batch_start:batch_end]
            
            self.logger.info(
                f"Processing batch {batch_start//batch_size + 1}/"
                f"{(n_subjects + batch_size - 1)//batch_size}: "
                f"subjects {batch_start+1}-{batch_end}"
            )
            
            # Extract batch data
            batch_X = {sid: X[sid] for sid in batch_ids}
            
            # Process batch through individual-level steps
            X_batch = batch_X
            for step_idx in range(transition_idx):
                name, step = self.steps[step_idx]
                
                # Fit on first batch only
                if batch_start == 0:
                    X_batch = step.fit_transform(X_batch, y, **fit_params)
                else:
                    X_batch = step.transform(X_batch)
                
                # Log memory
                if self.logger.isEnabledFor(logging.DEBUG):
                    import sys
                    mem_mb = sys.getsizeof(X_batch) / 1024 / 1024
                    self.logger.debug(
                        f"  Step {step_idx+1} ({name}): batch memory ≈ {mem_mb:.1f} MB"
                    )
            
            # Store batch results
            if isinstance(X_batch, pd.DataFrame):
                all_results.append(X_batch)
            elif isinstance(X_batch, dict):
                # If still dict, something went wrong with transition detection
                # Fall back to standard handling
                self.logger.warning(
                    f"Individual steps output is still dict at transition point. "
                    f"Attempting to extract DataFrames."
                )
                for sid, data in X_batch.items():
                    if isinstance(data, pd.DataFrame):
                        all_results.append(data)
            
            # Free memory
            del X_batch
            del batch_X
        
        # Phase 2: Process population-level steps with all data
        if transition_idx < len(self.steps):
            self.logger.info(
                f"Processing population-level steps ({transition_idx+1}-{len(self.steps)}) "
                "with all subjects' data"
            )
            
            # Combine all batch results
            if not all_results:
                raise ValueError("No results from individual-level steps")
            
            combined_data = pd.concat(all_results, ignore_index=True)
            
            # Process through population-level steps
            X_pop = combined_data
            for step_idx in range(transition_idx, len(self.steps)):
                name, step = self.steps[step_idx]
                
                self.logger.info(f"  Running step {step_idx+1}: {name}")
                X_pop = step.fit_transform(X_pop, y, **fit_params)
            
            final_result = X_pop
        else:
            # No population-level steps, just combine batch results
            if not all_results:
                raise ValueError("No results from streaming pipeline")
            final_result = pd.concat(all_results, ignore_index=True)
        
        self.fitted_ = True
        
        self.logger.info(
            f"Streaming processing complete: {len(final_result)} total records"
        )
        
        return final_result
    
    def transform(self, X: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform new data with hybrid streaming processing.
        
        Args:
            X: Dict of subject_id -> initial data
            
        Returns:
            Combined DataFrame with results
        """
        if not self.fitted_:
            raise ValueError("Pipeline not fitted. Call fit() or fit_transform() first.")
        
        subject_ids = list(X.keys())
        n_subjects = len(subject_ids)
        
        # Determine batch size
        if self.batch_size == 0:
            batch_size = n_subjects
        else:
            batch_size = self.batch_size
        
        # Find transition point (same as fit_transform)
        transition_idx = len(self.steps)
        for idx, (name, step) in enumerate(self.steps):
            if 'group' in name.lower() or 'population' in name.lower():
                transition_idx = idx
                break
        
        # Phase 1: Individual-level steps in batches
        all_results = []
        for batch_start in range(0, n_subjects, batch_size):
            batch_end = min(batch_start + batch_size, n_subjects)
            batch_ids = subject_ids[batch_start:batch_end]
            
            batch_X = {sid: X[sid] for sid in batch_ids}
            
            # Transform through individual-level steps
            X_batch = batch_X
            for step_idx in range(transition_idx):
                name, step = self.steps[step_idx]
                X_batch = step.transform(X_batch)
            
            # Store results
            if isinstance(X_batch, pd.DataFrame):
                all_results.append(X_batch)
            elif isinstance(X_batch, dict):
                for sid, data in X_batch.items():
                    if isinstance(data, pd.DataFrame):
                        all_results.append(data)
            
            del X_batch
            del batch_X
        
        # Phase 2: Population-level steps
        if transition_idx < len(self.steps):
            if not all_results:
                raise ValueError("No results from individual-level steps")
            
            combined_data = pd.concat(all_results, ignore_index=True)
            
            X_pop = combined_data
            for step_idx in range(transition_idx, len(self.steps)):
                name, step = self.steps[step_idx]
                X_pop = step.transform(X_pop)
            
            final_result = X_pop
        else:
            if not all_results:
                raise ValueError("No results from streaming pipeline")
            final_result = pd.concat(all_results, ignore_index=True)
        
        return final_result
