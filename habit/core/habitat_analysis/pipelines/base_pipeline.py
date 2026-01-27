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

from habit.utils.parallel_utils import parallel_map


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


class IndividualLevelStep(BasePipelineStep):
    """
    Marker class for individual-level pipeline steps.
    
    Individual-level steps process each subject independently.
    They should implement transform() to handle Dict[subject_id -> data].
    
    Pipeline automatically identifies these steps and handles parallelization
    at the subject level, so steps only need to focus on processing logic.
    """
    pass


class GroupLevelStep(BasePipelineStep):
    """
    Marker class for group-level pipeline steps.
    
    Group-level steps process all subjects together (e.g., population clustering,
    group-level preprocessing). They operate on aggregated data from all subjects.
    """
    pass


class HabitatPipeline:
    """
    Main pipeline for habitat analysis.
    
    Similar to sklearn Pipeline but adapted for habitat-specific workflow.
    
    Follows sklearn design philosophy:
    - fit() for training: learn parameters and save state
    - transform() for testing: use saved state to transform data
    - No mode parameter needed: state is managed via fitted_ attribute
    
    **Memory Management Strategy (Per-Subject Parallel Processing)**:
    The pipeline uses a two-stage processing strategy:
    
    1. **Individual-level stage**: Each subject independently goes through all 
       individual-level steps as an atomic operation (voxel extraction → preprocessing → 
       clustering → supervoxel extraction → aggregation). Multiple subjects are 
       processed in parallel (controlled by `processes` parameter).
       
    2. **Group-level stage**: After all subjects complete individual processing,
       group-level steps (group preprocessing → population clustering) are executed
       on the aggregated results.
    
    **Memory Control**:
    Peak memory = processes × single_subject_memory
    - processes=8: Fast, ~800MB-1.6GB (8 subjects processed simultaneously)
    - processes=4: Balanced, ~400MB-800MB (default)
    - processes=2: Memory-efficient, ~200MB-400MB
    - processes=1: Minimum memory, ~100MB-200MB
    
    **Why This Design**:
    - Clean architecture: Steps focus on single-subject logic, Pipeline handles parallelization
    - Memory bounded by `processes` parameter (max N subjects in memory at once)
    - No nested parallelization or complex coordination
    - Easy to maintain and extend
    
    Attributes:
        steps: List of (name, step) tuples
        config: Configuration object
        fitted_: bool indicating whether the pipeline has been fitted
        individual_steps: List of individual-level steps
        group_steps: List of group-level steps
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
            self.individual_steps = loaded.individual_steps
            self.group_steps = loaded.group_steps
        else:
            if not steps:
                raise ValueError("steps cannot be empty if load_from is not provided")
            self.steps = steps
            self.config = config
            self.fitted_ = False
            
            # Automatically separate individual-level and group-level steps by type
            # No hardcoded step names - clean architecture!
            self.individual_steps = [
                (name, step) for name, step in steps 
                if isinstance(step, IndividualLevelStep)
            ]
            self.group_steps = [
                (name, step) for name, step in steps 
                if isinstance(step, GroupLevelStep)
            ]
    
    def fit(
        self,
        X_train: Dict[str, Any],  # Dict of subject_id -> data
        y: Optional[Any] = None,
        **fit_params
    ) -> 'HabitatPipeline':
        """
        Fit pipeline on training data using per-subject parallel processing.
        
        Similar to fit_transform but discards the transformed output.
        Useful when you need to fit the pipeline and then transform different data.
        
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
        
        logger = logging.getLogger(__name__)
        
        # Stage 1: Individual-level processing (chunked parallel)
        if self.individual_steps:
            logger.info(
                f"Stage 1: Processing {len(X_train)} subjects through "
                f"{len(self.individual_steps)} individual-level steps..."
            )
            
            # Mark all individual-level steps as fitted (they are stateless)
            for name, step in self.individual_steps:
                step.fitted_ = True
            
            # Process subjects in parallel to control memory
            X_individual = self._process_subjects_parallel(X_train)
        else:
            X_individual = X_train
        
        # Stage 2: Group-level processing (all subjects together)
        X = X_individual
        if self.group_steps:
            logger.info(
                f"Stage 2: Fitting {len(self.group_steps)} group-level steps "
                f"on all subjects..."
            )
            for name, step in self.group_steps:
                step.fit(X, y, **fit_params)
                X = step.transform(X)
        
        self.fitted_ = True
        return self
    
    def transform(
        self,
        X_test: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Transform test data using fitted pipeline with per-subject parallel processing.
        
        Applies the same two-stage processing as fit_transform:
        1. Individual-level: Each subject through all steps (atomic operation), parallelized
        2. Group-level: All subjects processed together
        
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
        
        logger = logging.getLogger(__name__)
        
        # Stage 1: Individual-level processing (chunked parallel)
        if self.individual_steps:
            logger.info(
                f"Stage 1: Processing {len(X_test)} test subjects through "
                f"{len(self.individual_steps)} individual-level steps..."
            )
            X_individual = self._process_subjects_parallel(X_test)
        else:
            X_individual = X_test
        
        # Stage 2: Group-level processing (all subjects together)
        X_out = X_individual
        if self.group_steps:
            logger.info(
                f"Stage 2: Processing all test subjects through "
                f"{len(self.group_steps)} group-level steps..."
            )
            for name, step in self.group_steps:
                X_out = step.transform(X_out)
        
        return X_out  # Final output should be DataFrame with habitat labels
    
    def fit_transform(
        self,
        X: Dict[str, Any],
        y: Optional[Any] = None,
        **fit_params
    ) -> pd.DataFrame:
        """
        Fit and transform in one call using per-subject parallel processing.
        
        **Processing Strategy**:
        1. Individual-level stage: Each subject goes through all 5 individual-level steps
           as an atomic operation (Step1→Step2→Step3→Step4→Step5). Multiple subjects
           are processed in parallel (controlled by `processes` parameter).
        2. Group-level stage: After all subjects complete, group-level steps are executed
           on the aggregated results.
        
        This ensures peak memory = processes × single_subject_memory, bounded and predictable.
        
        Args:
            X: Input data (dict of subject_id -> data)
            y: Optional target data
            **fit_params: Additional fitting parameters
            
        Returns:
            Transformed data (DataFrame with habitat labels)
        """
        if self.fitted_:
            raise ValueError(
                "Pipeline already fitted. Use transform() for new data, "
                "or create a new pipeline instance for training."
            )
        
        logger = logging.getLogger(__name__)
        
        # Stage 1: Individual-level processing (chunked parallel)
        if self.individual_steps:
            logger.info(
                f"Stage 1: Processing {len(X)} subjects through "
                f"{len(self.individual_steps)} individual-level steps..."
            )
            
            # Mark all individual-level steps as fitted (they are stateless)
            for name, step in self.individual_steps:
                step.fitted_ = True
            
            # Process subjects in parallel to control memory
            X_individual = self._process_subjects_parallel(X)
        else:
            X_individual = X
        
        # Stage 2: Group-level processing (all subjects together)
        X_out = X_individual
        if self.group_steps:
            logger.info(
                f"Stage 2: Processing all subjects through "
                f"{len(self.group_steps)} group-level steps..."
            )
            for name, step in self.group_steps:
                X_out = step.fit_transform(X_out, y, **fit_params)
        
        self.fitted_ = True
        return X_out
    
    def _process_single_subject(self, item: Tuple[str, Any]) -> Tuple[str, Any]:
        """
        Process one subject through all individual-level steps sequentially.
        
        This is the atomic operation unit for parallelization. Steps are executed
        sequentially for this single subject without any internal parallelization.
        
        Args:
            item: Tuple of (subject_id, input_data)
            
        Returns:
            Tuple of (subject_id, output_data)
        """
        subject_id, subject_data = item
        
        # Process this subject through all individual-level steps
        data = subject_data
        try:
            for name, step in self.individual_steps:
                # Call step.transform with single-subject dict (sklearn compatible)
                single_dict = {subject_id: data}
                result_dict = step.transform(single_dict)
                # Extract data for this subject from result
                # (handle both Dict and single-item Dict)
                if subject_id in result_dict:
                    data = result_dict[subject_id]
                else:
                    # If step returns a dict but without subject_id key, use the single value
                    data = list(result_dict.values())[0] if result_dict else data
            
            return (subject_id, data)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error processing subject {subject_id} in step {name}: {e}", exc_info=True)
            raise
    
    def _process_subjects_parallel(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process subjects through all individual-level steps using parallel processing.
        
        **Architecture Design**:
        - Pipeline layer: Handles parallelization (this method)
        - Step layer: Processes single subject sequentially (no parallel logic)
        
        Strategy: Each subject goes through all individual-level steps as an atomic
        operation (step1→step2→step3→step4→step5). Multiple subjects are processed
        in parallel using parallel_map.
        
        This ensures:
        - Peak memory = processes × single_subject_memory (controlled by parallel workers)
        - Clean separation: Steps focus on logic, Pipeline handles parallelization
        - Easy to maintain: No parallel logic scattered in steps
        
        Args:
            X: Dict of subject_id -> input data
            
        Returns:
            Dict of subject_id -> processed data (ready for group-level steps)
        """
        logger = logging.getLogger(__name__)
        n_processes = getattr(self.config, 'processes', 4) if self.config else 4
        
        # Process all subjects in parallel
        n_subjects = len(X)
        logger.info(
            f"Processing {n_subjects} subjects with {n_processes} parallel workers "
            f"(Step1-5 as atomic operation for each subject)..."
        )
        
        successful_results, failed_subjects = parallel_map(
            func=self._process_single_subject,
            items=list(X.items()),
            n_processes=n_processes,
            desc="Processing subjects (individual-level pipeline)",
            logger=logger,
            show_progress=True
        )
        
        # Collect results
        # Note: parallel_map automatically unpacks (subject_id, data) tuples
        # subject_id is in proc_result.item_id, data is in proc_result.result
        results = {}
        for proc_result in successful_results:
            subject_id = proc_result.item_id
            data = proc_result.result
            results[subject_id] = data
        
        # Log failures
        if failed_subjects:
            logger.error(
                f"Failed to process {len(failed_subjects)} subject(s): "
                f"{', '.join(str(s) for s in failed_subjects)}"
            )
        
        logger.info(
            f"Individual-level processing completed: "
            f"{len(results)}/{n_subjects} subjects successful"
        )
        
        return results
    
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
