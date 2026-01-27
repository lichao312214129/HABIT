"""
Group-level preprocessing step for habitat analysis pipeline.

This step manages PreprocessingState internally for group-level normalization.

Note: This module should not be run directly. Import it as part of the package:
    from habit.core.habitat_analysis.pipelines.steps import GroupPreprocessingStep
"""

from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from ..base_pipeline import GroupLevelStep
    from ...utils.preprocessing_state import PreprocessingState
except ImportError as e:
    # Provide helpful error message if imported incorrectly
    import sys
    if __name__ == "__main__":
        print("Error: This module cannot be run directly.")
        print("Please import it as part of the package:")
        print("  from habit.core.habitat_analysis.pipelines.steps import GroupPreprocessingStep")
        sys.exit(1)
    raise


class GroupPreprocessingStep(GroupLevelStep):
    """
    Group-level preprocessing using PreprocessingState.
    
    Stateful: fit() learns statistics from training data, transform() applies to new data.
    
    Note: This step manages PreprocessingState internally, no need for external Mode classes.
    
    Attributes:
        preprocessing_state: PreprocessingState instance for managing group-level statistics
        methods: List of preprocessing method configurations
        out_dir: Output directory for saving state (if needed)
        fitted_: bool indicating whether the step has been fitted
    """
    
    def __init__(self, methods: List[Dict], out_dir: str):
        """
        Initialize group preprocessing step.
        
        Args:
            methods: List of preprocessing method configurations
            out_dir: Output directory for saving state (if needed)
        """
        super().__init__()
        self.preprocessing_state = PreprocessingState()  # Create internally
        self.methods = methods
        self.out_dir = out_dir
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None, **fit_params) -> 'GroupPreprocessingStep':
        """
        Fit preprocessing state on data (learn statistics).
        
        Args:
            X: Combined supervoxel features DataFrame
            y: Optional target data (not used)
            **fit_params: Additional fitting parameters (not used)
            
        Returns:
            self
        """
        self.preprocessing_state.fit(X, self.methods)
        self.fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using fitted preprocessing state.
        
        Args:
            X: Supervoxel features DataFrame
            
        Returns:
            Preprocessed DataFrame
            
        Raises:
            ValueError: If step has not been fitted
        """
        if not self.fitted_:
            raise ValueError(
                "Must fit before transform. "
                "Either call fit() first, or load a saved pipeline."
            )
        return self.preprocessing_state.transform(X)
