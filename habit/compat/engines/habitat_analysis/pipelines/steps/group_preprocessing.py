# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Group-level preprocessing step for habitat analysis pipeline.

This step manages PreprocessingState internally for group-level normalization.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from ..base_pipeline import GroupLevelStep
from ...feature_preprocessing import PreprocessingState


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
