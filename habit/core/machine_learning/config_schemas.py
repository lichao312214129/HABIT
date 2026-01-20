"""
Configuration Schemas for Machine Learning Workflows
Uses Pydantic for robust validation and type safety.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, ConfigDict

class InputFileConfig(BaseModel):
    path: str
    name: str = ""
    subject_id_col: str
    label_col: str
    features: Optional[List[str]] = None
    split_col: Optional[str] = None
    pred_col: Optional[str] = None

class NormalizationConfig(BaseModel):
    method: Literal['z_score', 'min_max', 'robust', 'max_abs', 'normalizer', 'quantile', 'power'] = 'z_score'
    params: Dict[str, Any] = Field(default_factory=dict)

class FeatureSelectionMethod(BaseModel):
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)

class ModelConfig(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)

class VisualizationConfig(BaseModel):
    enabled: bool = True
    plot_types: List[str] = Field(default_factory=lambda: ['roc', 'dca', 'calibration', 'pr', 'confusion', 'shap'])
    dpi: int = 600
    format: str = "pdf"

class MLConfig(BaseModel):
    model_config = ConfigDict(extra='allow') # Allow extra fields for backward compatibility during migration

    input: List[InputFileConfig]
    output: str
    random_state: int = 42
    
    # Validation/Splitting
    split_method: Literal['random', 'stratified', 'custom'] = 'stratified'
    test_size: float = 0.3
    train_ids_file: Optional[str] = None
    test_ids_file: Optional[str] = None
    
    # K-Fold specific
    n_splits: int = 5
    stratified: bool = True
    
    # Core components
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    feature_selection_methods: List[FeatureSelectionMethod] = Field(default_factory=list)
    models: Dict[str, ModelConfig]
    
    # Flags
    is_visualize: bool = True
    is_save_model: bool = True
    
    # Visualization detail
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @validator('test_size')
    def test_size_range(cls, v):
        if not (0 < v < 1):
            raise ValueError('test_size must be between 0 and 1')
        return v

def validate_config(config_dict: Dict[str, Any]) -> MLConfig:
    """Validate raw dictionary against the schema."""
    return MLConfig(**config_dict)
