"""
Configuration Schemas for Machine Learning Workflows
Uses Pydantic for robust validation and type safety.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, ConfigDict

from habit.core.common.config_base import BaseConfig

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

class ComparisonFileConfig(BaseModel):
    path: str
    model_name: str
    subject_id_col: str
    label_col: str
    prob_col: str
    pred_col: Optional[str] = None
    split_col: Optional[str] = None

class MergedDataConfig(BaseModel):
    enabled: bool = True
    save_name: str = "combined_predictions.csv"

class SplitConfig(BaseModel):
    enabled: bool = False

class VisualizationItemConfig(BaseModel):
    enabled: bool = True
    save_name: Optional[str] = None
    title: Optional[str] = None
    n_bins: Optional[int] = None

class ComparisonVisualizationConfig(BaseModel):
    roc: VisualizationItemConfig = Field(
        default_factory=lambda: VisualizationItemConfig(
            enabled=True, save_name="roc_curves.pdf", title="ROC Curves"
        )
    )
    dca: VisualizationItemConfig = Field(
        default_factory=lambda: VisualizationItemConfig(
            enabled=True, save_name="decision_curves.pdf", title="Decision Curves"
        )
    )
    calibration: VisualizationItemConfig = Field(
        default_factory=lambda: VisualizationItemConfig(
            enabled=True,
            save_name="calibration_curves.pdf",
            title="Calibration Curves",
            n_bins=10
        )
    )
    pr_curve: VisualizationItemConfig = Field(
        default_factory=lambda: VisualizationItemConfig(
            enabled=True, save_name="precision_recall_curves.pdf", title="Precision-Recall Curves"
        )
    )

class DelongTestConfig(BaseModel):
    enabled: bool = True
    save_name: str = "delong_results.json"

class BasicMetricsConfig(BaseModel):
    enabled: bool = False

class YoudenMetricsConfig(BaseModel):
    enabled: bool = False

class TargetMetricsConfig(BaseModel):
    enabled: bool = False
    targets: Dict[str, float] = Field(default_factory=dict)

    @validator('targets')
    def target_values_range(cls, v):
        for key, value in v.items():
            if not (0 < value < 1):
                raise ValueError(f"Target '{key}' must be between 0 and 1")
        return v

class MetricsConfig(BaseModel):
    basic_metrics: BasicMetricsConfig = Field(default_factory=BasicMetricsConfig)
    youden_metrics: YoudenMetricsConfig = Field(default_factory=YoudenMetricsConfig)
    target_metrics: TargetMetricsConfig = Field(default_factory=TargetMetricsConfig)

class ModelComparisonConfig(BaseConfig):
    model_config = ConfigDict(extra='allow')

    output_dir: str
    files_config: List[ComparisonFileConfig] = Field(default_factory=list)
    merged_data: MergedDataConfig = Field(default_factory=MergedDataConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    visualization: ComparisonVisualizationConfig = Field(default_factory=ComparisonVisualizationConfig)
    delong_test: DelongTestConfig = Field(default_factory=DelongTestConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @validator('output_dir')
    def output_dir_required(cls, v):
        if not v or not str(v).strip():
            raise ValueError("output_dir is required and cannot be empty")
        return v

class MLConfig(BaseConfig):
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

class PredictionConfig(BaseConfig):
    """Configuration for Model Prediction."""
    model_config = ConfigDict(extra='allow')

    model_path: str
    data_path: str
    output_dir: str
    model_name: Optional[str] = None
    evaluate: bool = False
    
    # Column configuration
    label_col: Optional[str] = None  # Ground truth label column name (for evaluation)
    label_col_candidates: List[str] = Field(
        default_factory=lambda: ['label', 'Target', 'class', 'diagnosis', 'outcome', 'y']
    )  # Fallback candidates if label_col not specified
    
    # Output column names
    output_label_col: str = 'predicted_label'  # Column name for predicted labels in output
    output_prob_col: str = 'predicted_probability'  # Column name for predicted probabilities in output
    
    # Probability extraction configuration
    probability_class_index: Optional[int] = None  # For multiclass: which class index to extract (None = all)
    binary_positive_class_index: int = 1  # For binary classification: which index is positive class (default: 1)

    @validator('output_dir')
    def output_dir_required(cls, v):
        if not v or not str(v).strip():
            raise ValueError("output_dir is required and cannot be empty")
        return v

def validate_config(config_dict: Dict[str, Any]) -> MLConfig:
    """Validate raw dictionary against the schema."""
    return MLConfig(**config_dict)
