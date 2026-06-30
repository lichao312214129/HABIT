# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Configuration Schemas for Machine Learning Workflows.

Uses Pydantic for robust validation and type safety.

V1 note
-------
The legacy ``PredictionConfig`` class has been merged into :class:`MLConfig`.
Train and predict now share one config schema; ``run_mode`` selects the
behaviour and a ``model_validator`` enforces the cross-field invariants
(predict requires ``pipeline_path``; train requires non-empty ``models``).
"""

from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, validator, ConfigDict

from habit.core.common.configs.base import BaseConfig


class InputFileConfig(BaseModel):
    path: str
    name: str = ""
    subject_id_col: str
    label_col: str
    features: Optional[List[str]] = None
    features_from_log: Optional[str] = None
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


class ResamplingConfig(BaseModel):
    """
    Training-set resampling configuration.

    Notes:
        - This module applies only to training data.
        - It is disabled by default.
    """
    enabled: bool = False
    method: Literal['random_over', 'random_under', 'smote'] = 'random_over'
    position: Literal[
        'before_feature_selection',
        'before_normalization',
        'after_normalization',
        'before_model',
    ] = 'before_model'
    ratio: float = 1.0
    random_state: Optional[int] = Field(
        None,
        description=(
            "Random seed for resampling. When null/omitted, inherits MLConfig.random_state."
        ),
    )

    @validator('ratio')
    def ratio_range(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("resampling.ratio must be > 0")
        return v


# Backward-compatible alias for external imports that still use SamplingConfig.
SamplingConfig = ResamplingConfig


class VisualizationConfig(BaseModel):
    enabled: bool = True
    plot_types: List[str] = Field(default_factory=lambda: ['roc', 'dca', 'calibration', 'pr', 'confusion', 'shap'])
    dpi: int = 600
    format: str = "pdf"


class ComparisonFileConfig(BaseModel):
    """
    Single prediction file row inside ``files_config``.

    ``model_name`` can be inferred before field validation runs so aliases like
    ``name`` participate correctly (ordering of per-field validators alone is
    insufficient when ``name`` is declared after ``model_name``).
    """

    path: str
    model_name: Optional[str] = None
    name: Optional[str] = None
    subject_id_col: str
    label_col: str
    prob_col: str
    pred_col: Optional[str] = None
    split_col: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def _resolve_model_name(cls, data: Any) -> Any:
        """
        Resolve ``model_name`` from explicit value, ``name`` alias, or path stem.

        Priority:
            1) model_name (explicit non-empty)
            2) name (alias, non-empty)
            3) file stem from path
        """
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        path = raw.get('path')
        explicit = raw.get('model_name')
        if explicit is not None and str(explicit).strip():
            raw['model_name'] = str(explicit).strip()
            return raw
        alias_name = raw.get('name')
        if alias_name is not None and str(alias_name).strip():
            raw['model_name'] = str(alias_name).strip()
            return raw
        if path is not None and str(path).strip():
            stem = str(path).replace('\\', '/').split('/')[-1].split('.')[0]
            raw['model_name'] = stem
            return raw
        raise ValueError('model_name is required (or provide name/path to infer it).')


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

    @field_validator('targets')
    @classmethod
    def target_values_range(cls, v: Dict[str, float]) -> Dict[str, float]:
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

    __pydantic_extra__: dict[str, Any]

    output_dir: str
    files_config: List[ComparisonFileConfig] = Field(min_length=1)
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
    """
    Unified configuration for the standard / k-fold ML workflows.

    The same schema covers both training and prediction. ``run_mode`` selects
    the behaviour:

    - ``run_mode='train'``: train models on ``input`` files. ``models`` must
      be non-empty.
    - ``run_mode='predict'``: load ``pipeline_path`` and predict on
      ``input[0].path``. ``models`` is ignored.
    """
    model_config = ConfigDict(extra='allow')  # Forward-compatible for new keys.

    __pydantic_extra__: dict[str, Any]

    # Mode dispatch.
    run_mode: Literal['train', 'predict'] = 'train'
    pipeline_path: Optional[str] = None  # Required when run_mode='predict'.

    # Data input. In predict mode only ``input[0]`` is consumed.
    input: List[InputFileConfig]
    output: str

    random_state: int = 42

    # Validation / Splitting (train mode only).
    split_method: Literal['random', 'stratified', 'custom'] = 'stratified'
    test_size: float = 0.3
    train_ids_file: Optional[str] = None
    test_ids_file: Optional[str] = None

    # K-Fold specific.
    n_splits: int = 5
    stratified: bool = True

    # Core components.
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    resampling: ResamplingConfig = Field(default_factory=ResamplingConfig)
    feature_selection_methods: List[FeatureSelectionMethod] = Field(default_factory=list)
    # Optional in predict mode; required + non-empty in train mode (enforced
    # by the model_validator below).
    models: Optional[Dict[str, ModelConfig]] = None

    # Flags.
    is_visualize: bool = True
    is_save_model: bool = True

    # Visualization detail.
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    # Predict-mode specific (ignored in train mode).
    evaluate: bool = False
    output_label_col: str = 'predicted_label'
    output_prob_col: str = 'predicted_probability'
    # For multiclass: which class index to extract (None = all).
    probability_class_index: Optional[int] = None
    # For binary classification: which index is positive class (default: 1).
    binary_positive_class_index: int = 1

    @validator('test_size')
    def test_size_range(cls, v):
        if not (0 < v < 1):
            raise ValueError('test_size must be between 0 and 1')
        return v

    @model_validator(mode='before')
    @classmethod
    def _migrate_legacy_sampling_key(cls, data: Any) -> Any:
        """
        Accept the previous ``sampling`` key while exposing ``resampling``.

        YAML files are part of the user-facing interface.  Renaming the block
        should not break existing configurations immediately, so the legacy
        key is copied into the new key when ``resampling`` is absent.
        """
        if not isinstance(data, dict):
            return data
        if 'resampling' in data or 'sampling' not in data:
            return data
        migrated = dict(data)
        migrated['resampling'] = migrated.pop('sampling')
        return migrated

    @property
    def sampling(self) -> ResamplingConfig:
        """
        Backward-compatible attribute alias for older Python callers.

        New code should read ``config.resampling``.  This property keeps older
        code that accessed ``config.sampling`` working after the YAML key was
        renamed.
        """
        return self.resampling

    @model_validator(mode='after')
    def _validate_run_mode(self) -> 'MLConfig':
        if self.run_mode == 'train':
            if not self.models:
                raise ValueError(
                    "MLConfig: run_mode='train' requires a non-empty 'models' "
                    "dictionary."
                )
        else:  # predict
            if not self.pipeline_path:
                raise ValueError(
                    "MLConfig: run_mode='predict' requires 'pipeline_path' "
                    "(path to a saved *_final_pipeline.pkl)."
                )
            if not self.input:
                raise ValueError(
                    "MLConfig: run_mode='predict' requires at least one entry "
                    "in 'input' (input[0].path is used as the data file)."
                )
        return self


def validate_config(config_dict: Dict[str, Any]) -> MLConfig:
    """Validate a raw dictionary against the schema and return :class:`MLConfig`."""
    return MLConfig(**config_dict)


# -----------------------------------------------------------------------------
# Test-Retest Analysis Schemas
# -----------------------------------------------------------------------------

class TestRetestConfig(BaseConfig):
    """Configuration for test-retest reproducibility analysis."""

    test_habitat_table: str = Field(..., description="Path to test group habitat feature table (CSV or Excel)")
    retest_habitat_table: str = Field(..., description="Path to retest group habitat feature table (CSV or Excel)")

    features: Optional[List[str]] = Field(None, description="List of feature names for similarity calculation (None = all)")
    similarity_method: Literal['pearson', 'spearman', 'kendall', 'euclidean', 'cosine', 'manhattan', 'chebyshev'] = Field(
        'pearson',
        description="Similarity calculation method"
    )

    input_dir: str = Field(..., description="Directory containing retest group NRRD files")
    out_dir: str = Field(..., description="Output directory for processed files")

    processes: int = Field(4, description="Number of parallel processes", gt=0)
    debug: bool = Field(False, description="Enable debug logging")
