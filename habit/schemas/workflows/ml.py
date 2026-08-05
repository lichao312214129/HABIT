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
Workflow configuration schemas for machine learning.

Step ``params`` inside ``feature_selection_methods`` and ``models`` are validated
against ``habit.core.schemas.steps`` via :mod:`habit.core.schemas.validation`.
"""

from typing import ClassVar, List, Dict, Any, Optional, Tuple, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, validator, ConfigDict

from habit.schemas.base import BaseConfig
from habit.schemas.validation import validate_step_params


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

    @model_validator(mode="after")
    def _validate_params_against_registry(self) -> "FeatureSelectionMethod":
        """Coerce ``params`` using the registered selector *Params model."""
        self.params = validate_step_params(
            "feature_selection",
            self.method,
            self.params,
            preserve_keys=frozenset(),
        )
        return self


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


class ExplainabilityConfig(BaseModel):
    """
    Tuning knobs for the explanation figures beyond the SHAP summary plots.

    The figures themselves are switched on through ``plot_types``
    (``shap_dependence`` / ``shap_waterfall`` / ``permutation``); this block only
    controls how much of each is produced.
    """
    #: Number of highest-attribution features to draw dependence plots for.
    shap_dependence_top_k: int = 3
    #: Number of individual samples to export waterfall explanations for.
    shap_waterfall_samples: int = 3
    #: Shuffles per feature for permutation importance.
    permutation_repeats: int = 10
    #: sklearn scoring name used as the permutation-importance reference metric.
    permutation_scoring: str = 'roc_auc'
    #: Maximum number of features shown in the permutation-importance figure.
    permutation_top_k: int = 20
    permutation_random_state: Optional[int] = Field(
        None,
        description=(
            "Random seed for permutation shuffles. When null/omitted, inherits "
            "MLConfig.random_state."
        ),
    )


class VisualizationConfig(BaseModel):
    #: Figures implemented by ``PlotManager``. The three explanation figures are
    #: opt-in because they are markedly slower than the curve plots.
    ALLOWED_PLOT_TYPES: ClassVar[Tuple[str, ...]] = (
        'roc',
        'dca',
        'calibration',
        'pr',
        'confusion',
        'shap',
        'shap_dependence',
        'shap_waterfall',
        'permutation',
    )

    enabled: bool = True
    plot_types: List[str] = Field(default_factory=lambda: ['roc', 'dca', 'calibration', 'pr', 'confusion', 'shap'])
    dpi: int = 600
    format: str = "pdf"
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)

    @field_validator('plot_types')
    @classmethod
    def plot_types_known(cls, v: List[str]) -> List[str]:
        """
        Reject unknown figure names instead of silently skipping them.

        A typo here would otherwise produce a run that quietly omits the very
        figure the user asked for.
        """
        unknown = [name for name in v if name not in cls.ALLOWED_PLOT_TYPES]
        if unknown:
            raise ValueError(
                f"Unknown visualization.plot_types: {unknown}. "
                f"Supported: {list(cls.ALLOWED_PLOT_TYPES)}."
            )
        return v


class BootstrapConfig(BaseModel):
    """
    Bootstrap resampling settings for metric confidence intervals.

    Disabled by default so existing runs keep their exact report shape. When
    enabled, every reported performance metric gains a percentile confidence
    interval, which imaging journals routinely require alongside point
    estimates.
    """
    enabled: bool = False
    n_iterations: int = 1000
    ci_level: float = 0.95
    # Resample within each class so every replicate preserves the observed class
    # prevalence. Unstratified resampling of a small or imbalanced cohort yields
    # replicates that contain a single class, where AUC is undefined and the
    # interval silently narrows toward the surviving replicates.
    stratified: bool = True
    random_state: Optional[int] = Field(
        None,
        description=(
            "Random seed for bootstrap resampling. When null/omitted, inherits "
            "MLConfig.random_state."
        ),
    )

    @field_validator('n_iterations')
    @classmethod
    def n_iterations_sufficient(cls, v: int) -> int:
        """Require enough replicates to estimate a percentile interval."""
        if v < 100:
            raise ValueError(
                "bootstrap.n_iterations must be at least 100 for a usable "
                "percentile interval"
            )
        return v

    @field_validator('ci_level')
    @classmethod
    def ci_level_range(cls, v: float) -> float:
        """Confidence level must be a proportion, not a percentage."""
        if not (0 < v < 1):
            raise ValueError("bootstrap.ci_level must be between 0 and 1, e.g. 0.95")
        return v


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
    # When True, a model parameter the underlying estimator does not accept
    # aborts the run instead of being reported and ignored.
    strict_model_params: bool = False

    # Visualization detail.
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    # Evaluation detail: bootstrap confidence intervals for reported metrics.
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

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

    @model_validator(mode="before")
    @classmethod
    def _coerce_registered_step_params(cls, data: Any) -> Any:
        """Validate feature selector and model ``params`` against registered schemas."""
        if not isinstance(data, dict):
            return data
        out = dict(data)

        methods = out.get("feature_selection_methods")
        if isinstance(methods, list):
            coerced_methods: List[Any] = []
            for item in methods:
                if isinstance(item, dict) and item.get("method"):
                    params = validate_step_params(
                        "feature_selection",
                        str(item["method"]),
                        dict(item.get("params") or {}),
                        preserve_keys=frozenset(),
                    )
                    coerced_methods.append({**item, "params": params})
                else:
                    coerced_methods.append(item)
            out["feature_selection_methods"] = coerced_methods

        models = out.get("models")
        if isinstance(models, dict):
            coerced_models: Dict[str, Any] = {}
            for model_name, block in models.items():
                if isinstance(block, dict):
                    params = validate_step_params(
                        "model",
                        str(model_name),
                        dict(block.get("params") or {}),
                        preserve_keys=frozenset(),
                    )
                    coerced_models[model_name] = {**block, "params": params}
                else:
                    coerced_models[model_name] = block
            out["models"] = coerced_models

        return out

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
