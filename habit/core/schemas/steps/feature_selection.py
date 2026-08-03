"""
Pydantic parameter schemas for feature selection methods.

Pipeline-injected arguments (``X``, ``y``, ``selected_features``, ``outdir``) are omitted.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RFECV_ESTIMATORS: tuple[str, ...] = (
    "LogisticRegression",
    "RandomForestClassifier",
    "SVC",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LinearRegression",
    "RandomForestRegressor",
    "SVR",
    "GradientBoostingRegressor",
    "XGBRegressor",
)


class VarianceParams(BaseModel):
    threshold: float = Field(0.0, ge=0.0, json_schema_extra={"order": 1})
    plot_variances: bool = Field(True, json_schema_extra={"order": 2})
    top_k: Optional[int] = Field(None, ge=1, json_schema_extra={"order": 3})
    top_percent: Optional[float] = Field(
        None, ge=0.0, le=100.0, json_schema_extra={"order": 4}
    )


class CorrelationParams(BaseModel):
    threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Maximum absolute correlation to keep."
    )
    method: Literal["pearson", "spearman", "kendall"] = Field("spearman")
    visualize: bool = Field(False)


class VifParams(BaseModel):
    max_vif: float = Field(10.0, gt=0.0)
    visualize: bool = Field(False)


class LassoParams(BaseModel):
    cv: int = Field(10, ge=2, le=20)
    n_alphas: int = Field(100, ge=1)
    alphas: Optional[List[float]] = Field(None, description="Optional explicit alpha grid.")
    random_state: int = Field(42)
    visualize: bool = Field(False)


#: Shared description for the dual-notation ``n_features_to_select`` parameter.
N_FEATURES_TO_SELECT_DESCRIPTION: str = (
    "Number of top-ranked features to keep. Use an integer >= 1 for an absolute "
    "count (e.g. 20), or a value in (0, 1) for a ratio of the candidate features "
    "(e.g. 0.2 keeps the top 20%). Overrides p_threshold when set."
)


class AnovaParams(BaseModel):
    p_threshold: float = Field(0.05, gt=0.0, lt=1.0)
    n_features_to_select: Optional[float] = Field(
        None, gt=0.0, description=N_FEATURES_TO_SELECT_DESCRIPTION
    )
    plot_importance: bool = Field(True)


class Chi2Params(BaseModel):
    p_threshold: float = Field(0.05, gt=0.0, lt=1.0)
    n_features_to_select: Optional[float] = Field(
        None, gt=0.0, description=N_FEATURES_TO_SELECT_DESCRIPTION
    )
    plot_importance: bool = Field(True)


class StatisticalTestParams(BaseModel):
    p_threshold: float = Field(0.05, gt=0.0, lt=1.0)
    n_features_to_select: Optional[float] = Field(
        None, gt=0.0, description=N_FEATURES_TO_SELECT_DESCRIPTION
    )
    normality_test_threshold: float = Field(0.05, gt=0.0, lt=1.0)
    plot_importance: bool = Field(True)
    force_test: Optional[Literal["ttest", "mannwhitney"]] = Field(
        None, description="Force a specific test instead of auto-selection."
    )


class MrmrParams(BaseModel):
    n_features: int = Field(10, ge=1)
    task_type: Literal["classification", "regression"] = Field("classification")


class RfecvParams(BaseModel):
    estimator: Literal[
        "LogisticRegression",
        "RandomForestClassifier",
        "SVC",
        "GradientBoostingClassifier",
        "XGBClassifier",
        "LinearRegression",
        "RandomForestRegressor",
        "SVR",
        "GradientBoostingRegressor",
        "XGBRegressor",
    ] = Field("RandomForestClassifier")
    step: int = Field(1, ge=1)
    cv: int = Field(5, ge=2)
    scoring: str = Field("roc_auc")
    min_features_to_select: int = Field(1, ge=1)
    n_jobs: int = Field(-1)
    random_state: Optional[int] = Field(None)
    visualize: bool = Field(False)


class StepwiseParams(BaseModel):
    direction: Literal["forward", "backward", "both"] = Field("backward")
    threshold_in: float = Field(0.05, gt=0.0, lt=1.0)
    threshold_out: float = Field(0.05, gt=0.0, lt=1.0)
    criterion: Literal["aic", "bic", "pvalue"] = Field("aic")
    verbose: bool = Field(False)


class UnivariateLogisticParams(BaseModel):
    alpha: float = Field(0.05, gt=0.0, lt=1.0)


class IccParams(BaseModel):
    icc_results: Optional[str] = Field(
        None,
        description="Path to ICC results JSON (alias for icc_results_path).",
        json_schema_extra={"widget": "path_file"},
    )
    icc_results_path: Optional[str] = Field(
        None,
        description="Path to ICC results JSON file.",
        json_schema_extra={"widget": "path_file"},
    )
    keys: Optional[List[str]] = Field(None, description="Group keys (alias for groups).")
    groups: Optional[List[str]] = Field(None, description="ICC group identifiers.")
    threshold: float = Field(0.75, ge=0.0, le=1.0)
    metric: Optional[str] = Field(None, description="ICC metric name (e.g. ICC3).")


FEATURE_SELECTION_PARAM_MODELS: Dict[str, type[BaseModel]] = {
    "variance": VarianceParams,
    "correlation": CorrelationParams,
    "vif": VifParams,
    "lasso": LassoParams,
    "anova": AnovaParams,
    "chi2": Chi2Params,
    "statistical_test": StatisticalTestParams,
    "mrmr": MrmrParams,
    "rfecv": RfecvParams,
    "stepwise": StepwiseParams,
    "univariate_logistic": UnivariateLogisticParams,
    "icc": IccParams,
}
