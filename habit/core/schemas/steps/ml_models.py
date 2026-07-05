"""
Pydantic parameter schemas for ML models (``ModelConfig.params``).

Each model maps to the parameters accepted by the corresponding wrapper in
``habit.core.machine_learning.models``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class LogisticRegressionParams(BaseModel):
    C: float = Field(1.0, gt=0.0)
    penalty: Literal["l1", "l2", "elasticnet", "none"] = Field("l2")
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = Field("liblinear")
    max_iter: int = Field(1000, ge=1)
    random_state: int = Field(42)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)


class SvmParams(BaseModel):
    C: float = Field(1.0, gt=0.0)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)
    random_state: int = Field(42)
    max_iter: int = Field(1000, ge=1)


class RandomForestParams(BaseModel):
    n_estimators: int = Field(100, ge=1)
    max_depth: Optional[int] = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    max_features: Union[Literal["sqrt", "log2"], int, float, None] = Field("sqrt")
    bootstrap: bool = Field(True)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)
    random_state: int = Field(42)


class XGBoostParams(BaseModel):
    n_estimators: int = Field(100, ge=1)
    max_depth: int = Field(3, ge=1)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    subsample: float = Field(0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(0.8, gt=0.0, le=1.0)
    objective: str = Field("binary:logistic")
    eval_metric: str = Field("logloss")
    random_state: int = Field(42)


class GradientBoostingParams(BaseModel):
    loss: str = Field("log_loss")
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    n_estimators: int = Field(100, ge=1)
    subsample: float = Field(1.0, gt=0.0, le=1.0)
    criterion: str = Field("friedman_mse")
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    max_depth: int = Field(3, ge=1)
    max_features: Optional[Union[str, int, float]] = Field(None)
    random_state: int = Field(42)


class DecisionTreeParams(BaseModel):
    criterion: Literal["gini", "entropy", "log_loss"] = Field("gini")
    splitter: Literal["best", "random"] = Field("best")
    max_depth: Optional[int] = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    max_features: Optional[Union[str, int, float]] = Field(None)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)
    random_state: int = Field(42)


class KnnParams(BaseModel):
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = Field("uniform")
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = Field("auto")
    leaf_size: int = Field(30, ge=1)
    p: int = Field(2, ge=1)
    metric: str = Field("minkowski")
    n_jobs: int = Field(-1)


class MlpParams(BaseModel):
    hidden_layer_sizes: str = Field(
        "100",
        description="Hidden layer sizes as comma-separated integers, e.g. 100,50.",
    )
    activation: Literal["identity", "logistic", "tanh", "relu"] = Field("relu")
    solver: Literal["lbfgs", "sgd", "adam"] = Field("adam")
    alpha: float = Field(0.0001, ge=0.0)
    batch_size: Union[int, Literal["auto"]] = Field("auto")
    learning_rate: Literal["constant", "invscaling", "adaptive"] = Field("constant")
    learning_rate_init: float = Field(0.001, gt=0.0)
    max_iter: int = Field(200, ge=1)
    shuffle: bool = Field(True)
    random_state: int = Field(42)
    early_stopping: bool = Field(False)
    validation_fraction: float = Field(0.1, gt=0.0, lt=1.0)


class AdaBoostParams(BaseModel):
    n_estimators: int = Field(50, ge=1)
    learning_rate: float = Field(1.0, gt=0.0)
    algorithm: Literal["SAMME", "SAMME.R"] = Field("SAMME.R")
    random_state: int = Field(42)


class GaussianNbParams(BaseModel):
    priors: Optional[List[float]] = Field(None)
    var_smoothing: float = Field(1e-9, ge=0.0)


class MultinomialNbParams(BaseModel):
    alpha: float = Field(1.0, ge=0.0)
    fit_prior: bool = Field(True)
    class_prior: Optional[List[float]] = Field(None)


class BernoulliNbParams(BaseModel):
    alpha: float = Field(1.0, ge=0.0)
    binarize: float = Field(0.0, ge=0.0)
    fit_prior: bool = Field(True)
    class_prior: Optional[List[float]] = Field(None)


class AutoGluonTabularParams(BaseModel):
    label: Optional[str] = Field(None, description="Target column name.")
    problem_type: Optional[Literal["binary", "multiclass", "regression"]] = Field(None)
    eval_metric: Optional[str] = Field(None)
    path: str = Field("./AutogluonModels", json_schema_extra={"widget": "path_dir"})
    time_limit: int = Field(60, ge=1, description="Training time limit in seconds.")
    presets: Literal[
        "best_quality",
        "high_quality",
        "good_quality",
        "medium_quality",
        "optimize_for_deployment",
    ] = Field("medium_quality")
    hyperparameters: Optional[Dict[str, Any]] = Field(None)
    feature_importance: str = Field("auto")
    random_state: int = Field(42)


class CustomEnsembleParams(BaseModel):
    """Custom ensemble — advanced; base_models configured outside GUI."""

    model_config = ConfigDict(extra="allow")
    voting: Literal["hard", "soft"] = Field("soft")


MODEL_PARAM_MODELS: Dict[str, type[BaseModel]] = {
    "LogisticRegression": LogisticRegressionParams,
    "SVM": SvmParams,
    "RandomForest": RandomForestParams,
    "XGBoost": XGBoostParams,
    "GradientBoosting": GradientBoostingParams,
    "DecisionTree": DecisionTreeParams,
    "KNN": KnnParams,
    "MLP": MlpParams,
    "AdaBoost": AdaBoostParams,
    "GaussianNB": GaussianNbParams,
    "MultinomialNB": MultinomialNbParams,
    "BernoulliNB": BernoulliNbParams,
    "AutoGluonTabular": AutoGluonTabularParams,
    "CustomEnsemble": CustomEnsembleParams,
}
