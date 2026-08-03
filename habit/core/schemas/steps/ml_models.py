"""
Pydantic parameter schemas for ML models (``ModelConfig.params``).

Each model maps to the parameters accepted by the corresponding wrapper in
``habit.core.machine_learning.models``.

The declared fields are the *recommended* parameters: they drive GUI forms,
documentation and type coercion. They are not an exhaustive whitelist. Every
schema here accepts extra keys so that YAML can reach any parameter of the
underlying estimator, including ones added by a newer library version. Extra
keys are checked against the estimator's real signature when the model is
built (see ``habit.utils.estimator_utils``), so a typo is reported rather than
silently dropped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

class ModelParamsBase(BaseModel):
    """
    Base for model parameter schemas.

    Declared fields stay authoritative for GUI forms, documentation and type
    coercion, while ``extra='allow'`` lets any other parameter of the underlying
    estimator through untouched.
    """

    model_config = ConfigDict(extra="allow")


class LogisticRegressionParams(ModelParamsBase):
    C: float = Field(1.0, gt=0.0)
    penalty: Literal["l1", "l2", "elasticnet", "none"] = Field("l2")
    solver: Literal["newton-cg", "lbfgs", "liblinear", "sag", "saga"] = Field("liblinear")
    max_iter: int = Field(1000, ge=1)
    random_state: int = Field(42)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)


class SvmParams(ModelParamsBase):
    """Linear SVM (``LinearSVC``). For kernels use the ``SVC`` model instead."""

    C: float = Field(1.0, gt=0.0)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)
    random_state: int = Field(42)
    max_iter: int = Field(1000, ge=1)


class SvcParams(ModelParamsBase):
    """Kernel SVM (``SVC``) with native probability estimates."""

    C: float = Field(1.0, gt=0.0)
    kernel: Literal["linear", "poly", "rbf", "sigmoid"] = Field("rbf")
    gamma: Union[Literal["scale", "auto"], float] = Field("scale")
    degree: int = Field(3, ge=1, description="Only used by the poly kernel.")
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)
    probability: bool = Field(
        True,
        description=(
            "Required for ROC/AUC reporting. Slows training down because "
            "sklearn fits an internal calibration model."
        ),
    )
    random_state: int = Field(42)


class RandomForestParams(ModelParamsBase):
    n_estimators: int = Field(100, ge=1)
    max_depth: Optional[int] = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    max_features: Union[Literal["sqrt", "log2"], int, float, None] = Field("sqrt")
    bootstrap: bool = Field(True)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)
    random_state: int = Field(42)


class XGBoostParams(ModelParamsBase):
    n_estimators: int = Field(100, ge=1)
    max_depth: int = Field(3, ge=1)
    learning_rate: float = Field(0.1, gt=0.0, le=1.0)
    subsample: float = Field(0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(0.8, gt=0.0, le=1.0)
    objective: str = Field("binary:logistic")
    eval_metric: str = Field("logloss")
    random_state: int = Field(42)


class GradientBoostingParams(ModelParamsBase):
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


class DecisionTreeParams(ModelParamsBase):
    criterion: Literal["gini", "entropy", "log_loss"] = Field("gini")
    splitter: Literal["best", "random"] = Field("best")
    max_depth: Optional[int] = Field(None, ge=1)
    min_samples_split: int = Field(2, ge=2)
    min_samples_leaf: int = Field(1, ge=1)
    max_features: Optional[Union[str, int, float]] = Field(None)
    class_weight: Optional[Union[str, Dict[Any, float]]] = Field(None)
    random_state: int = Field(42)


class KnnParams(ModelParamsBase):
    n_neighbors: int = Field(5, ge=1)
    weights: Literal["uniform", "distance"] = Field("uniform")
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = Field("auto")
    leaf_size: int = Field(30, ge=1)
    p: int = Field(2, ge=1)
    metric: str = Field("minkowski")
    n_jobs: int = Field(-1)


class MlpParams(ModelParamsBase):
    hidden_layer_sizes: Union[int, List[int], str] = Field(
        100,
        description=(
            "Hidden layer width(s): a single int (100), a list ([100, 50]), or "
            "comma-separated integers ('100,50')."
        ),
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


class AdaBoostParams(ModelParamsBase):
    n_estimators: int = Field(50, ge=1)
    learning_rate: float = Field(1.0, gt=0.0)
    algorithm: Optional[Literal["SAMME"]] = Field(
        None,
        description=(
            "Leave unset to use the sklearn default. 'SAMME.R' was removed in "
            "scikit-learn 1.6 and the parameter itself is being phased out."
        ),
    )
    random_state: int = Field(42)


class GaussianNbParams(ModelParamsBase):
    priors: Optional[List[float]] = Field(None)
    var_smoothing: float = Field(1e-9, ge=0.0)


class MultinomialNbParams(ModelParamsBase):
    alpha: float = Field(1.0, ge=0.0)
    fit_prior: bool = Field(True)
    class_prior: Optional[List[float]] = Field(None)


class BernoulliNbParams(ModelParamsBase):
    alpha: float = Field(1.0, ge=0.0)
    binarize: float = Field(0.0, ge=0.0)
    fit_prior: bool = Field(True)
    class_prior: Optional[List[float]] = Field(None)


class AutoGluonTabularParams(ModelParamsBase):
    """
    Parameters for AutoGluon's ``TabularPredictor``.

    AutoGluon splits its API in two: the constructor defines the task, and
    ``fit()`` controls training. Both accept ``**kwargs``, so the recommended
    layout mirrors that split explicitly::

        AutoGluonTabular:
          params:
            predictor:
              eval_metric: roc_auc
              verbosity: 2
            fit:
              time_limit: 300
              presets: high_quality
              num_bag_folds: 5

    The older flat layout (``time_limit`` / ``presets`` / ``eval_metric`` at the
    top level) still works: each key is routed to whichever side of the API
    declares it. Keys unknown to both go to ``fit``, where AutoGluon's own
    ``**kwargs`` handles them.
    """

    predictor: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for TabularPredictor(...).",
    )
    fit: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for TabularPredictor.fit(...).",
    )

    # Recommended flat keys, kept for GUI forms and backward compatibility.
    label: Optional[str] = Field(None, description="Target column name.")
    problem_type: Optional[Literal["binary", "multiclass", "regression"]] = Field(None)
    eval_metric: Optional[str] = Field(None)
    path: str = Field("./AutogluonModels", json_schema_extra={"widget": "path_dir"})
    time_limit: int = Field(60, ge=1, description="Training time limit in seconds.")
    presets: str = Field(
        "medium_quality",
        description=(
            "AutoGluon preset, e.g. best_quality, high_quality, good_quality, "
            "medium_quality, optimize_for_deployment. Not restricted to a fixed "
            "list so that presets added by newer AutoGluon versions work."
        ),
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(None)

    # HABIT-level parameters, never forwarded to AutoGluon.
    feature_importance: str = Field("auto")
    random_state: int = Field(42)


class CustomEnsembleParams(ModelParamsBase):
    """Custom ensemble — advanced; base_models configured outside GUI."""

    voting: Literal["hard", "soft"] = Field("soft")


MODEL_PARAM_MODELS: Dict[str, type[BaseModel]] = {
    "LogisticRegression": LogisticRegressionParams,
    "SVM": SvmParams,
    "SVC": SvcParams,
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
