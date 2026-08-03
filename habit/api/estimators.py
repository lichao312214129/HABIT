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
"""Scikit-learn compatible estimators built on HABIT public workflows.

The estimators in this module deliberately keep their orchestration layer thin:
domain validation, service construction, and image processing remain owned by
the established HABIT workflow implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import accuracy_score

from habit._version import __version__
from habit.api.exceptions import CompatibilityError, HABITAPIError, NotFittedError

_ESTIMATOR_SERIALIZATION_SCHEMA_VERSION = 1


class EstimatorPersistenceMixin:
    """Provide explicit, type-safe joblib persistence for public estimators."""

    serialization_metadata_: Dict[str, Any]

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialize this fitted estimator.

        Args:
            path: Destination joblib path. Parent directories are created.

        Raises:
            NotFittedError: If the estimator has not been fitted.
        """
        _require_fitted(self)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": _ESTIMATOR_SERIALIZATION_SCHEMA_VERSION,
            "habit_version": __version__,
            "sklearn_version": sklearn.__version__,
            "estimator_type": type(self).__name__,
            "config_version": getattr(
                getattr(self, "config", None),
                "config_version",
                None,
            ),
        }
        joblib.dump({"metadata": metadata, "estimator": self}, destination)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EstimatorPersistenceMixin":
        """
        Load an estimator previously saved by :meth:`save`.

        Args:
            path: Source joblib path.

        Returns:
            Loaded estimator of the requested class.

        Raises:
            CompatibilityError: If the artifact uses an unsupported persistence
                schema version.
            TypeError: If the serialized object has a different estimator type.
        """
        payload = joblib.load(Path(path))
        metadata: Dict[str, Any] = {}
        if isinstance(payload, Mapping) and "estimator" in payload:
            metadata = dict(payload.get("metadata", {}))
            schema_version = metadata.get("schema_version")
            if schema_version != _ESTIMATOR_SERIALIZATION_SCHEMA_VERSION:
                raise CompatibilityError(
                    "Unsupported public estimator serialization schema "
                    f"{schema_version!r}; expected "
                    f"{_ESTIMATOR_SERIALIZATION_SCHEMA_VERSION}."
                )
            loaded = payload["estimator"]
        else:
            # Support artifacts written by HABIT before persistence metadata was
            # introduced. Future serializers always use the envelope above.
            loaded = payload
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Serialized object is {type(loaded).__name__}, not {cls.__name__}."
            )
        if metadata:
            loaded.serialization_metadata_ = metadata
        return loaded


def _require_fitted(estimator: Any) -> None:
    """Raise a sklearn-standard error unless an estimator has been fitted."""
    if not getattr(estimator, "_is_fitted_", False):
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' before using this method."
        )


def _as_feature_dataframe(
    X: Union[pd.DataFrame, np.ndarray],
    *,
    fitted_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Normalize tabular features while preserving and validating column names.

    Args:
        X: Feature matrix as a DataFrame or two-dimensional ndarray.
        fitted_columns: Required column order learned during fit, if available.

    Returns:
        A two-dimensional feature DataFrame in fitted column order.

    Raises:
        HABITAPIError: If the input is not two-dimensional or its columns differ.
    """
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise HABITAPIError("X must be a two-dimensional DataFrame or ndarray.")
        if fitted_columns is None:
            frame = pd.DataFrame(X)
        else:
            if X.shape[1] != len(fitted_columns):
                raise HABITAPIError(
                    f"X has {X.shape[1]} columns; expected {len(fitted_columns)}."
                )
            frame = pd.DataFrame(X, columns=list(fitted_columns))
    else:
        raise HABITAPIError(
            "X must be a pandas DataFrame or a two-dimensional ndarray."
        )

    if frame.ndim != 2 or frame.shape[1] == 0:
        raise HABITAPIError("X must contain at least one feature column.")
    if not frame.columns.is_unique:
        raise HABITAPIError("X column names must be unique.")

    if fitted_columns is not None:
        expected = list(fitted_columns)
        actual = list(frame.columns)
        missing = [name for name in expected if name not in frame.columns]
        unexpected = [name for name in actual if name not in expected]
        if missing or unexpected:
            raise HABITAPIError(
                "X columns do not match fitted columns. "
                f"Missing: {missing or 'none'}; unexpected: {unexpected or 'none'}."
            )
        frame = frame.loc[:, expected]
    return frame


class SubjectFeatureAggregator(
    EstimatorPersistenceMixin, BaseEstimator, TransformerMixin
):
    """
    Aggregate a HABIT habitat-result long table into one row per subject.

    The standard HABIT result identifiers are ``subject`` and ``habitats``.
    Each numeric feature is aggregated for every fitted habitat label and named
    ``<feature>__habitat_<label>``. Missing subject/habitat combinations are
    represented by ``fill_value`` so output schema remains stable. By default,
    subject identifiers remain in the DataFrame index rather than becoming a
    model feature; set ``keep_subject_column=True`` for export-oriented tables.
    """

    def __init__(
        self,
        subject_column: str = "subject",
        habitat_column: str = "habitats",
        aggregation: str = "mean",
        fill_value: float = 0.0,
        keep_subject_column: bool = False,
    ) -> None:
        """Initialize aggregation parameters without inspecting any data."""
        self.subject_column = subject_column
        self.habitat_column = habitat_column
        self.aggregation = aggregation
        self.fill_value = fill_value
        self.keep_subject_column = keep_subject_column

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
    ) -> "SubjectFeatureAggregator":
        """Learn feature, habitat, and output-column schema from a long table."""
        frame, feature_columns = self._validate_long_table(X)
        self.feature_columns_ = tuple(feature_columns)
        self.identifier_columns_ = (self.subject_column, self.habitat_column)
        self.input_columns_ = tuple(frame.columns)
        self.feature_names_in_ = np.asarray(self.feature_columns_, dtype=object)
        self.n_features_in_ = len(self.feature_columns_)
        self.habitat_labels_ = tuple(
            sorted(frame[self.habitat_column].dropna().unique())
        )
        if not self.habitat_labels_:
            raise HABITAPIError(
                f"'{self.habitat_column}' must contain at least one label."
            )
        self.output_feature_names_ = tuple(
            f"{feature}__habitat_{label}"
            for feature in self.feature_columns_
            for label in self.habitat_labels_
        )
        self._is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a long table and align its columns to the fitted schema."""
        _require_fitted(self)
        frame, _ = self._validate_long_table(X, require_fitted_features=True)
        grouped = frame.groupby([self.subject_column, self.habitat_column], sort=True)[
            list(self.feature_columns_)
        ].agg(self.aggregation)
        subjects = pd.Index(sorted(frame[self.subject_column].dropna().unique()))
        result = pd.DataFrame(index=subjects)
        for feature in self.feature_columns_:
            for label in self.habitat_labels_:
                name = f"{feature}__habitat_{label}"
                if (feature in grouped.columns) and (
                    label in grouped.index.get_level_values(1)
                ):
                    values = grouped.xs(label, level=self.habitat_column)[feature]
                    result[name] = values.reindex(subjects)
                else:
                    result[name] = np.nan
        result.index.name = self.subject_column
        result = result.fillna(self.fill_value)
        if self.keep_subject_column:
            return result.reset_index()
        return result

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Return stable subject-table output column names."""
        _require_fitted(self)
        names = self.output_feature_names_
        if self.keep_subject_column:
            names = (self.subject_column, *names)
        return cast(np.ndarray, np.asarray(names, dtype=object))

    def _validate_long_table(
        self,
        X: pd.DataFrame,
        *,
        require_fitted_features: bool = False,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate required identifiers and select usable numeric feature columns."""
        if not isinstance(X, pd.DataFrame):
            raise HABITAPIError(
                "X must be a pandas DataFrame containing habitat results."
            )
        required = {self.subject_column, self.habitat_column}
        missing = sorted(required.difference(X.columns))
        if missing:
            raise HABITAPIError(f"X is missing required columns: {missing}.")
        frame = X.copy()
        feature_columns = [
            column
            for column in frame.columns
            if (
                column not in required
                and column not in {"supervoxel", "count"}
                and not str(column).endswith("-original")
                and pd.api.types.is_numeric_dtype(frame[column])
            )
        ]
        if require_fitted_features:
            missing_features = [
                column
                for column in self.feature_columns_
                if column not in frame.columns
            ]
            if missing_features:
                raise HABITAPIError(
                    f"X is missing fitted feature columns: {missing_features}."
                )
            unexpected_features = [
                column
                for column in feature_columns
                if column not in self.feature_columns_
            ]
            if unexpected_features:
                raise HABITAPIError(
                    f"X has unexpected numeric feature columns: {unexpected_features}."
                )
            feature_columns = list(self.feature_columns_)
        elif not feature_columns:
            raise HABITAPIError("X must contain at least one numeric feature column.")
        return frame, feature_columns


class HabitClassifier(EstimatorPersistenceMixin, BaseEstimator, ClassifierMixin):
    """
    sklearn classifier facade backed by HABIT's ``MLConfig`` and PipelineBuilder.

    Only registered classifier models are supported. Regression configurations
    are intentionally rejected because HABIT's ML workflow is classification
    oriented and its public evaluation contract assumes class probabilities.
    """

    def __init__(
        self,
        config: Any,
        model_name: Optional[str] = None,
        model_params: Optional[Mapping[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Store clone-safe configuration parameters without building heavy objects."""
        self.config = config
        self.model_name = model_name
        self.model_params = model_params
        self.output_dir = output_dir

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[Any]],
    ) -> "HabitClassifier":
        """Build and fit the configured HABIT classification pipeline."""
        frame = _as_feature_dataframe(X)
        targets = np.asarray(y)
        if targets.ndim != 1 or len(targets) != len(frame):
            raise HABITAPIError(
                "y must be one-dimensional and match the number of X rows."
            )
        if len(np.unique(targets)) < 2:
            raise HABITAPIError("HabitClassifier requires at least two target classes.")

        config = self._validated_config()
        model_name, model_params = self._resolve_model_spec(config)
        pipeline = self._build_pipeline(
            config, model_name, model_params, frame.columns.tolist()
        )
        pipeline.fit(frame, targets)

        model = pipeline.named_steps["model"]
        underlying = getattr(model, "model", model)
        if not hasattr(underlying, "predict") or not hasattr(underlying, "classes_"):
            raise HABITAPIError(
                f"Registered model '{model_name}' is not a fitted classifier."
            )
        self.pipeline_ = pipeline
        self.model_ = model
        self.classes_ = np.asarray(underlying.classes_)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = frame.shape[1]
        self._is_fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class labels using the fitted HABIT pipeline."""
        frame = self._validated_fitted_input(X)
        return cast(np.ndarray, np.asarray(self.pipeline_.predict(frame)))

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities using the fitted HABIT pipeline."""
        frame = self._validated_fitted_input(X)
        probabilities = np.asarray(self.pipeline_.predict_proba(frame))
        if probabilities.ndim == 1:
            probabilities = np.column_stack((1.0 - probabilities, probabilities))
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.classes_):
            raise HABITAPIError(
                "The configured model did not return one probability column per class."
            )
        return cast(np.ndarray, probabilities)

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[Any]],
    ) -> float:
        """Return classification accuracy, matching sklearn classifier convention."""
        return float(accuracy_score(y, self.predict(X)))

    def _validated_fitted_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> pd.DataFrame:
        """Validate inference data against the exact fitted feature schema."""
        _require_fitted(self)
        return _as_feature_dataframe(X, fitted_columns=self.feature_names_in_)

    def _validated_config(self) -> Any:
        """Coerce the configured public MLConfig lazily to avoid eager imports."""
        from habit.core.machine_learning.config_schemas import MLConfig

        if isinstance(self.config, MLConfig):
            config = self.config
        elif isinstance(self.config, Mapping):
            config = MLConfig.model_validate(dict(self.config))
        else:
            raise HABITAPIError(
                "config must be an MLConfig or a mapping accepted by MLConfig."
            )
        if config.run_mode != "train":
            raise HABITAPIError(
                "HabitClassifier requires MLConfig with run_mode='train'."
            )
        return config

    def _resolve_model_spec(self, config: Any) -> Tuple[str, Dict[str, Any]]:
        """Resolve one configured classifier without mutating the supplied config."""
        configured_models = config.models or {}
        if self.model_name is None:
            if len(configured_models) != 1:
                raise HABITAPIError(
                    "model_name is required unless MLConfig.models contains exactly one model."
                )
            model_name = next(iter(configured_models))
        else:
            model_name = self.model_name

        configured = configured_models.get(model_name)
        if self.model_params is not None:
            supplied_params = dict(self.model_params)
            params = (
                supplied_params
                if "params" in supplied_params
                else {"params": supplied_params}
            )
        elif configured is not None:
            params = {"params": dict(configured.params)}
        else:
            raise HABITAPIError(
                f"Model '{model_name}' is not configured in MLConfig.models."
            )
        return model_name, params

    @staticmethod
    def _build_pipeline(
        config: Any,
        model_name: str,
        model_params: Dict[str, Any],
        feature_names: List[str],
    ) -> Any:
        """Delegate pipeline construction to HABIT's registered model infrastructure."""
        from habit.core.machine_learning.pipeline_builder import PipelineBuilder

        return PipelineBuilder(config=config).build(
            model_name=model_name,
            model_params=model_params,
            feature_names=feature_names,
        )


class OutcomeClassifier(HabitClassifier):
    """Clinically named public alias for HABIT's outcome-classification estimator.

    This class intentionally adds no separate implementation.  It gives
    clinical researchers a task-oriented name while retaining the complete
    sklearn-compatible ``HabitClassifier`` parameter, fit, prediction, and
    persistence contract.
    """


class HabitatClusterer(EstimatorPersistenceMixin, BaseEstimator, TransformerMixin):
    """
    sklearn-style facade for HABIT image habitat clustering.

    ``X`` is an optional sequence of subject identifiers; image and mask data
    remain discovered from ``config.data_dir`` exactly as in the HABIT CLI.
    ``transform`` reuses the fitted in-memory pipeline, so it does not require
    the training output directory or trigger an implicit serialization round
    trip. ``fit_transform`` returns the fit result directly to prevent a
    second image prediction pass.
    """

    def __init__(
        self,
        config: Any,
        save_results_csv: Optional[bool] = None,
    ) -> None:
        """Store validated-at-fit habitat configuration parameters."""
        self.config = config
        self.save_results_csv = save_results_csv

    def fit(
        self,
        X: Optional[Sequence[str]] = None,
        y: Optional[np.ndarray] = None,
    ) -> "HabitatClusterer":
        """Create HABIT services and delegate training to ``HabitatAnalysis.fit``."""
        config = self._validated_config()
        subjects = self._subjects_from_input(X)
        analysis = self._create_analysis(config)
        results = analysis.fit(
            subjects=subjects, save_results_csv=self.save_results_csv
        )
        self.analysis_ = analysis
        self.pipeline_ = analysis.pipeline
        self.results_ = results
        self.feature_names_in_ = np.asarray(["subject"], dtype=object)
        self.n_features_in_ = 1
        self._is_fitted_ = True
        return self

    def transform(self, X: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Predict habitat labels with the fitted in-memory pipeline."""
        _require_fitted(self)
        subjects = self._subjects_from_input(X)
        return self.analysis_.transform_with_pipeline(
            pipeline=self.pipeline_,
            subjects=subjects,
            save_results_csv=self.save_results_csv,
        )

    def fit_transform(
        self,
        X: Optional[Sequence[str]] = None,
        y: Optional[np.ndarray] = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        """Fit once and return training results without an additional prediction run."""
        self.fit(X, y)
        return self.results_

    def predict(self, X: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Alias :meth:`transform` because habitat inference returns a result table."""
        return self.transform(X)

    def _validated_config(self) -> Any:
        """Coerce habitat configuration only when image processing is requested."""
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

        if isinstance(self.config, HabitatAnalysisConfig):
            return self.config
        if isinstance(self.config, Mapping):
            return HabitatAnalysisConfig.model_validate(dict(self.config))
        raise HABITAPIError(
            "config must be a HabitatAnalysisConfig or a mapping accepted by it."
        )

    @staticmethod
    def _create_analysis(config: Any) -> Any:
        """Delegate service assembly to HABIT's existing configurator."""
        from habit.core.habitat_analysis.configurator import HabitatConfigurator

        return HabitatConfigurator(config=config).create_habitat_analysis()

    @staticmethod
    def _subjects_from_input(X: Optional[Sequence[str]]) -> Optional[List[str]]:
        """Normalize the estimator's optional subject-id input contract."""
        if X is None:
            return None
        if isinstance(X, (str, bytes)) or not isinstance(X, Sequence):
            raise HABITAPIError(
                "X must be None or a sequence of subject identifier strings."
            )
        subjects = list(X)
        if not all(isinstance(subject, str) and subject for subject in subjects):
            raise HABITAPIError(
                "Every X subject identifier must be a non-empty string."
            )
        return subjects
