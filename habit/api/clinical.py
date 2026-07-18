"""High-level, sklearn-style APIs for clinical habitat research workflows.

These classes intentionally compose the existing configuration-driven public
workflow runners instead of replacing them.  This keeps the CLI and its YAML
contracts stable while giving notebook and service users a concise
``fit``/``transform``/``predict`` interaction model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.exceptions import HABITAPIError, NotFittedError

if TYPE_CHECKING:
    from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
    from habit.core.preprocessing.config_schemas import PreprocessingConfig

__all__ = [
    "Cohort",
    "PreparedCohort",
    "HabitatResult",
    "ClinicalPreprocessor",
    "HabitatSegmenter",
]


@dataclass(frozen=True)
class Cohort:
    """A named imaging cohort represented by a HABIT-compatible data directory.

    The object deliberately keeps the public boundary simple: HABIT does not
    replace general image I/O or preprocessing libraries.  It records the
    prepared cohort root that HABIT's existing workflow configuration expects,
    while optional metadata remains available for notebooks, applications, and
    reporting code.
    """

    data_dir: Union[str, Path]
    name: Optional[str] = None
    subject_ids: Optional[Sequence[str]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize cohort metadata without requiring the directory to exist yet."""
        normalized_data_dir = Path(self.data_dir)
        normalized_subjects: Optional[tuple[str, ...]]
        if self.subject_ids is None:
            normalized_subjects = None
        else:
            normalized_subjects = tuple(self.subject_ids)
            if not normalized_subjects or any(
                not isinstance(subject, str) or not subject.strip()
                for subject in normalized_subjects
            ):
                raise HABITAPIError(
                    "subject_ids must be omitted or contain non-empty strings."
                )
            if len(set(normalized_subjects)) != len(normalized_subjects):
                raise HABITAPIError("subject_ids must not contain duplicates.")

        object.__setattr__(self, "data_dir", normalized_data_dir)
        object.__setattr__(self, "subject_ids", normalized_subjects)
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )

    @classmethod
    def from_directory(
        cls,
        data_dir: Union[str, Path],
        *,
        name: Optional[str] = None,
        subject_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Cohort":
        """Create a cohort reference from an existing or planned data directory."""
        return cls(
            data_dir=data_dir,
            name=name,
            subject_ids=subject_ids,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class PreparedCohort(Cohort):
    """A cohort produced by :class:`ClinicalPreprocessor`."""

    preprocessing_result: Optional[WorkflowResult[None]] = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class HabitatResult:
    """Typed clinical result returned by :class:`HabitatSegmenter`.

    Attributes:
        table: Per-subject habitat result table returned by the existing HABIT
            workflow.
        cohort: Cohort used for this training or prediction run.
        workflow_result: Full artifact and provenance result from the stable
            configuration-driven public API.
        pipeline_path: Fitted habitat pipeline when the run created or used one.
    """

    table: pd.DataFrame
    cohort: Cohort
    workflow_result: WorkflowResult[pd.DataFrame]
    pipeline_path: Optional[Path] = None

    @property
    def output_dir(self) -> Optional[Path]:
        """Return the directory containing this run's artifacts."""
        return self.workflow_result.output_dir


def _cohort_from_input(
    cohort: Optional[Union[Cohort, str, Path]],
    *,
    default_data_dir: Union[str, Path],
) -> Cohort:
    """Normalize optional high-level cohort input at an API boundary."""
    if cohort is None:
        return Cohort.from_directory(default_data_dir)
    if isinstance(cohort, Cohort):
        return cohort
    if isinstance(cohort, (str, Path)):
        return Cohort.from_directory(cohort)
    raise HABITAPIError("cohort must be a Cohort, string path, Path, or None.")


class ClinicalPreprocessor(BaseEstimator):
    """Apply an established clinical preprocessing workflow to a cohort.

    ``config`` remains the complete, reproducible definition of the workflow.
    The estimator interface removes the need for users to interact with core
    configurators or individual image-processing parameters during routine
    studies.  Existing ``habit preprocess -c <yaml>`` behavior is unchanged.

    This class intentionally does not inherit ``TransformerMixin``. Sklearn's
    set-output wrapper would otherwise force ``fit_transform(X)`` to require a
    positional ``X``, which conflicts with config-driven calls that omit the
    cohort and use ``config['data_dir']`` alone.
    """

    def __init__(
        self,
        config: Union["PreprocessingConfig", Mapping[str, Any]],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Store clone-safe public parameters without importing core workflows."""
        self.config = config
        self.logger = logger

    def fit(
        self,
        X: Optional[Union[Cohort, str, Path]] = None,
        y: Optional[Any] = None,
    ) -> "ClinicalPreprocessor":
        """Validate the preprocessing specification and selected input cohort."""
        del y
        from habit.core.preprocessing.config_schemas import PreprocessingConfig

        config = coerce_config(self.config, PreprocessingConfig)
        cohort = _cohort_from_input(X, default_data_dir=config.data_dir)
        self.config_ = self._with_data_dir(config, cohort.data_dir)
        self.input_cohort_ = cohort
        self._is_fitted_ = True
        return self

    def transform(
        self,
        X: Optional[Union[Cohort, str, Path]] = None,
    ) -> PreparedCohort:
        """Run preprocessing and return the resulting cohort directory."""
        self._require_fitted()
        cohort = _cohort_from_input(X, default_data_dir=self.config_.data_dir)
        config = self._with_data_dir(self.config_, cohort.data_dir)

        from habit.api.preprocessing import run_preprocess

        result = run_preprocess(config, logger=self.logger)
        # BatchProcessor writes images/masks under ``<out_dir>/processed_images``.
        # Downstream habitat workflows expect that directory as ``data_dir``.
        output_root = Path(result.output_dir or config.out_dir)
        prepared = PreparedCohort(
            data_dir=output_root / "processed_images",
            name=cohort.name,
            subject_ids=cohort.subject_ids,
            metadata=cohort.metadata,
            preprocessing_result=result,
        )
        self.result_ = result
        self.output_cohort_ = prepared
        return prepared

    def fit_transform(
        self,
        X: Optional[Union[Cohort, str, Path]] = None,
        y: Optional[Any] = None,
        **fit_params: Any,
    ) -> PreparedCohort:
        """Fit and transform, optionally using only ``config['data_dir']``.

        Args:
            X: Optional cohort or directory. When omitted, ``config['data_dir']``
                is used.
            y: Unused; accepted for sklearn API compatibility.
            **fit_params: Unsupported; raising keeps the public contract strict.

        Returns:
            Prepared cohort whose ``data_dir`` points at ``processed_images``.
        """
        if fit_params:
            unexpected = ", ".join(sorted(fit_params))
            raise HABITAPIError(
                f"Unsupported ClinicalPreprocessor fit parameters: {unexpected}."
            )
        return self.fit(X, y).transform(X)

    @staticmethod
    def _with_data_dir(
        config: "PreprocessingConfig",
        data_dir: Union[str, Path],
    ) -> "PreprocessingConfig":
        """Return a copy so caller-owned configuration is never mutated."""
        return config.model_copy(update={"data_dir": str(data_dir)})

    def _require_fitted(self) -> None:
        """Raise sklearn's standard error before a fitted-state operation."""
        if not getattr(self, "_is_fitted_", False):
            raise NotFittedError(
                "ClinicalPreprocessor is not fitted. Call fit or fit_transform first."
            )


class HabitatSegmenter(BaseEstimator, TransformerMixin):
    """Train and apply a cohort-level habitat segmentation model.

    The class exposes the familiar sklearn lifecycle while preserving HABIT's
    established train/predict workflow semantics.  ``fit`` trains from the
    configured cohort; ``predict`` or ``transform`` applies the saved pipeline
    to another cohort.  The CLI continues to use its existing YAML entry point.
    """

    def __init__(
        self,
        config: Union["HabitatAnalysisConfig", Mapping[str, Any]],
        prediction_output_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Store high-level parameters without constructing processing services."""
        self.config = config
        self.prediction_output_dir = prediction_output_dir
        self.logger = logger

    def fit(
        self,
        X: Optional[Union[Cohort, str, Path]] = None,
        y: Optional[Any] = None,
    ) -> "HabitatSegmenter":
        """Fit a habitat model on the configured or supplied training cohort."""
        del y
        config, cohort = self._training_config(X)

        from habit.api.habitat import run_habitat_analysis

        workflow_result = run_habitat_analysis(config, logger=self.logger)
        pipeline_path = workflow_result.artifacts.get(
            "pipeline",
            Path(config.out_dir) / "habitat_pipeline.pkl",
        )
        self.config_ = config
        self.training_cohort_ = cohort
        self.pipeline_path_ = Path(pipeline_path)
        self.result_ = self._to_result(workflow_result, cohort, self.pipeline_path_)
        self._is_fitted_ = True
        return self

    def transform(
        self,
        X: Optional[Union[Cohort, str, Path]] = None,
    ) -> HabitatResult:
        """Apply the fitted habitat model to a cohort."""
        return self.predict(X)

    def predict(
        self,
        X: Optional[Union[Cohort, str, Path]] = None,
    ) -> HabitatResult:
        """Predict habitat assignments for a compatible cohort."""
        self._require_fitted()
        cohort = _cohort_from_input(X, default_data_dir=self.config_.data_dir)
        config = self._prediction_config(cohort)

        from habit.api.habitat import run_habitat_analysis

        workflow_result = run_habitat_analysis(config, logger=self.logger)
        result = self._to_result(workflow_result, cohort, self.pipeline_path_)
        self.prediction_result_ = result
        return result

    def fit_transform(
        self,
        X: Optional[Union[Cohort, str, Path]] = None,
        y: Optional[Any] = None,
        **fit_params: Any,
    ) -> HabitatResult:
        """Fit once and return the training habitat result without prediction."""
        if fit_params:
            unexpected = ", ".join(sorted(fit_params))
            raise HABITAPIError(
                f"Unsupported HabitatSegmenter fit parameters: {unexpected}."
            )
        self.fit(X, y)
        return self.result_

    def _training_config(
        self,
        cohort_input: Optional[Union[Cohort, str, Path]],
    ) -> tuple["HabitatAnalysisConfig", Cohort]:
        """Resolve a train-mode config without mutating user-owned input.

        When a cohort (or path) is supplied, ``data_dir`` may be omitted from the
        constructor config and is taken from that cohort instead.
        """
        from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig

        config_input: Union["HabitatAnalysisConfig", Mapping[str, Any]] = self.config
        if isinstance(config_input, Mapping):
            config_mapping = dict(config_input)
            if "data_dir" not in config_mapping:
                if cohort_input is None:
                    raise HABITAPIError(
                        "HabitatSegmenter requires config['data_dir'] when no "
                        "training cohort is passed to fit/fit_transform."
                    )
                provisional = _cohort_from_input(cohort_input, default_data_dir=".")
                config_mapping["data_dir"] = str(provisional.data_dir)
            config = coerce_config(config_mapping, HabitatAnalysisConfig)
        else:
            config = coerce_config(config_input, HabitatAnalysisConfig)

        if config.run_mode != "train":
            raise HABITAPIError(
                "HabitatSegmenter.fit requires a train-mode HabitatAnalysisConfig."
            )
        cohort = _cohort_from_input(cohort_input, default_data_dir=config.data_dir)
        return (
            config.model_copy(update={"data_dir": str(cohort.data_dir)}),
            cohort,
        )

    def _prediction_config(self, cohort: Cohort) -> "HabitatAnalysisConfig":
        """Build an explicit prediction config from the fitted model contract."""
        output_dir = (
            Path(self.prediction_output_dir)
            if self.prediction_output_dir is not None
            else Path(self.config_.out_dir) / "predict"
        )
        return self.config_.model_copy(
            update={
                "data_dir": str(cohort.data_dir),
                "out_dir": str(output_dir),
                "run_mode": "predict",
                "pipeline_path": str(self.pipeline_path_),
            }
        )

    @staticmethod
    def _to_result(
        workflow_result: WorkflowResult[pd.DataFrame],
        cohort: Cohort,
        pipeline_path: Optional[Path],
    ) -> HabitatResult:
        """Translate the general workflow result into the clinical result type."""
        if workflow_result.data is None:
            raise HABITAPIError("Habitat analysis did not return a result table.")
        return HabitatResult(
            table=workflow_result.data,
            cohort=cohort,
            workflow_result=workflow_result,
            pipeline_path=pipeline_path,
        )

    def _require_fitted(self) -> None:
        """Raise sklearn's standard error before prediction."""
        if not getattr(self, "_is_fitted_", False):
            raise NotFittedError(
                "HabitatSegmenter is not fitted. Call fit or fit_transform first."
            )
