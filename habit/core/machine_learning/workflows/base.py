"""
Abstract base class for all machine-learning workflows.

The base class owns infrastructure that every workflow needs:

* configuration validation,
* logging,
* shared collaborators (data manager, pipeline builder, plot manager),
* a :class:`Resampler` adapter,
* a pre-built :class:`RunnerContext` that runners can be constructed from.

Concrete workflows stay thin: they decide which runner to call and how to
glue the result through the reporting layer.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import pandas as pd

from habit.core.common.config_base import ConfigAccessor
from habit.core.common.config_validator import ConfigValidator
from habit.utils.io_utils import save_csv, save_json
from habit.utils.log_utils import LoggerManager, get_module_logger, setup_logger

from ..config_schemas import MLConfig
from ..data_manager import DataManager
from ..pipeline_utils import PipelineBuilder
from ..resampling import Resampler
from ..runners.context import RunnerContext
from ..visualization.plot_manager import PlotManager


class BaseWorkflow(ABC):
    """
    Abstract base class for ML workflows.

    Handles infrastructure (logging, data loading, results persistence) and
    builds a :class:`RunnerContext` that subclasses pass to their runner.
    """

    def __init__(
        self,
        config: Union[MLConfig, Dict[str, Any]],
        module_name: str,
    ) -> None:
        """
        Parameters
        ----------
        config:
            ``MLConfig`` Pydantic object or a dict (validated and converted).
        module_name:
            Name of the workflow module (used for log/output prefixes).
        """
        self.module_name = module_name
        self.config_obj = self._validate_config(config, module_name)

        # Use ConfigAccessor for unified access; keep dict for compatibility.
        self.config_accessor = ConfigAccessor(self.config_obj)
        self.config = self.config_obj.to_dict()

        # Output directory.
        self.output_dir = getattr(
            self.config_obj, "output", f"./results/{module_name}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Logger - reuse existing one when LoggerManager is already configured.
        manager = LoggerManager()
        if manager.get_log_file() is not None:
            self.logger = get_module_logger(module_name)
        else:
            self.logger = setup_logger(
                module_name, self.output_dir, f"{module_name}.log"
            )

        # Common collaborators.
        self.data_manager = DataManager(self.config_obj, self.logger)
        self.plot_manager = PlotManager(self.config_obj, self.output_dir)
        self.pipeline_builder = PipelineBuilder(self.config_obj, self.output_dir)
        self.random_state = getattr(self.config_obj, "random_state", 42)

        # Resampler adapter - replaces the previous private
        # ``_train_with_optional_sampling`` method on this class.  The method
        # is kept on the workflow for backward compatibility but now simply
        # delegates to the resampler.
        sampling_cfg = getattr(self.config_obj, "sampling", None)
        self.resampler = Resampler(
            sampling_cfg=sampling_cfg,
            random_state=self.random_state,
            logger=self.logger,
        )

        # Pre-built runner context shared by every runner this workflow uses.
        self.runner_context = RunnerContext(
            data_manager=self.data_manager,
            pipeline_builder=self.pipeline_builder,
            resampler=self.resampler,
            logger=self.logger,
            config=self.config_obj,
        )

        # Results storage for backward compatibility with existing callers.
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(
        config: Union[MLConfig, Dict[str, Any]], module_name: str
    ) -> MLConfig:
        """Coerce caller-provided config into a validated ``MLConfig``."""
        if isinstance(config, MLConfig):
            return config
        if isinstance(config, dict):
            try:
                return ConfigValidator.validate_dict(config, MLConfig, strict=True)
            except Exception as exc:
                raise ValueError(
                    f"Configuration validation failed for {module_name}: {exc}. "
                    "Please ensure your configuration matches the MLConfig schema. "
                    "Use MLConfig.from_file() or ConfigValidator.validate_and_load() "
                    "to load configuration."
                ) from exc
        raise TypeError(
            f"Invalid configuration type for {module_name}: expected MLConfig or "
            f"dict, got {type(config)}"
        )

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Common data loading entry point used by K-Fold."""
        self.logger.info("Loading and preparing data...")
        self.data_manager.load_data()
        X = self.data_manager.data.drop(columns=[self.data_manager.label_col])
        y = self.data_manager.data[self.data_manager.label_col]
        return X, y

    def _resample_training_data(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Backward-compat wrapper - delegates to the :class:`Resampler`."""
        return self.resampler.resample(X_train, y_train)

    def _train_with_optional_sampling(
        self, estimator: Any, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """
        Backward-compat wrapper for older callers / tests.

        New code should prefer ``self.resampler.fit_with_resampling`` or use
        the runner layer.
        """
        return self.resampler.fit_with_resampling(estimator, X_train, y_train)

    @abstractmethod
    def run(self) -> None:
        """Main entry point implemented by concrete workflows."""
        raise NotImplementedError

    def _save_common_results(self, summary_df: pd.DataFrame, prefix: str = "") -> None:
        """Save common result artefacts (used by legacy code paths)."""
        save_json(
            self.results, os.path.join(self.output_dir, f"{prefix}results.json")
        )
        save_csv(summary_df, os.path.join(self.output_dir, f"{prefix}summary.csv"))
