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
PlotManager Module
Handles the orchestration of all visualization tasks for machine learning workflows.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from .plotting import Plotter
from habit.utils.log_utils import get_module_logger
from habit.utils.random_utils import resolve_random_state

class PlotManager:
    def __init__(self, config: Any, output_dir: str):
        """
        Initialize PlotManager.
        
        Args:
            config: MLConfig object.
            output_dir: Output directory path.
        """
        self.config = config
        self.output_dir = output_dir
        self.plotter = Plotter(output_dir)
        self.logger = get_module_logger('evaluation.plot_manager')
        
        # Use direct attribute access (Expects Pydantic MLConfig or ModelComparisonConfig)
        # Handle both dict and Pydantic model for backward compatibility
        if isinstance(config, dict):
            # If config is a dict, try to access visualization field
            if 'visualization' in config:
                self.viz_config = config['visualization']
            else:
                # Create a default visualization config structure
                from ..config_schemas import VisualizationConfig
                self.viz_config = VisualizationConfig()
            self.is_visualize = config.get('is_visualize', True)
        else:
            # Pydantic model - use attribute access
            self.viz_config = config.visualization
            self.is_visualize = getattr(config, 'is_visualize', True)
        
        # Default plots
        self.default_plots = ['roc', 'dca', 'calibration', 'pr', 'confusion']
        
        # Get plot_types from Pydantic object
        if hasattr(self.viz_config, 'plot_types'):
            self.plot_types = self.viz_config.plot_types
        else:
            self.plot_types = self.default_plots

        # Explanation-figure knobs; absent for dict configs and for the
        # comparison workflow, which does not produce explanation figures.
        self.explainability = getattr(self.viz_config, 'explainability', None)

    @property
    def _shap_plot_requested(self) -> bool:
        """Whether any figure in this run needs SHAP attributions."""
        return any(
            name in self.plot_types
            for name in ('shap', 'shap_dependence', 'shap_waterfall')
        )

    def _explainability_option(self, name: str, default: Any) -> Any:
        """
        Read one explanation setting, falling back to its default.

        Args:
            name: Attribute name on the explainability config block.
            default: Value used when the config block is unavailable.

        Returns:
            Any: The configured value or ``default``.
        """
        if self.explainability is None:
            return default
        return getattr(self.explainability, name, default)

    def run_workflow_plots(self, results: Dict[str, Any], prefix: str = "", 
                          X_test: Optional[pd.DataFrame] = None, dataset_type: str = 'test'):
        """
        Main entry point for generating all configured plots.
        
        Args:
            results: Dictionary containing model results (y_true, y_prob, y_pred, etc.)
            prefix: Filename prefix (e.g., 'standard_' or 'kfold_')
            X_test: Test features (needed for SHAP)
            dataset_type: 'train', 'test', or 'raw' (for kfold aggregated results)
        """
        if not self.is_visualize:
            self.logger.info("Visualization is disabled in config.")
            return

        self.logger.info(f"Generating evaluation plots for {prefix} workflow ({dataset_type} set)...")
        
        # Prepare multi-model data for curves
        plotting_data = {}
        for m_name, res in results.items():
            # Support different result structures:
            # 1. Standard workflow with train/test split: res['train'] or res['test']
            # 2. KFold aggregated results: res['raw']
            # 3. Legacy format: direct res
            if dataset_type in res:
                data = res[dataset_type]
            elif 'raw' in res:
                data = res['raw']
            else:
                data = res
                
            if 'y_true' in data and 'y_prob' in data:
                plotting_data[m_name] = (np.array(data['y_true']), np.array(data['y_prob']))

        if not plotting_data:
            self.logger.warning("No data available for plotting.")
            return

        # 1. Multi-model curves
        self._generate_curve_plots(plotting_data, prefix)

        # 2. Individual model plots
        self._generate_individual_plots(results, prefix, X_test, dataset_type)

    def _generate_curve_plots(self, plotting_data: Dict, prefix: str):
        """ROC, DCA, Calibration, PR Curves"""
        title_suffix = prefix.replace('_', ' ').title()
        
        if 'roc' in self.plot_types:
            self.plotter.plot_roc_v2(plotting_data, save_name=f'{prefix}roc_curve.pdf', title=f'{title_suffix} ROC')
        
        if 'dca' in self.plot_types:
            self.plotter.plot_dca_v2(plotting_data, save_name=f'{prefix}decision_curve.pdf', title=f'{title_suffix} DCA')
            
        if 'calibration' in self.plot_types:
            self.plotter.plot_calibration_v2(plotting_data, save_name=f'{prefix}calibration_curve.pdf', title=f'{title_suffix} Calibration')
            
        if 'pr' in self.plot_types:
            self.plotter.plot_pr_curve(plotting_data, save_name=f'{prefix}pr_curve.pdf', title=f'{title_suffix} PR Curve')

    def _generate_individual_plots(self, results: Dict, prefix: str, 
                                  X_test: Optional[pd.DataFrame], dataset_type: str = 'test'):
        """Confusion Matrix and SHAP"""
        for m_name, res in results.items():
            # Extract data based on dataset_type
            if dataset_type in res:
                data = res[dataset_type]
            elif 'raw' in res:
                data = res['raw']
            else:
                data = res
            
            # Confusion Matrix
            if 'confusion' in self.plot_types and 'y_pred' in data:
                self.plotter.plot_confusion_matrix(
                    np.array(data['y_true']), 
                    np.array(data['y_pred']), 
                    save_name=f'{prefix}{m_name}_confusion_matrix.pdf',
                    title=f'{m_name} Confusion Matrix'
                )

            # SHAP-based explanation figures
            if self._shap_plot_requested and 'pipeline' in res and X_test is not None:
                self._generate_shap_plots(
                    model_name=m_name,
                    trained_estimator=res['pipeline'],
                    X_test=X_test,
                    prefix=prefix,
                )

            # Permutation importance on the raw input features
            if (
                'permutation' in self.plot_types
                and 'pipeline' in res
                and X_test is not None
                and 'y_true' in data
            ):
                self._generate_permutation_importance(
                    model_name=m_name,
                    trained_estimator=res['pipeline'],
                    X_test=X_test,
                    y_true=np.array(data['y_true']),
                    prefix=prefix,
                )

    def _generate_shap_plots(
        self,
        model_name: str,
        trained_estimator: Any,
        X_test: pd.DataFrame,
        prefix: str,
    ) -> None:
        """
        Render every requested SHAP figure from a single attribution pass.

        Args:
            model_name: Model whose predictions are being explained.
            trained_estimator: Fitted pipeline saved in the workflow results.
            X_test: Raw feature frame before the pipeline transforms it.
            prefix: Filename prefix identifying the split and workflow.
        """
        try:
            self.logger.info(f"Computing SHAP values for {model_name}...")
            model_obj, X_for_shap, feature_names = self._resolve_shap_inputs(
                trained_estimator, X_test
            )
            self.logger.debug(
                "Final SHAP input shape=%s, n_features=%d",
                X_for_shap.shape if hasattr(X_for_shap, "shape") else "N/A",
                len(feature_names),
            )
            shap_values, expected_value = self.plotter.compute_shap_explanation(
                model_obj, X_for_shap
            )
        except Exception as e:
            import traceback
            self.logger.warning(f"Could not compute SHAP for {model_name}: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return

        if 'shap' in self.plot_types:
            self._try_plot(
                f"SHAP summary for {model_name}",
                self.plotter.plot_shap,
                model_obj,
                X_for_shap,
                feature_names=feature_names,
                save_name=f'{prefix}{model_name}_shap.pdf',
                shap_values=shap_values,
            )

        if 'shap_dependence' in self.plot_types:
            self._try_plot(
                f"SHAP dependence for {model_name}",
                self.plotter.plot_shap_dependence,
                X_for_shap,
                feature_names=feature_names,
                shap_values=shap_values,
                top_k=self._explainability_option('shap_dependence_top_k', 3),
                save_name=f'{prefix}{model_name}_shap_dependence.pdf',
            )

        if 'shap_waterfall' in self.plot_types:
            self._try_plot(
                f"SHAP waterfall for {model_name}",
                self.plotter.plot_shap_waterfall,
                X_for_shap,
                feature_names=feature_names,
                shap_values=shap_values,
                expected_value=expected_value,
                n_samples=self._explainability_option('shap_waterfall_samples', 3),
                save_name=f'{prefix}{model_name}_shap_waterfall.pdf',
            )

    def _generate_permutation_importance(
        self,
        model_name: str,
        trained_estimator: Any,
        X_test: pd.DataFrame,
        y_true: np.ndarray,
        prefix: str,
    ) -> None:
        """
        Compute and render permutation importance for the raw input features.

        The whole pipeline is scored, not just its final estimator, so the
        reported values describe how much the run's performance depends on each
        input column as supplied by the user.

        Args:
            model_name: Model being explained.
            trained_estimator: Fitted pipeline accepting ``X_test`` directly.
            X_test: Raw feature frame for the split being explained.
            y_true: Ground-truth labels aligned with ``X_test``.
            prefix: Filename prefix identifying the split and workflow.
        """
        from ..evaluation.explainability import compute_permutation_importance

        try:
            self.logger.info(
                f"Computing permutation importance for {model_name}..."
            )
            importance = compute_permutation_importance(
                trained_estimator,
                X_test,
                y_true,
                scoring=self._explainability_option(
                    'permutation_scoring', 'roc_auc'
                ),
                n_repeats=self._explainability_option('permutation_repeats', 10),
                random_state=resolve_random_state(
                    self._explainability_option('permutation_random_state', None),
                    getattr(self.config, 'random_state', None),
                ),
            )
        except Exception as e:
            import traceback
            self.logger.warning(
                f"Could not compute permutation importance for {model_name}: {e}"
            )
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return

        csv_path = os.path.join(
            self.output_dir, f'{prefix}{model_name}_permutation_importance.csv'
        )
        importance.to_csv(csv_path, index=False)
        self.logger.info(f"Permutation importance table saved: {csv_path}")

        self._try_plot(
            f"permutation importance for {model_name}",
            self.plotter.plot_permutation_importance,
            importance,
            save_name=f'{prefix}{model_name}_permutation_importance.pdf',
            title=f'{model_name} Permutation Importance',
            top_k=self._explainability_option('permutation_top_k', 20),
        )

    def _try_plot(self, description: str, plot_func: Any, *args: Any, **kwargs: Any) -> None:
        """
        Run one plotting call, logging failures instead of aborting the run.

        A single unrenderable figure must never lose an otherwise complete
        training run, so every explanation figure is isolated.

        Args:
            description: Human-readable figure description used in logs.
            plot_func: Plotter method to invoke.
            *args: Positional arguments forwarded to ``plot_func``.
            **kwargs: Keyword arguments forwarded to ``plot_func``.
        """
        try:
            plot_func(*args, **kwargs)
            self.logger.info(f"Generated {description}.")
        except Exception as e:
            import traceback
            self.logger.warning(f"Could not generate {description}: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def _resolve_shap_inputs(
        self, trained_estimator: Any, X_input: pd.DataFrame
    ) -> Tuple[Any, np.ndarray, List[str]]:
        """
        Resolve SHAP-ready model and feature matrix from a trained estimator.

        Why this adapter exists:
        - Standard workflows may save a plain sklearn Pipeline.
        - When post calibration is enabled, workflows save CalibratedClassifierCV.
        - SHAP should explain the feature model (the base estimator), not the
          calibration wrapper itself. Therefore, we unwrap calibrators to retrieve
          the underlying fitted estimator and transform X consistently.

        Args:
            trained_estimator: Trained object saved in workflow results.
            X_input: Raw input features (pre-pipeline transform).

        Returns:
            Tuple[Any, np.ndarray, List[str]]:
                - model_obj: Model object for SHAP explainer.
                - X_for_shap: Transformed feature array for SHAP.
                - feature_names: Feature names aligned to X_for_shap columns.

        Raises:
            ValueError: If estimator structure is unsupported.
        """
        base_estimator = self._unwrap_for_shap(trained_estimator)
        pipeline = self._extract_pipeline_from_estimator(base_estimator)
        if pipeline is None:
            raise ValueError(
                f"Unsupported estimator for SHAP: {type(base_estimator).__name__}. "
                "Expected a sklearn Pipeline or a calibrator wrapping a Pipeline."
            )

        self.logger.debug(f"Original X shape: {X_input.shape}")
        X_transformed = X_input.copy()
        for step_name, transformer in pipeline.steps[:-1]:
            # Samplers (e.g. ResamplingStep / imblearn samplers) are training-only
            # steps: they expose fit_resample and are bypassed at predict/transform
            # time by imblearn. SHAP explains inference-time behavior, so skip any
            # step that does not implement transform to mirror that runtime
            # semantics (otherwise calling .transform on a sampler raises).
            if not hasattr(transformer, "transform"):
                self.logger.debug(
                    "Skipping non-transform step '%s' for SHAP (sampler/train-only).",
                    step_name,
                )
                continue
            X_transformed = transformer.transform(X_transformed)
            self.logger.debug(
                "After '%s' step: shape=%s",
                step_name,
                X_transformed.shape if hasattr(X_transformed, "shape") else "N/A",
            )

        if hasattr(X_transformed, "columns"):
            feature_names = list(X_transformed.columns)
            X_for_shap = X_transformed.values
        else:
            selector_after = pipeline.named_steps.get("selector_after")
            if selector_after is not None and hasattr(selector_after, "selected_features_"):
                feature_names = list(selector_after.selected_features_)
            else:
                feature_count = X_transformed.shape[1]
                feature_names = [f"Feature_{i}" for i in range(feature_count)]
            X_for_shap = np.asarray(X_transformed)

        model_obj = pipeline.named_steps["model"]
        return model_obj, X_for_shap, feature_names

    def _extract_pipeline_from_estimator(self, estimator: Any) -> Any:
        """
        Extract sklearn Pipeline from estimator if available.

        Args:
            estimator: Candidate estimator.

        Returns:
            Any: Pipeline object or None when unavailable.
        """
        if hasattr(estimator, "named_steps") and hasattr(estimator, "steps"):
            return estimator
        return None

    def _unwrap_for_shap(self, estimator: Any) -> Any:
        """
        Unwrap wrappers (e.g., CalibratedClassifierCV) to base estimator.

        For post-calibrated models we explain the underlying fitted estimator.
        This preserves feature-level interpretability and avoids relying on
        calibrator internals during SHAP generation.

        Args:
            estimator: Trained estimator possibly wrapped by calibrator.

        Returns:
            Any: Unwrapped estimator suitable for pipeline extraction.
        """
        class_name = type(estimator).__name__
        if class_name != "CalibratedClassifierCV":
            return estimator

        # Newer sklearn APIs expose `estimator`; older versions may use
        # `base_estimator`. Prefer a fitted calibrated classifier when present.
        calibrated_list = getattr(estimator, "calibrated_classifiers_", None)
        if calibrated_list:
            first_calibrator = calibrated_list[0]
            fitted_estimator = getattr(first_calibrator, "estimator", None)
            if fitted_estimator is not None:
                return fitted_estimator
            fitted_estimator = getattr(first_calibrator, "base_estimator", None)
            if fitted_estimator is not None:
                return fitted_estimator

        direct_estimator = getattr(estimator, "estimator", None)
        if direct_estimator is not None:
            return direct_estimator
        direct_estimator = getattr(estimator, "base_estimator", None)
        if direct_estimator is not None:
            return direct_estimator

        return estimator
