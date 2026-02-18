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

            # SHAP
            if 'shap' in self.plot_types and 'pipeline' in res and X_test is not None:
                try:
                    trained_estimator = res['pipeline']
                    self.logger.info(f"Generating SHAP plot for {m_name}...")
                    model_obj, X_for_shap, feature_names = self._resolve_shap_inputs(
                        trained_estimator, X_test
                    )
                    self.logger.debug(
                        "Final SHAP input shape=%s, n_features=%d",
                        X_for_shap.shape if hasattr(X_for_shap, "shape") else "N/A",
                        len(feature_names),
                    )
                    
                    self.plotter.plot_shap(
                        model_obj, 
                        X_for_shap, 
                        feature_names=feature_names,
                        save_name=f'{prefix}{m_name}_shap.pdf'
                    )
                    self.logger.info(f"SHAP plot saved: {prefix}{m_name}_shap.pdf")
                except Exception as e:
                    import traceback
                    self.logger.warning(f"Could not generate SHAP for {m_name}: {e}")
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
