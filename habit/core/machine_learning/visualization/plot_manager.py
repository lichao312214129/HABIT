"""
PlotManager Module
Handles the orchestration of all visualization tasks for machine learning workflows.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
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
                    pipeline = res['pipeline']
                    model_obj = pipeline.named_steps['model']
                    
                    self.logger.info(f"Generating SHAP plot for {m_name}...")
                    self.logger.debug(f"Original X shape: {X_test.shape}")
                    
                    # Transform X_test through the pipeline up to (but not including) the model
                    # This ensures we use the same features that the model was trained on
                    X_transformed = X_test.copy()
                    for step_name, transformer in pipeline.steps[:-1]:  # Exclude the final 'model' step
                        X_transformed = transformer.transform(X_transformed)
                        self.logger.debug(f"After '{step_name}' step: shape={X_transformed.shape if hasattr(X_transformed, 'shape') else 'N/A'}")
                    
                    # Get feature names after transformation
                    if hasattr(X_transformed, 'columns'):
                        feature_names = list(X_transformed.columns)
                        X_for_shap = X_transformed.values
                    else:
                        # If X_transformed is numpy array, try to get feature names from the last selector
                        selector_after = pipeline.named_steps.get('selector_after')
                        if selector_after and hasattr(selector_after, 'selected_features_'):
                            feature_names = selector_after.selected_features_
                        else:
                            feature_names = [f'Feature_{i}' for i in range(X_transformed.shape[1])]
                        X_for_shap = X_transformed
                    
                    self.logger.debug(f"Final X shape for SHAP: {X_for_shap.shape}, n_features={len(feature_names)}")
                    self.logger.debug(f"Feature names: {feature_names}")
                    
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
