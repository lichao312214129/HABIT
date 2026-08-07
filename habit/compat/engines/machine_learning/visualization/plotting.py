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
Plotting Module
Provides various evaluation chart plotting functions
"""

import os
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd

from habit.utils.optional_deps import require

# matplotlib and seaborn are OPTIONAL dependencies (habitat-analysis[viz]).
# This module draws its own diagnostic figures at module scope, so the gate
# stays at module scope too: the import failure then names the extra instead
# of raising a bare ModuleNotFoundError.
_VIZ_PURPOSE = "machine-learning evaluation figures (ROC, DCA, calibration, SHAP)"
plt = require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)
sns = require("seaborn", extra="viz", purpose=_VIZ_PURPOSE)
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from sklearn.calibration import calibration_curve  # Calibration curve related
from ..evaluation.metrics import calculate_net_benefit
from habit.utils.visualization_utils import process_shap_explanation
from habit.utils.font_config import PUBLICATION_FONT, setup_publication_font

from habit.utils.log_utils import get_module_logger
logger = get_module_logger(__name__)

class Plotter:
    def __init__(self, output_dir: str, dpi: int = 600) -> None:
        """
        Initialize the plotter
        
        Args:
            output_dir (str): Output directory path
            dpi (int): Resolution for non-PDF format images
        """
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style with publication font (platform-aware fallback)
        setup_publication_font()
        # plt.style.use('seaborn')
        # sns.set_context("paper", font_scale=1.2)

    def _build_model_curve_styles(self, model_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Build high-contrast styles for multi-model line plots.

        This style generator is intentionally deterministic so the same model order
        always gets the same visual encoding within one plotting call.

        Design choices:
        - Use color-blind friendly palettes first.
        - If model count exceeds palette size, cycle line styles and markers.
        - Keep baseline curves (e.g., Treat All / Treat None) separate in callers.

        Args:
            model_names: Ordered model names to style.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from model name to style dict.
        """
        # Primary high-contrast palettes (color-blind friendly).
        palette = (
            sns.color_palette("colorblind", 10)
            + sns.color_palette("tab10", 10)
            + sns.color_palette("Set2", 8)
        )
        line_styles = ["-", "--", "-.", ":"]
        markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

        styles: Dict[str, Dict[str, Any]] = {}
        palette_size = len(palette)
        for idx, model_name in enumerate(model_names):
            color = palette[idx % palette_size]
            linestyle = line_styles[(idx // palette_size) % len(line_styles)]
            marker = markers[(idx // (palette_size * len(line_styles))) % len(markers)]

            styles[model_name] = {
                "color": color,
                "linestyle": linestyle,
                "marker": marker,
            }
        return styles
        
    def plot_roc_v2(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]], save_name: str = 'ROC.pdf', title: str = 'test') -> None:
        """
        Plot ROC curves for a single dataset (optimized version)
        
        Args:
            models_data: Dictionary with model names as keys and (y_true, y_pred_proba) tuples as values
            save_name: Name of the file to save the plot
            title: Data type for title display ('train' or 'test')
        """
        # Create figure - optimized for SCI journal requirements (single column)
        plt.figure(figsize=(5, 5))
        model_styles = self._build_model_curve_styles(list(models_data.keys()))
        
        # Plot ROC curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = np.trapz(tpr, fpr)
            style = model_styles[model_name]
            plt.plot(
                fpr,
                tpr,
                label=f'{model_name} (AUC = {auc:.2f})',
                linewidth=1.8,
                color=style["color"],
                linestyle=style["linestyle"],
            )
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.ylabel('True Positive Rate', fontsize=10, fontfamily=PUBLICATION_FONT)
        
        # Set title based on data type
        plt.title(title, fontsize=11, fontfamily=PUBLICATION_FONT)
        plt.legend(loc="lower right", fontsize=9)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        # plt.gca().set_facecolor('white')
        
        # Only show left and bottom spines and set their width to 1.5
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # Save figure
        plt.tight_layout()
        
        # 根据文件扩展名决定是否应用压缩和DPI设置
        file_ext = os.path.splitext(save_name)[1].lower()
        if file_ext == '.pdf':
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        elif file_ext in ['.tif', '.tiff']:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi, format='tif', compression='tiff_lzw')
        else:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi)
        
        plt.close()
      
    def plot_dca_v2(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]], save_name: str = 'DCA.pdf', title: str = 'test') -> None:
        """
        Plot Decision Curve Analysis (DCA) for a single dataset (optimized version)
        
        Args:
            models_data: Dictionary with model names as keys and (y_true, y_pred_proba) tuples as values
            save_name: Name of the file to save the plot
            title: Data type for title display ('train' or 'test')
        """
        # Create figure - optimized for SCI journal requirements (single column)
        plt.figure(figsize=(5, 5))
        
        # Define threshold range
        thresholds = np.linspace(0, 1, 100)
        model_styles = self._build_model_curve_styles(list(models_data.keys()))
        
        # Extract y_true as reference (any model can be used since y_true should be consistent)
        if not models_data:
            logger.warning("No data provided for DCA plot")
            return
        
        # 检测模型的输出概率是否超过0-1，如果超过则进行归一化
        for model_name, (y_true, y_pred_proba) in models_data.items():
            if np.any(y_pred_proba > 1) or np.any(y_pred_proba < 0):
                logger.warning(f"Model {model_name} has predicted probabilities outside [0, 1]")
                y_pred_proba = (y_pred_proba - np.min(y_pred_proba)) / (np.max(y_pred_proba) - np.min(y_pred_proba))
                models_data[model_name] = (y_true, y_pred_proba)
        y_true = next(iter(models_data.values()))[0]
        
        # Calculate and plot "Treat All" curve
        net_benefit_all = np.array([calculate_net_benefit(y_true, np.ones_like(y_true), t) for t in thresholds])
        plt.plot(thresholds, net_benefit_all, 'k--', label='Treat All', linewidth=1.5)
        
        # Calculate and plot "Treat None" curve
        net_benefit_none = np.array([calculate_net_benefit(y_true, np.zeros_like(y_true), t) for t in thresholds])
        plt.plot(thresholds, net_benefit_none, 'k-', label='Treat None', linewidth=1.5)
        
        # Plot decision curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            net_benefits = np.array([calculate_net_benefit(y_true, y_pred_proba, t) for t in thresholds])
            style = model_styles[model_name]
            plt.plot(
                thresholds,
                net_benefits,
                linewidth=1.8,
                label=model_name,
                color=style["color"],
                linestyle=style["linestyle"],
            )
        
        # Beautify the plot
        plt.xlabel('Threshold Probability', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.ylabel('Net Benefit', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.title(title, fontsize=11, fontfamily=PUBLICATION_FONT)
        plt.grid(True, linestyle='--', alpha=0.7)
        # plt.gca().set_facecolor('#f8f9fa')
        plt.legend(loc='best', fontsize=9)
        
        # Only show left and bottom spines and set their width to 1.5
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # Safely set y-axis range, handling possible NaN or Inf
        y_min = -0.05  # Default minimum
        y_max = 0.5    # Default maximum
        
        # Safely get minimum of net_benefit_none
        if len(net_benefit_none) > 0 and np.isfinite(net_benefit_none).any():
            none_min = np.nanmin(net_benefit_none[np.isfinite(net_benefit_none)])
            if np.isfinite(none_min):
                y_min = min(y_min, none_min)
        
        # Safely get maximum of net_benefit_all
        if len(net_benefit_all) > 0 and np.isfinite(net_benefit_all).any():
            all_max = np.nanmax(net_benefit_all[np.isfinite(net_benefit_all)])
            if np.isfinite(all_max):
                y_max = max(y_max, all_max + 0.1)
        
        plt.ylim([y_min, y_max])
        
        # Save image
        plt.tight_layout()
        
        # 根据文件扩展名决定是否应用压缩和DPI设置
        file_ext = os.path.splitext(save_name)[1].lower()
        if file_ext == '.pdf':
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        elif file_ext in ['.tif', '.tiff']:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi, format='tif', compression='tiff_lzw')
        else:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi)
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             save_name: str = 'Confusion_Matrix.pdf', 
                             title: str = 'Confusion Matrix',
                             class_names: List[str] = None,
                             normalize: bool = False,
                             cmap: str = 'Blues') -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            save_name (str): Name of the file to save the plot
            title (str): Title of the plot
            class_names (List[str]): Names of the classes (default: None, will use '0', '1' for binary classification)
            normalize (bool): Whether to normalize the confusion matrix (default: False)
            cmap (str): Colormap to use (default: 'Blues')
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Set class names if not provided
        if class_names is None:
            if cm.shape[0] == 2:  # Binary classification
                class_names = ['Negative', 'Positive']
            else:  # Multi-class classification
                class_names = [str(i) for i in range(cm.shape[0])]
        
        # Normalize the confusion matrix if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized {title}'
        else:
            fmt = 'd'
        
        # Create figure and plot confusion matrix - optimized for SCI journal
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar=True, square=True, linewidths=0.5)
        
        # Add labels and title
        plt.xlabel('Predicted Label', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.ylabel('True Label', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.title(title, fontsize=11, fontfamily=PUBLICATION_FONT)
        
        # Calculate and add metrics to the plot
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        plt.figtext(0.5, 0.01, 
                  f'Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}',
                  ha='center', fontsize=8, fontfamily=PUBLICATION_FONT, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Adjust plot aesthetics
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure
        file_ext = os.path.splitext(save_name)[1].lower()
        if file_ext == '.pdf':
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        elif file_ext in ['.tif', '.tiff']:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi, format='tif', compression='tiff_lzw')
        else:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi)
        
        plt.close()
    
    def plot_calibration_v2(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]], save_name: str = 'Calibration.pdf', n_bins: int = 5, title: str = 'test') -> None:
        """
        Plot calibration curves for a single dataset (optimized version)
        
        Args:
            models_data: Dictionary with model names as keys and (y_true, y_pred_proba) tuples as values
            save_name: Name of the file to save the plot
            n_bins: Number of bins to use for calibration curve
            title: Data type for title display ('train' or 'test')
        """
        # Create figure - optimized for SCI journal requirements (single column)
        plt.figure(figsize=(5, 5))
        model_styles = self._build_model_curve_styles(list(models_data.keys()))
        
        # Plot calibration curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            # Ensure predicted probabilities are within 0-1 range
            y_pred_normalized = (y_pred_proba - np.min(y_pred_proba)) / (np.max(y_pred_proba) - np.min(y_pred_proba))
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_pred_normalized, n_bins=n_bins, strategy='quantile')
            style = model_styles[model_name]
            plt.plot(
                prob_pred,
                prob_true,
                linewidth=1.8,
                markersize=5,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                label=model_name,
            )
        
        # Add ideal calibration line and beautify the plot
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.ylabel('Positive Sample Proportion', fontsize=10, fontfamily=PUBLICATION_FONT)
        
        # Set title based on data type
        plt.title(title, fontsize=11, fontfamily=PUBLICATION_FONT)
        
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        # plt.gca().set_facecolor('#f8f9fa')
        
        # Only show left and bottom spines and set their width to 1.5
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # Expand axis range
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Save image
        plt.tight_layout()
        
        # 根据文件扩展名决定是否应用压缩和DPI设置
        file_ext = os.path.splitext(save_name)[1].lower()
        if file_ext == '.pdf':
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        elif file_ext in ['.tif', '.tiff']:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi, format='tif', compression='tiff_lzw')
        else:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi)
        
        plt.close()
             
    def _build_shap_explainer(self, model: Any, X: np.ndarray) -> Any:
        """
        Choose the SHAP explainer matching the model family.

        Args:
            model (Any): Trained model, possibly a HABIT wrapper exposing the
                underlying sklearn estimator as ``.model``.
            X (np.ndarray): Background/feature data used by the explainer.

        Returns:
            Any: An unfitted-on-data SHAP explainer for this model.
        """
        import shap

        # Get model type - if not available, try to infer from model object
        model_type = getattr(model, 'model_type', None)

        if model_type == 'linear':
            # For custom linear models, try to access the underlying sklearn model
            if hasattr(model, 'model'):
                # Access the internal sklearn model
                return shap.LinearExplainer(model.model, X)
            if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                # If model has coefficients and intercept directly
                return shap.LinearExplainer((model.coef_, model.intercept_), X)
            # Fallback to KernelExplainer if we can't access the model structure
            return shap.KernelExplainer(model.predict_proba, X)
        if model_type == 'tree':
            # For tree-based models
            if hasattr(model, 'model'):
                # Access the internal sklearn model
                return shap.TreeExplainer(model.model)
            # Try to use the model directly
            return shap.TreeExplainer(model)
        # Default to KernelExplainer for other model types
        return shap.KernelExplainer(model.predict_proba, X)

    def compute_shap_explanation(
        self, model: Any, X: np.ndarray
    ) -> Tuple[Any, Any]:
        """
        Compute SHAP values once so several figures can share them.

        SHAP attribution is by far the most expensive part of the explanation
        step (``KernelExplainer`` in particular). Summary, dependence and
        waterfall figures therefore all consume the values returned here rather
        than re-running the explainer.

        Args:
            model (Any): Trained model to explain.
            X (np.ndarray): Feature matrix the values are computed on.

        Returns:
            Tuple[Any, Any]: ``(shap_values, expected_value)`` where
            ``expected_value`` is the explainer's base value, needed to anchor
            per-sample waterfall plots.
        """
        explainer = self._build_shap_explainer(model, X)
        shap_values = process_shap_explanation(explainer.shap_values(X))
        return shap_values, getattr(explainer, 'expected_value', None)

    @staticmethod
    def _shap_matrix_for_positive_class(shap_values: Any) -> np.ndarray:
        """
        Reduce SHAP output to a 2D ``(n_samples, n_features)`` matrix.

        Explainers report per-class attributions in several shapes depending on
        the SHAP version and model family: a list of per-class arrays, a 3D
        array, or an already-2D array for single-output models. Dependence and
        waterfall plots need one class, and the positive (last) class is the one
        clinical reporting refers to.

        Args:
            shap_values (Any): Raw output of ``shap_values``.

        Returns:
            np.ndarray: Two-dimensional attribution matrix.
        """
        values = getattr(shap_values, 'values', shap_values)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        return values

    @staticmethod
    def _shap_base_value_for_positive_class(expected_value: Any) -> float:
        """
        Extract the scalar base value matching the positive class.

        Args:
            expected_value (Any): Explainer ``expected_value``; scalar for
                single-output models, sequence for per-class models.

        Returns:
            float: Base value used as the waterfall plot's starting point.
        """
        if expected_value is None:
            return 0.0
        values = np.atleast_1d(np.asarray(expected_value, dtype=float))
        return float(values[-1])

    def plot_shap(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        save_name: str = 'SHAP.pdf',
        shap_values: Any = None,
    ) -> None:
        """
        Plot SHAP values with bar and beeswarm plots
        
        Args:
            model (Any): Trained model
            X (np.ndarray): Feature data
            feature_names (List[str]): List of feature names
            save_name (str): Name of the file to save the plot
            shap_values (Any): Precomputed SHAP values from
                :meth:`compute_shap_explanation`. When omitted they are computed
                here, preserving the original single-call behaviour.
        """
        import shap

        if shap_values is None:
            shap_values, _ = self.compute_shap_explanation(model, X)
        
        # Plot 1: Feature importance bar plot - optimized for SCI journal
        plt.figure(figsize=(6, 5))
        plt.title('Feature Importance', fontsize=11, fontfamily=PUBLICATION_FONT)
        shap.summary_plot(
            shap_values, 
            X,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        bar_filename = os.path.splitext(save_name)[0] + '_bar' + os.path.splitext(save_name)[1]
        self._save_figure(bar_filename)
        plt.close()
        
        # Plot 2: Beeswarm plot - optimized for SCI journal
        plt.figure(figsize=(6, 5))
        plt.title('Feature Impact Distribution', fontsize=11, fontfamily=PUBLICATION_FONT)
        shap.summary_plot(
            shap_values, 
            X,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        self._save_figure(save_name)
        plt.close()

    def plot_shap_dependence(
        self,
        X: np.ndarray,
        feature_names: List[str],
        shap_values: Any,
        top_k: int = 3,
        save_name: str = 'shap_dependence.pdf',
    ) -> None:
        """
        Plot SHAP dependence for the most influential features.

        A dependence plot shows how one feature's attribution varies across its
        own value range, which is what distinguishes a monotonic predictor from
        one with a threshold or non-monotonic effect. The summary plots cannot
        show this because they collapse each feature to a distribution.

        Args:
            X (np.ndarray): Feature matrix the SHAP values were computed on.
            feature_names (List[str]): Column names aligned with ``X``.
            shap_values (Any): Values from :meth:`compute_shap_explanation`.
            top_k (int): Number of highest mean-absolute-attribution features to
                plot, each written to its own file.
            save_name (str): Base filename; the feature rank and name are
                appended to keep one figure per feature.
        """
        import shap

        values = self._shap_matrix_for_positive_class(shap_values)
        feature_data = np.asarray(getattr(X, 'values', X))
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"SHAP values have {values.shape[1]} features but "
                f"{len(feature_names)} feature names were supplied."
            )

        mean_absolute = np.abs(values).mean(axis=0)
        ranked_indices = np.argsort(mean_absolute)[::-1][: max(int(top_k), 0)]

        stem, extension = os.path.splitext(save_name)
        for rank, feature_index in enumerate(ranked_indices, start=1):
            feature_name = feature_names[feature_index]
            plt.figure(figsize=(5, 4))
            shap.dependence_plot(
                int(feature_index),
                values,
                feature_data,
                feature_names=feature_names,
                interaction_index='auto',
                show=False,
            )
            plt.title(
                f'SHAP Dependence: {feature_name}',
                fontsize=11,
                fontfamily=PUBLICATION_FONT,
            )
            plt.tight_layout()
            safe_name = self._sanitize_filename_part(feature_name)
            self._save_figure(f'{stem}_{rank}_{safe_name}{extension}')
            plt.close()

    def plot_shap_waterfall(
        self,
        X: np.ndarray,
        feature_names: List[str],
        shap_values: Any,
        expected_value: Any = None,
        n_samples: int = 2,
        save_name: str = 'shap_waterfall.pdf',
    ) -> None:
        """
        Plot per-sample SHAP waterfall explanations.

        Summary plots describe the cohort; a waterfall plot explains one
        prediction, which is the form clinicians ask for when they want to know
        why a specific patient was scored as they were. Samples are picked at
        the extremes and the median of the predicted-risk ordering, so the
        exported set covers a low-, mid- and high-risk example rather than an
        arbitrary first few rows.

        Args:
            X (np.ndarray): Feature matrix the SHAP values were computed on.
            feature_names (List[str]): Column names aligned with ``X``.
            shap_values (Any): Values from :meth:`compute_shap_explanation`.
            expected_value (Any): Explainer base value from the same call.
            n_samples (int): Number of samples to export.
            save_name (str): Base filename; the sample index is appended.
        """
        import shap

        values = self._shap_matrix_for_positive_class(shap_values)
        feature_data = np.asarray(getattr(X, 'values', X))
        base_value = self._shap_base_value_for_positive_class(expected_value)

        sample_indices = self._select_representative_samples(
            values.sum(axis=1), n_samples=int(n_samples)
        )

        stem, extension = os.path.splitext(save_name)
        for sample_index in sample_indices:
            explanation = shap.Explanation(
                values=values[sample_index],
                base_values=base_value,
                data=feature_data[sample_index],
                feature_names=list(feature_names),
            )
            plt.figure(figsize=(6, 5))
            shap.plots.waterfall(explanation, show=False)
            plt.title(
                f'SHAP Explanation: Sample {sample_index}',
                fontsize=11,
                fontfamily=PUBLICATION_FONT,
            )
            plt.tight_layout()
            self._save_figure(f'{stem}_sample{sample_index}{extension}')
            plt.close()

    @staticmethod
    def _select_representative_samples(
        scores: np.ndarray,
        n_samples: int = 2,
    ) -> List[int]:
        """
        Pick sample indices spanning the predicted-risk range.

        Args:
            scores (np.ndarray): Per-sample score used for ordering, e.g. the
                summed SHAP attribution.
            n_samples (int): Number of indices to return.

        Returns:
            List[int]: Indices evenly spaced over the score ranking, from the
            lowest to the highest score.
        """
        total = len(scores)
        n_samples = max(min(int(n_samples), total), 0)
        if n_samples == 0:
            return []
        order = np.argsort(scores)
        if n_samples == 1:
            return [int(order[total // 2])]
        positions = np.linspace(0, total - 1, n_samples).round().astype(int)
        return [int(order[position]) for position in positions]

    def plot_permutation_importance(
        self,
        importance: pd.DataFrame,
        save_name: str = 'permutation_importance.pdf',
        title: str = 'Permutation Importance',
        top_k: int = 20,
    ) -> None:
        """
        Plot permutation importance as a horizontal bar chart with error bars.

        Args:
            importance (pd.DataFrame): Table from
                ``compute_permutation_importance`` with ``feature``,
                ``importance_mean`` and ``importance_std`` columns, already
                sorted by descending importance.
            save_name (str): Name of the file to save the plot.
            title (str): Title of the plot.
            top_k (int): Maximum number of features to display.
        """
        top = importance.head(max(int(top_k), 1)).iloc[::-1]

        # Height grows with the number of bars so labels never overlap.
        figure_height = max(3.0, 0.28 * len(top) + 1.2)
        plt.figure(figsize=(6, figure_height))
        plt.barh(
            top['feature'],
            top['importance_mean'],
            xerr=top['importance_std'],
            color='#4C72B0',
            edgecolor='black',
            linewidth=0.5,
            error_kw={'ecolor': '#444444', 'elinewidth': 0.8, 'capsize': 2},
        )
        # A permuted feature that the model does not use scores around zero;
        # the reference line makes that boundary explicit.
        plt.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
        plt.xlabel(
            'Mean Score Decrease', fontsize=10, fontfamily=PUBLICATION_FONT
        )
        plt.ylabel('Feature', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.title(title, fontsize=11, fontfamily=PUBLICATION_FONT)
        plt.yticks(fontsize=8)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')

        plt.tight_layout()
        self._save_figure(save_name)
        plt.close()

    @staticmethod
    def _sanitize_filename_part(name: str) -> str:
        """
        Make a feature name safe to embed in a filename.

        Radiomics feature names routinely contain spaces, slashes and brackets,
        any of which either break a path or produce an unreadable filename.

        Args:
            name (str): Raw feature name.

        Returns:
            str: Filename-safe fragment.
        """
        safe = ''.join(
            character if character.isalnum() or character in '-_' else '_'
            for character in str(name)
        )
        return safe.strip('_')[:60] or 'feature'

    def _save_figure(self, save_name: str) -> None:
        """
        Helper method to save figures with appropriate format and DPI
        
        Args:
            save_name (str): Name of the file to save
        """
        file_ext = os.path.splitext(save_name)[1].lower()
        if file_ext == '.pdf':
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        elif file_ext in ['.tif', '.tiff']:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi, format='tif', compression='tiff_lzw')
        else:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi)
    
    def plot_pr_curve(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                  save_name: str = 'PR_curve.pdf', 
                  title: str = 'evaluation') -> None:

        """
        Plot Precision-Recall curve for multiple models
        Args:
            models_data: Dictionary with model names as keys and (y_true, y_pred_proba) tuples as values
            save_name: Name of the file to save the plot
            title: Data type for title display ('train', 'test', or 'evaluation')

        """
        # Create figure - optimized for SCI journal requirements (single column)
        plt.figure(figsize=(5, 5))
        model_styles = self._build_model_curve_styles(list(models_data.keys()))
        # Plot PR curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, drop_intermediate=True)

            # Calculate average precision score
            AUPRC = auc(recall, precision)
            style = model_styles[model_name]
            plt.plot(
                recall,
                precision,
                linewidth=1.8,
                color=style["color"],
                linestyle=style["linestyle"],
                label=f'{model_name} (AUPRC = {AUPRC:.2f})',
            )
        
        # Beautify the plot
        plt.xlabel('Recall', fontsize=10, fontfamily=PUBLICATION_FONT)  # 修改X轴标签

        plt.ylabel('Precision', fontsize=10, fontfamily=PUBLICATION_FONT)
        plt.title(f'{title}', fontsize=11, fontfamily=PUBLICATION_FONT)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        # Set axis limits for left-to-right, bottom-to-top direction
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])

        # Only show left and bottom spines and set their width to 1.5
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        plt.tight_layout()
        
        # 根据文件扩展名决定是否应用压缩和DPI设置
        file_ext = os.path.splitext(save_name)[1].lower()
        if file_ext == '.pdf':
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight')
        elif file_ext in ['.tif', '.tiff']:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi, format='tif', compression='tiff_lzw')
        else:
            plt.savefig(os.path.join(self.output_dir, save_name), bbox_inches='tight', 
                        dpi=self.dpi)
        
        plt.close()