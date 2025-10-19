"""
Plotting Module
Provides various evaluation chart plotting functions
"""

import os
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from sklearn.calibration import calibration_curve  # Calibration curve related
from ..evaluation.metrics import calculate_net_benefit
from ....utils.visualization_utils import process_shap_explanation
from ....utils.font_config import setup_publication_font

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
        
        # Set plotting style with Arial font for publication quality
        setup_publication_font()
        # plt.style.use('seaborn')
        # sns.set_context("paper", font_scale=1.2)
        
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
        
        # Plot ROC curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})', linewidth=1.5)
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate', fontsize=10, fontfamily='Arial')
        plt.ylabel('True Positive Rate', fontsize=10, fontfamily='Arial')
        
        # Set title based on data type
        plt.title(title, fontsize=11, fontfamily='Arial')
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
        
        # Extract y_true as reference (any model can be used since y_true should be consistent)
        if not models_data:
            print("No data provided for DCA plot")
            return
        
        # 检测模型的输出概率是否超过0-1，如果超过则进行归一化
        for model_name, (y_true, y_pred_proba) in models_data.items():
            if np.any(y_pred_proba > 1) or np.any(y_pred_proba < 0):
                print(f"Warning: Model {model_name} has predicted probabilities outside [0, 1]")
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
            plt.plot(thresholds, net_benefits, '-', linewidth=1.5, label=model_name)
        
        # Beautify the plot
        plt.xlabel('Threshold Probability', fontsize=10, fontfamily='Arial')
        plt.ylabel('Net Benefit', fontsize=10, fontfamily='Arial')
        plt.title(title, fontsize=11, fontfamily='Arial')
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
        plt.xlabel('Predicted Label', fontsize=10, fontfamily='Arial')
        plt.ylabel('True Label', fontsize=10, fontfamily='Arial')
        plt.title(title, fontsize=11, fontfamily='Arial')
        
        # Calculate and add metrics to the plot
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        plt.figtext(0.5, 0.01, 
                  f'Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}',
                  ha='center', fontsize=8, fontfamily='Arial', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
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
        
        # Plot calibration curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            # Ensure predicted probabilities are within 0-1 range
            y_pred_normalized = (y_pred_proba - np.min(y_pred_proba)) / (np.max(y_pred_proba) - np.min(y_pred_proba))
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_pred_normalized, n_bins=n_bins, strategy='quantile')
            # Use different marker styles for each model for better distinction
            # Common matplotlib markers: 'o' (circle), '^' (triangle_up), 'D' (diamond), 'v' (triangle_down), '*' (star), 'x' (x), '+' (plus), 'p' (pentagon), etc.
            # Here, as an example, use 'o' (circle) marker instead of 's' (square)
            plt.plot(prob_pred, prob_true, '.-', linewidth=1.5, markersize=8, label=model_name)
        
        # Add ideal calibration line and beautify the plot
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability', fontsize=10, fontfamily='Arial')
        plt.ylabel('Positive Sample Proportion', fontsize=10, fontfamily='Arial')
        
        # Set title based on data type
        plt.title(title, fontsize=11, fontfamily='Arial')
        
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
             
    def plot_shap(self, model: Any, X: np.ndarray, feature_names: List[str], save_name: str = 'SHAP.pdf') -> None:
        """
        Plot SHAP values with bar and beeswarm plots
        
        Args:
            model (Any): Trained model
            X (np.ndarray): Feature data
            feature_names (List[str]): List of feature names
            save_name (str): Name of the file to save the plot
        """
        # Get model type - if not available, try to infer from model object
        model_type = getattr(model, 'model_type', None)
        
        # Calculate SHAP values based on model type
        if model_type == 'linear':
            # For custom linear models, try to access the underlying sklearn model
            if hasattr(model, 'model'):
                # Access the internal sklearn model
                sklearn_model = model.model
                explainer = shap.LinearExplainer(sklearn_model, X)
            elif hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                # If model has coefficients and intercept directly
                explainer = shap.LinearExplainer((model.coef_, model.intercept_), X)
            else:
                # Fallback to KernelExplainer if we can't access the model structure
                explainer = shap.KernelExplainer(model.predict_proba, X)
        elif model_type == 'tree':
            # For tree-based models
            if hasattr(model, 'model'):
                # Access the internal sklearn model
                sklearn_model = model.model
                explainer = shap.TreeExplainer(sklearn_model)
            else:
                # Try to use the model directly
                explainer = shap.TreeExplainer(model)
        else:
            # Default to KernelExplainer for other model types
            explainer = shap.KernelExplainer(model.predict_proba, X)
        
        # Get SHAP values
        shap_values = explainer.shap_values(X)
        
        # Process SHAP explanation for consistency
        shap_values = process_shap_explanation(shap_values)
        
        # Plot 1: Feature importance bar plot - optimized for SCI journal
        plt.figure(figsize=(6, 5))
        plt.title('Feature Importance', fontsize=11, fontfamily='Arial')
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
        plt.title('Feature Impact Distribution', fontsize=11, fontfamily='Arial')
        shap.summary_plot(
            shap_values, 
            X,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        self._save_figure(save_name)
        plt.close()
    
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
        # Plot PR curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, drop_intermediate=True)

            # Calculate average precision score
            AUPRC = auc(recall, precision)
            plt.plot(recall, precision, linewidth=1.5, label=f'{model_name} (AUPRC = {AUPRC:.2f})')
        
        # Beautify the plot
        plt.xlabel('Recall', fontsize=10, fontfamily='Arial')  # 修改X轴标签

        plt.ylabel('Precision', fontsize=10, fontfamily='Arial')
        plt.title(f'{title}', fontsize=11, fontfamily='Arial')
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