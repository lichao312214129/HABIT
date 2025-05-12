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
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve  # Calibration curve related
from ..evaluation.metrics import calculate_net_benefit
from ....utils.visualization_utils import process_shap_explanation

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
        
        # Set plotting style
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
        # Create figure
        plt.figure(figsize=(8, 8))
        
        # Plot ROC curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=1.5)
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        
        # Set title based on data type
        title = f"{title} Set ROC Curves"
        plt.title(title, fontsize=14)
        plt.legend(loc="lower right")
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().set_facecolor('#f8f9fa')
        
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
        plt.figure(figsize=(8, 8))
        
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
        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        title = f"{'Training' if title.lower() == 'train' else 'Testing'} Set Decision Curve Analysis"
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().set_facecolor('#f8f9fa')
        plt.legend(loc='best')
        
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
        
        # Create figure and plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar=True, square=True, linewidths=0.5)
        
        # Add labels and title
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14)
        
        # Calculate and add metrics to the plot
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        plt.figtext(0.5, 0.01, 
                  f'Accuracy: {accuracy:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}',
                  ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
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
    
    def calculate_net_benefit(self, y_true, y_pred_proba, threshold):
        """
        Calculate the net benefit at a given threshold for decision curve analysis.
        
        Args:
            y_true (np.ndarray): True binary labels (0 or 1)
            y_pred_proba (np.ndarray): Predicted probabilities
            threshold (float): Decision threshold for classification
            
        Returns:
            float: Net benefit value at the given threshold
        """
        # Handle boundary cases
        if threshold >= 0.999:  # Prevent division by values close to zero
            return 0.0
            
        # Convert to binary prediction based on threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate true positives and false positives
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        
        # Total sample count
        n = len(y_true)
        
        # Calculate net benefit
        if TP + FP == 0:
            return 0
        else:
            benefit = (TP / n) - (FP / n) * (threshold / (1 - threshold))
            
            # Ensure returned value is finite
            if not np.isfinite(benefit):
                return 0.0
            return benefit
    
    def plot_calibration_v2(self, models_data: Dict[str, Tuple[np.ndarray, np.ndarray]], save_name: str = 'Calibration.pdf', n_bins: int = 5, title: str = 'test') -> None:
        """
        Plot calibration curves for a single dataset (optimized version)
        
        Args:
            models_data: Dictionary with model names as keys and (y_true, y_pred_proba) tuples as values
            save_name: Name of the file to save the plot
            n_bins: Number of bins to use for calibration curve
            title: Data type for title display ('train' or 'test')
        """
        plt.figure(figsize=(8, 8))
        
        # Plot calibration curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            # Ensure predicted probabilities are within 0-1 range
            y_pred_normalized = (y_pred_proba - np.min(y_pred_proba)) / (np.max(y_pred_proba) - np.min(y_pred_proba))
            
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_pred_normalized, n_bins=n_bins, strategy='quantile')
            plt.plot(prob_pred, prob_true, 's-', linewidth=1.5, markersize=8, label=model_name)
        
        # Add ideal calibration line and beautify the plot
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Positive Sample Proportion', fontsize=12)
        
        # Set title based on data type
        title = f"{'Training' if title.lower() == 'train' else 'Testing'} Set Calibration Curves"
        plt.title(title, fontsize=14)
        
        plt.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().set_facecolor('#f8f9fa')
        
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
             
    def plot_shap(self, model: Any, X: np.ndarray, feature_names: List[str], save_name: str = 'SHAP.pdf', 
                 n_samples_to_plot: int = 5) -> None:
        """
        Plot SHAP values with multiple visualization types
        
        Args:
            model (Any): Trained model
            X (np.ndarray): Feature data
            feature_names (List[str]): List of feature names
            save_name (str): Name of the file to save the plot
            n_samples_to_plot (int): Number of individual samples to plot for force and waterfall plots
        """
        # Create SHAP explainer
        try:
            shap.initjs()
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend for saving figures
            
            # First check if model has get_model method (custom model wrapper)
            if hasattr(model, 'get_model'):
                # Get the underlying model
                base_model = model.get_model()
                if hasattr(base_model, 'feature_importances_'):
                    # Tree model
                    explainer = shap.TreeExplainer(base_model)
                elif hasattr(base_model, 'coef_'):
                    # Linear model
                    explainer = shap.LinearExplainer(base_model, X)
                else:
                    # Other models - use KernelExplainer
                    explainer = shap.KernelExplainer(model.predict_proba, X)
            elif hasattr(model, 'feature_importances_'):
                # Tree model
                explainer = shap.TreeExplainer(model)
            elif hasattr(model, 'coef_'):
                # Linear model
                explainer = shap.LinearExplainer(model, X)
            else:
                # Other models
                explainer = shap.KernelExplainer(model.predict, X)
            
            # Create Explanation object for the newer SHAP API if possible
            try:
                # For newer SHAP versions
                shap_values = explainer(X)
                use_new_api = True
            except Exception:
                # For older SHAP versions
                shap_values = explainer.shap_values(X)
                use_new_api = False
                
                # For classification problems, shap_values may be a list containing SHAP values for each class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Take SHAP values for positive class
            
            # Ensure feature names list matches the number of features
            if len(feature_names) != X.shape[1]:
                print(f"Warning: Feature names list length ({len(feature_names)}) does not match number of features ({X.shape[1]})")
                feature_names = [f"Feature {i}" for i in range(X.shape[1])]
            
            # 1. Beeswarm Plot (Summary Plot)
            plt.figure(figsize=(10, 6))
            if use_new_api:
                shap.plots.beeswarm(shap_values, show=False)
            else:
                shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.title('Feature Importance (SHAP)')
            
            # Save beeswarm plot
            beeswarm_save_name = save_name.replace(os.path.splitext(save_name)[1], f'_beeswarm{os.path.splitext(save_name)[1]}')
            file_ext = os.path.splitext(beeswarm_save_name)[1].lower()
            if file_ext == '.pdf':
                plt.savefig(os.path.join(self.output_dir, beeswarm_save_name), bbox_inches='tight')
            elif file_ext in ['.tif', '.tiff']:
                plt.savefig(os.path.join(self.output_dir, beeswarm_save_name), bbox_inches='tight', 
                            dpi=self.dpi, format='tif', compression='tiff_lzw')
            else:
                plt.savefig(os.path.join(self.output_dir, beeswarm_save_name), bbox_inches='tight', 
                            dpi=self.dpi)
            plt.close()
            
            # 2. Bar Plot with improved saturation
            plt.figure(figsize=(10, 6))
            if use_new_api:
                try:
                    # First try the direct bar plot function in newer versions
                    shap.plots.bar(shap_values, show=False)
                except Exception:
                    # Fallback to using summary plot with bar plot_type
                    if hasattr(shap_values, 'values'):
                        # Extract values from Explanation object if available
                        vals = shap_values.values
                        if vals.ndim > 2:  # Multi-class case
                            vals = vals[:, :, 1]  # Take positive class values
                        # Calculate mean absolute value for each feature
                        feature_importance = np.abs(vals).mean(0)
                        # Sort features by importance
                        sorted_idx = np.argsort(feature_importance)
                        # Create a bar plot manually with better colors
                        plt.figure(figsize=(10, 6))
                        barlist = plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                        # Set a more saturated color
                        for bar in barlist:
                            bar.set_color('#1f77b4')  # A more saturated blue
                        plt.yticks(range(len(sorted_idx)), 
                                 [feature_names[i] if hasattr(shap_values, 'feature_names') else f"Feature {i}" 
                                  for i in sorted_idx])
                    else:
                        # Use summary plot with bar plot_type for older versions
                        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False, 
                                         color='#1f77b4')  # Use more saturated color
            else:
                # Use a more saturated color palette for older API
                current_cmap = plt.cm.get_cmap('Blues')  # Get the 'Blues' colormap
                saturated_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
                    'saturated_blues', [current_cmap(0.3), current_cmap(1.0)], N=256)
                shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", 
                                 show=False, cmap=saturated_cmap)
            plt.title('Feature Importance (SHAP Bar Plot)')
            
            # Save bar plot
            # Dynamically replace the file extension with '_bar' while preserving the original extension
            bar_save_name = save_name.replace(os.path.splitext(save_name)[1], f'_bar{os.path.splitext(save_name)[1]}')
            file_ext = os.path.splitext(bar_save_name)[1].lower()
            if file_ext == '.pdf':
                plt.savefig(os.path.join(self.output_dir, bar_save_name), bbox_inches='tight')
            elif file_ext in ['.tif', '.tiff']:
                plt.savefig(os.path.join(self.output_dir, bar_save_name), bbox_inches='tight', 
                            dpi=self.dpi, format='tif', compression='tiff_lzw')
            else:
                plt.savefig(os.path.join(self.output_dir, bar_save_name), bbox_inches='tight', 
                            dpi=self.dpi)
            plt.close()
            
            # ================================================
            # 3. Force Plots
            # Generate force plots for non-notebook environments
            
            # Process SHAP values to limit decimal places to 3
            formatted_shap_values = process_shap_explanation(shap_values, decimal_places=3)
            # Process X values to ensure feature display values also have only 3 decimal places
            formatted_X = np.round(X, 3)
            
            # Generate individual force plots for each sample
            for i in range(min(n_samples_to_plot, X.shape[0])):
                try:
                    # Create force plot for a single sample
                    force_plot_name = save_name.replace('.pdf', f'_force_sample_{i+1}.pdf')
                    force_plot_path = os.path.join(self.output_dir, force_plot_name)
                    
                    if use_new_api:
                        # Individual force plot using new API
                        shap_values_instance = formatted_shap_values[i]
                        
                        # Create force plot with explicit figure size and DPI
                        plt.figure(figsize=(20, 3))
                        shap.plots.force(shap_values_instance, matplotlib=True, show=False)
                        plt.title(f'SHAP Force Plot for Sample {i+1}')
                        
                        # Save force plot with high DPI
                        plt.savefig(force_plot_path, bbox_inches='tight', dpi=300)
                        plt.close()
                        
                except Exception as e:
                    print(f"Warning: Failed to plot force plot for sample {i+1}: {str(e)}")
            
            # ================================================
            # 4. Waterfall Plot for first n_samples_to_plot samples
            if use_new_api:
                for i in range(min(n_samples_to_plot, X.shape[0])):
                    try:
                        plt.figure(figsize=(10, 6))
                        shap.plots.waterfall(shap_values[i], show=False)
                        plt.title(f'SHAP Waterfall Plot for Sample {i+1}')
                        
                        # Save waterfall plot
                        waterfall_save_name = save_name.replace('.pdf', f'_waterfall_sample_{i+1}.pdf')
                        file_ext = os.path.splitext(waterfall_save_name)[1].lower()
                        if file_ext == '.pdf':
                            plt.savefig(os.path.join(self.output_dir, waterfall_save_name), bbox_inches='tight')
                        elif file_ext in ['.tif', '.tiff']:
                            plt.savefig(os.path.join(self.output_dir, waterfall_save_name), bbox_inches='tight', 
                                       dpi=self.dpi, format='tif', compression='tiff_lzw')
                        else:
                            plt.savefig(os.path.join(self.output_dir, waterfall_save_name), bbox_inches='tight', 
                                       dpi=self.dpi)
                        plt.close()
                    except Exception as e:
                        print(f"Warning: Failed to plot waterfall for sample {i+1}: {str(e)}")
            
            # 5. Feature importance heatmap for new API or dependence plots for old API
            try:
                if use_new_api:
                    # Heatmap
                    plt.figure(figsize=(12, 8))
                    shap.plots.heatmap(shap_values, show=False)
                    plt.title('SHAP Feature Importance Heatmap')
                    
                    # Save heatmap
                    heatmap_save_name = save_name.replace('.pdf', '_heatmap.pdf')
                    file_ext = os.path.splitext(heatmap_save_name)[1].lower()
                    if file_ext == '.pdf':
                        plt.savefig(os.path.join(self.output_dir, heatmap_save_name), bbox_inches='tight')
                    elif file_ext in ['.tif', '.tiff']:
                        plt.savefig(os.path.join(self.output_dir, heatmap_save_name), bbox_inches='tight', 
                                   dpi=self.dpi, format='tif', compression='tiff_lzw')
                    else:
                        plt.savefig(os.path.join(self.output_dir, heatmap_save_name), bbox_inches='tight', 
                                   dpi=self.dpi)
                    plt.close()
                else:
                    # Get feature importance scores
                    feature_importance = np.abs(shap_values).mean(0)
                    top_features_idx = np.argsort(feature_importance)[-3:][::-1]
                    
                    for idx in top_features_idx:
                        plt.figure(figsize=(10, 6))
                        shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, show=False)
                        plt.title(f'SHAP Dependence Plot for {feature_names[idx]}')
                        
                        # Save dependence plot
                        dep_save_name = save_name.replace('.pdf', f'_dependence_{feature_names[idx]}.pdf')
                        file_ext = os.path.splitext(dep_save_name)[1].lower()
                        if file_ext == '.pdf':
                            plt.savefig(os.path.join(self.output_dir, dep_save_name), bbox_inches='tight')
                        elif file_ext in ['.tif', '.tiff']:
                            plt.savefig(os.path.join(self.output_dir, dep_save_name), bbox_inches='tight', 
                                       dpi=self.dpi, format='tif', compression='tiff_lzw')
                        else:
                            plt.savefig(os.path.join(self.output_dir, dep_save_name), bbox_inches='tight', 
                                       dpi=self.dpi)
                        plt.close()
            except Exception as e:
                print(f"Warning: Failed to plot additional visualizations: {str(e)}")
            
        except Exception as e:
            print(f"Warning: Failed to plot SHAP values: {str(e)}")
    
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
        plt.figure(figsize=(6, 6))
        # Plot PR curves for each model
        for model_name, (y_true, y_pred_proba) in models_data.items():
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            # Calculate average precision score
            ap = np.trapz(precision, recall)
            # 关键修改：绘制时交换x和y轴，使图形从左下到右上
            plt.plot(1 - precision, recall, linewidth=2, label=f'{model_name} (AP = {ap:.3f})')
        
        # Beautify the plot
        plt.xlabel('1 - Precision')  # 修改X轴标签

        plt.ylabel('Recall')
        plt.title(f'Modified Precision-Recall Curve ({title.capitalize()})')
        plt.legend(loc='best')
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