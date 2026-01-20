"""
Model Evaluation Module
Provides functions for model training, evaluation, and result analysis
"""

import os
import json
from typing import Dict, List, Tuple, Union, Any, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

from .metrics import calculate_metrics, calculate_metrics_youden, delong_roc_ci
from ..visualization.plotting import Plotter
from ..statistics.delong_test import delong_roc_test, delong_roc_ci
from habit.utils.log_utils import get_module_logger

class ModelEvaluator:
    def __init__(self, output_dir: str):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory where evaluation results and plots will be saved
        """
        self.output_dir = output_dir
        self.plotter = Plotter(self.output_dir)
        self.logger = get_module_logger('evaluation.model')
    
    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate a single model on a single dataset.
        
        Args:
            model (Any): Trained model with predict and predict_proba methods
            X (pd.DataFrame): Feature data
            y (pd.Series): Label data
            dataset_name (str): Name of the dataset
            
        Returns:
            Dict[str, Any]: Dictionary containing evaluation results
        """
        # Prediction
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        
        # Handle potential 2D probability results
        if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]
            
        # Convert data format
        y_values = y.values if hasattr(y, 'values') else y
        
        # Calculate metrics
        metrics = calculate_metrics(y_values, y_pred, y_pred_proba)
        
        # Return results (same format as before)
        return {
            'metrics': metrics,
            'y_true': y_values.tolist() if hasattr(y_values, 'tolist') else list(y_values),
            'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
            'y_pred_proba': y_pred_proba.tolist() if hasattr(y_pred_proba, 'tolist') else list(y_pred_proba)
        }

    def plot_curves(
        self, 
        model_data: Dict[str, Dict[str, Dict[str, List]]],
        curve_type: Literal['roc', 'dca', 'calibration', 'pr', 'all'] = 'all',
        title: str = 'evaluation',
        output_dir: Optional[str] = None,
        prefix: str = '',
        n_bins: int = 10
    ) -> Dict[str, str]:
        """
        Plot various evaluation curves using methods from plotting.py
        
        Args:
            model_data (Dict): Dictionary containing model evaluation results
                Format: {dataset_name: {model_name: {'y_true': [...], 'y_pred_proba': [...]}}}
            curve_type (Literal): Type of curve to plot, can be 'roc', 'dca', 'calibration', 'pr', or 'all'
            title (str): Keyword for chart title
            output_dir (Optional[str]): Output directory, defaults to self.output_dir
            prefix (str): Prefix for output filenames
            n_bins (int): Number of bins for calibration curve
            
        Returns:
            Dict[str, str]: Dictionary containing paths to generated chart files
        """
        # Ensure output directory exists
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create or use Plotter instance
        plotter = self.plotter
        if output_dir != self.output_dir:
            plotter = Plotter(output_dir)
        
        # Used to store paths to generated chart files
        result_files = {}
        
        # Generate charts for each dataset (e.g., train or test)
        for dataset, models in model_data.items():
            # Convert current dataset data to plotting format
            plotting_data = {}
            for model_name, data in models.items():
                y_true = np.array(data.get('y_true', []))
                y_pred_proba = np.array(data.get('y_pred_proba', []))
                plotting_data[model_name] = (y_true, y_pred_proba)
            
            # Generate file name prefix for current dataset
            dataset_prefix = f"{prefix}{dataset}_" if prefix else f"{dataset}_"
            dataset_title = f"{title}_{dataset}"
            
            # Plot ROC curve
            if curve_type in ['roc', 'all']:
                roc_filename = f'{dataset_prefix}roc_curve.pdf'
                plotter.plot_roc_v2(plotting_data, save_name=roc_filename, title=dataset_title)
                result_files[f'roc_{dataset}'] = os.path.join(output_dir, roc_filename)
            
            # Plot decision curve
            if curve_type in ['dca', 'all']:
                dca_filename = f'{dataset_prefix}decision_curve.pdf'
                plotter.plot_dca_v2(plotting_data, save_name=dca_filename, title=dataset_title)
                result_files[f'dca_{dataset}'] = os.path.join(output_dir, dca_filename)
            
            # Plot calibration curve
            if curve_type in ['calibration', 'all']:
                calibration_filename = f'{dataset_prefix}calibration_curve.pdf'
                plotter.plot_calibration_v2(plotting_data, save_name=calibration_filename, title=dataset_title, n_bins=n_bins)
                result_files[f'calibration_{dataset}'] = os.path.join(output_dir, calibration_filename)
            
            # Plot precision-recall curve
            if curve_type in ['pr', 'all']:
                pr_filename = f'{dataset_prefix}precision_recall_curve.pdf'
                plotter.plot_pr_curve(plotting_data, save_name=pr_filename, title=dataset_title)
                result_files[f'pr_{dataset}'] = os.path.join(output_dir, pr_filename)
        
        return result_files

    def compare_models(self, test_data: Dict[str, Tuple[List, List]]) -> None:
        """
        Compare the performance of multiple models (using DeLong test)
        
        Args:
            test_data (Dict[str, Tuple[List, List]]): Test data dictionary, keys are model names, 
                                                     values are (y_true, y_pred_proba) tuples
        """
            
        self.logger.info("="*80)
        self.logger.info("Model AUC Comparison (DeLong test)")
        self.logger.info("="*80)
        
        # Get list of model names
        model_names = list(test_data.keys())
        
        # Used to store comparison results
        comparison_results = []
        
        # Compare each pair of models
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                y_true = test_data[model1][0]
                y_pred1 = test_data[model1][1]
                y_pred2 = test_data[model2][1]
                
                # Ensure all data are numpy array format
                y_true = np.array(y_true)
                y_pred1 = np.array(y_pred1)
                y_pred2 = np.array(y_pred2)
                
                # Perform DeLong test
                p_value = delong_roc_test(y_true, y_pred1, y_pred2)
                p_value = p_value[0][0]
                
                # Calculate AUC and confidence interval for each model
                auc1, ci1 = delong_roc_ci(y_true, y_pred1)
                auc2, ci2 = delong_roc_ci(y_true, y_pred2)
                
                # Store results
                comparison_result = {
                    'comparison': f"{model1} vs {model2}",
                    f'{model1}_auc': float(auc1),
                    f'{model1}_ci_lower': float(ci1[0]),
                    f'{model1}_ci_upper': float(ci1[1]),
                    f'{model2}_auc': float(auc2),
                    f'{model2}_ci_lower': float(ci2[0]),
                    f'{model2}_ci_upper': float(ci2[1]),
                    'p_value': float(p_value),
                    'significant_difference': bool(p_value < 0.05),
                    'conclusion': f"{model1} and {model2} AUC exists significant difference (p<0.05)" if p_value < 0.05 else f"{model1} and {model2} AUC no significant difference (p≥0.05)"
                }
                comparison_results.append(comparison_result)
                
                # Log results
                self.logger.info(f"Comparison: {model1} vs {model2}")
                self.logger.info(f"{model1} AUC: {auc1:.4f} (95% CI: {ci1[0]:.4f}-{ci1[1]:.4f})")
                self.logger.info(f"{model2} AUC: {auc2:.4f} (95% CI: {ci2[0]:.4f}-{ci2[1]:.4f})")
                self.logger.info(f"DeLong test p-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    self.logger.info(f"Conclusion: Significant difference in AUC between {model1} and {model2} (p<0.05)")
                else:
                    self.logger.info(f"Conclusion: No significant difference in AUC between {model1} and {model2} (p≥0.05)")
                
                self.logger.info("-"*80)
        
        # Save comparison results
        comparison_file = os.path.join(self.output_dir, 'delong_comparison.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=4)

    def _print_performance_table(self, results: Dict[str, Any]) -> None:
        """
        Print model performance table
        
        Args:
            results (Dict[str, Any]): Evaluation results dictionary
        """
        self.logger.info("="*80)
        self.logger.info("Model Performance Evaluation Table")
        self.logger.info("="*80)
        
        # Define metric name mapping
        metric_names = {
            'accuracy': 'Accuracy',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'ppv': 'Positive Predictive Value',
            'npv': 'Negative Predictive Value',
            'f1_score': 'F1-score',
            'auc': 'AUC',
            'hosmer_lemeshow_chi2': 'H-L Chi2',
            'hosmer_lemeshow_p_value': 'H-L P-value',
            'spiegelhalter_z_statistic': 'Spiegelhalter Z',
            'spiegelhalter_z_p_value': 'Spiegelhalter P-value'
        }
        
        # Get available model names from either train or test results
        available_models = set()
        if 'train' in results:
            available_models.update(results['train'].keys())
        if 'test' in results:
            available_models.update(results['test'].keys())
        
        if not available_models:
            self.logger.warning("No model results available")
            return
            
        # Log table header
        header = ["Metric"]
        for model_name in sorted(available_models):
            if 'train' in results:
                header.append(f"{model_name} (Train)")
            if 'test' in results:
                header.append(f"{model_name} (Test)")
        
        self.logger.info(" | ".join([f"{h:^15}" for h in header]))
        self.logger.info("-"*80)
        
        # Log values for each metric
        for metric in ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1_score', 'auc', 
                      'hosmer_lemeshow_chi2', 'hosmer_lemeshow_p_value', 
                      'spiegelhalter_z_statistic', 'spiegelhalter_z_p_value']:
            row = [metric_names.get(metric, metric)]
            
            for model_name in sorted(available_models):
                # Get train metrics if available
                if 'train' in results:
                    train_metrics = results['train'].get(model_name, {}).get('metrics', {})
                    train_value = train_metrics.get(metric, 'N/A')
                    row.append(f"{train_value:.4f}" if isinstance(train_value, (int, float)) else str(train_value))
                
                # Get test metrics if available
                if 'test' in results:
                    test_metrics = results['test'].get(model_name, {}).get('metrics', {})
                    test_value = test_metrics.get(metric, 'N/A')
                    row.append(f"{test_value:.4f}" if isinstance(test_value, (int, float)) else str(test_value))
            
            self.logger.info(" | ".join([f"{cell:^15}" for cell in row]))
        
        self.logger.info("="*80)

    def _save_performance_table(self, results: Dict[str, Any], filename: str = "performance_table.csv") -> None:
        """
        Save model performance table to CSV file
        
        Args:
            results (Dict[str, Any]): Evaluation results dictionary
            filename (str): Output filename for the CSV file
        """
        # Define metric name mapping
        metric_names = {
            'accuracy': 'Accuracy',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'ppv': 'Positive Predictive Value',
            'npv': 'Negative Predictive Value',
            'f1_score': 'F1-score',
            'auc': 'AUC',
            'hosmer_lemeshow_chi2': 'H-L Chi2',
            'hosmer_lemeshow_p_value': 'H-L P-value',
            'spiegelhalter_z_statistic': 'Spiegelhalter Z',
            'spiegelhalter_z_p_value': 'Spiegelhalter P-value'
        }
        
        # Get available model names from either train or test results
        available_models = set()
        if 'train' in results:
            available_models.update(results['train'].keys())
        if 'test' in results:
            available_models.update(results['test'].keys())
        
        if not available_models:
            self.logger.warning("No model results available for saving")
            return
        
        # Create DataFrame for saving
        performance_data = []
        
        # Define all metrics to include
        all_metrics = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'f1_score', 'auc', 
                      'hosmer_lemeshow_chi2', 'hosmer_lemeshow_p_value', 
                      'spiegelhalter_z_statistic', 'spiegelhalter_z_p_value']
        
        # Create row for each metric
        for metric in all_metrics:
            row_data = {
                'Metric': metric_names.get(metric, metric),
                'Metric_Code': metric
            }
            
            for model_name in sorted(available_models):
                # Get train metrics if available
                if 'train' in results:
                    train_metrics = results['train'].get(model_name, {}).get('metrics', {})
                    train_value = train_metrics.get(metric, np.nan)
                    row_data[f"{model_name}_Train"] = train_value
                
                # Get test metrics if available
                if 'test' in results:
                    test_metrics = results['test'].get(model_name, {}).get('metrics', {})
                    test_value = test_metrics.get(metric, np.nan)
                    row_data[f"{model_name}_Test"] = test_value
            
            performance_data.append(row_data)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(performance_data)
        
        # Save to output directory
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Performance table saved to: {output_path}")
        
        # Also save a detailed summary with additional statistics
        detailed_filename = filename.replace('.csv', '_detailed.csv')
        self._save_detailed_performance_summary(results, detailed_filename)
    
    def _save_detailed_performance_summary(self, results: Dict[str, Any], filename: str = "performance_detailed.csv") -> None:
        """
        Save detailed performance summary including statistical test interpretations
        
        Args:
            results (Dict[str, Any]): Evaluation results dictionary
            filename (str): Output filename for the detailed CSV file
        """
        # Get available model names
        available_models = set()
        if 'train' in results:
            available_models.update(results['train'].keys())
        if 'test' in results:
            available_models.update(results['test'].keys())
        
        if not available_models:
            return
        
        detailed_data = []
        
        for dataset_type in ['train', 'test']:
            if dataset_type not in results:
                continue
                
            for model_name in sorted(available_models):
                if model_name not in results[dataset_type]:
                    continue
                    
                metrics = results[dataset_type][model_name].get('metrics', {})
                
                # Create detailed row for this model-dataset combination
                row = {
                    'Model': model_name,
                    'Dataset': dataset_type.capitalize(),
                    'Accuracy': metrics.get('accuracy', np.nan),
                    'Sensitivity': metrics.get('sensitivity', np.nan),
                    'Specificity': metrics.get('specificity', np.nan),
                    'PPV': metrics.get('ppv', np.nan),
                    'NPV': metrics.get('npv', np.nan),
                    'F1_Score': metrics.get('f1_score', np.nan),
                    'AUC': metrics.get('auc', np.nan),
                    'Hosmer_Lemeshow_Chi2': metrics.get('hosmer_lemeshow_chi2', np.nan),
                    'Hosmer_Lemeshow_P_Value': metrics.get('hosmer_lemeshow_p_value', np.nan),
                    'Spiegelhalter_Z_Statistic': metrics.get('spiegelhalter_z_statistic', np.nan),
                    'Spiegelhalter_Z_P_Value': metrics.get('spiegelhalter_z_p_value', np.nan)
                }
                
                # Add interpretations for statistical tests
                hl_p = metrics.get('hosmer_lemeshow_p_value', np.nan)
                if not np.isnan(hl_p):
                    row['Hosmer_Lemeshow_Interpretation'] = "Good calibration (p≥0.05)" if hl_p >= 0.05 else "Poor calibration (p<0.05)"
                else:
                    row['Hosmer_Lemeshow_Interpretation'] = "N/A"
                
                sp_p = metrics.get('spiegelhalter_z_p_value', np.nan)
                if not np.isnan(sp_p):
                    row['Spiegelhalter_Z_Interpretation'] = "Good calibration (p≥0.05)" if sp_p >= 0.05 else "Poor calibration (p<0.05)"
                else:
                    row['Spiegelhalter_Z_Interpretation'] = "N/A"
                
                detailed_data.append(row)
        
        # Create DataFrame and save
        df_detailed = pd.DataFrame(detailed_data)
        output_path = os.path.join(self.output_dir, filename)
        df_detailed.to_csv(output_path, index=False)
        
        self.logger.info(f"Detailed performance summary saved to: {output_path}")


class MultifileEvaluator:
    def __init__(self, output_dir: str) -> None:
        """
        初始化多文件评估器
        
        Args:
            output_dir (str): 图表输出目录
        """
        self.output_dir = output_dir
        self.plotter = Plotter(output_dir)
        self.data = None
        self.models_data = {}
        self.label_col = None
        self.subject_id_col = None
        self.logger = get_module_logger('evaluation.multifile')
        
    def read_prediction_files(self, files_config: List[Dict]) -> 'MultifileEvaluator':
        """
        从多个文件读取预测结果
        
        Args:
            files_config (List[Dict]): 文件配置列表，每个元素包含：
                - path: 文件路径
                - model_name: 模型名称
                - subject_id_col: 受试者ID列名
                - label_col: 真实标签列名
                - prob_col: 预测概率列名
                - pred_col: 预测标签列名（可选）
                
        Returns:
            MultifileEvaluator: 自身实例，用于方法链式调用
        """
        # Simple and clear data fusion approach
        self.logger.info(f"Reading data from multiple files: {len(files_config)} files")
        
        # Step 1: Read and standardize all files
        standardized_dfs = []
        original_subject_id_col = None
        original_label_col = None
        
        for idx, file_config in enumerate(files_config):
            file_path = file_config['path']
            model_name = file_config.get('model_name', file_config.get('name', f"model{idx+1}"))
            subject_id_col = file_config.get('subject_id_col')
            label_col = file_config.get('label_col')
            prob_col = file_config.get('prob_col')
            pred_col = file_config.get('pred_col')
            
            # Check required parameters
            if not all([subject_id_col, label_col, prob_col]):
                raise ValueError(f"Missing required columns for file {file_path}")
            
            # Store original column names from first file
            if idx == 0:
                original_subject_id_col = subject_id_col
                original_label_col = label_col
            
            self.logger.info(f"Reading file: {file_path}")
            self.logger.info(f"Model name: {model_name}, Subject ID: {subject_id_col}, Label: {label_col}, Prob: {prob_col}")
            
            # Read file and check columns exist
            df = pd.read_csv(file_path)
            required_cols = [subject_id_col, label_col, prob_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns {missing_cols} in file {file_path}")
            
            # Standardize dataframe: use consistent column names for merging
            standardized_df = pd.DataFrame({
                'subject_id': df[subject_id_col].astype(str),
                'label': df[label_col],
                f'{model_name}_prob': df[prob_col]
            })
            
            # Add prediction column if specified and exists
            if pred_col and pred_col in df.columns:
                standardized_df[f'{model_name}_pred'] = df[pred_col]
            
            standardized_dfs.append((model_name, standardized_df))
        
        # Step 2: Merge all dataframes using standardized column names
        self.logger.info("Merging all datasets...")
        merged_df = None
        
        for model_name, df in standardized_dfs:
            if merged_df is None:
                # First dataframe becomes the base, set original column names
                merged_df = df.copy()
                self.subject_id_col = original_subject_id_col
                self.label_col = original_label_col
            else:
                # Merge subsequent dataframes on standardized column names
                merge_cols = ['subject_id', 'label']
                prob_cols = [col for col in df.columns if col.endswith('_prob') or col.endswith('_pred')]
                df_to_merge = df[merge_cols + prob_cols]
                merged_df = merged_df.merge(df_to_merge, on=merge_cols, how='outer')
        
        # Step 3: Set subject_id as index and save data
        merged_df.set_index('subject_id', inplace=True)
        
        # Reorder columns: label first, then all model columns
        label_col_list = ['label']
        model_cols = [col for col in merged_df.columns if col != 'label']
        merged_df = merged_df[label_col_list + model_cols]
        
        self.data = merged_df
        
        # Prepare models_data dictionary for plotting module
        # Keys are model names, values are (y_true, y_pred_proba) tuples  
        for model_name, _ in standardized_dfs:
            prob_column_name = f"{model_name}_prob"
            
            if prob_column_name in self.data.columns:
                self.models_data[model_name] = (
                    self.data['label'].values,
                    self.data[prob_column_name].values
                )
        
        return self
    
    def save_merged_data(self, filename: str = "merged_predictions.csv") -> None:
        """
        保存合并后的数据到CSV文件
        
        Args:
            filename (str): 输出文件名
        """
        if self.data is not None:
            # 创建一个包含索引的副本
            output_df = self.data.copy()
            output_df.reset_index(inplace=True)  # 将索引变为常规列
            
            # 打印列中NaN值的数量
            nan_counts = output_df.isna().sum()
            self.logger.info("NaN value statistics:")
            for col, count in nan_counts.items():
                if count > 0:
                    total = len(output_df)
                    percent = (count / total) * 100
                    self.logger.info(f"  {col}: {count}/{total} ({percent:.2f}%)")
            
            output_path = os.path.join(self.output_dir, filename)
            output_df.to_csv(output_path, index=False)
            self.logger.info(f"Merged data saved to {output_path}")
        else:
            self.logger.warning("No data to save. Please read prediction files first.")
    
    def plot_roc(self, save_name: str = "ROC.pdf", title: str = "evaluation") -> None:
        """
        为所有模型绘制ROC曲线
        
        Args:
            save_name (str): 保存文件名
            title (str): 图表标题
        """
        if not self.models_data:
            self.logger.warning("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_roc_v2(self.models_data, save_name=save_name, title=title)
        self.logger.info(f"ROC curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def plot_dca(self, save_name: str = "DCA.pdf", title: str = "evaluation") -> None:
        """
        为所有模型绘制决策曲线分析(DCA)
        
        Args:
            save_name (str): 保存文件名
            title (str): 图表标题
        """
        if not self.models_data:
            self.logger.warning("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_dca_v2(self.models_data, save_name=save_name, title=title)
        self.logger.info(f"DCA curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def plot_calibration(self, save_name: str = "Calibration.pdf", n_bins: int = 5, title: str = "evaluation") -> None:
        """
        为所有模型绘制校准曲线
        
        Args:
            save_name (str): 保存文件名
            n_bins (int): 校准曲线的分箱数
            title (str): 图表标题
        """
        if not self.models_data:
            self.logger.warning("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_calibration_v2(self.models_data, save_name=save_name, n_bins=n_bins, title=title)
        self.logger.info(f"Calibration curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def plot_pr_curve(self, save_name: str = "PR_curve.pdf", title: str = "evaluation") -> None:
        """
        为所有模型绘制精确率-召回率曲线
        
        Args:
            save_name (str): 保存文件名
            title (str): 图表标题
        """
        if not self.models_data:
            self.logger.warning("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_pr_curve(self.models_data, save_name=save_name, title=title)
        self.logger.info(f"PR curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def run_delong_test(self, output_json: Optional[str] = "delong_test_results.json") -> List[Dict]:
        """
        对所有模型对执行DeLong检验
        
        Args:
            output_json (Optional[str]): 输出JSON文件名，如不需要保存设为None
            
        Returns:
            List[Dict]: DeLong检验结果列表
        """
        if not self.models_data or len(self.models_data) < 2:
            self.logger.warning("Need at least two models for DeLong test.")
            return []
        
        results = []
        model_names = list(self.models_data.keys())
        
        # 获取真实标签
        y_true = self.data['label'].values
        # 执行成对比较
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                # 创建一个包含共同有效数据的DataFrame
                temp_df = pd.DataFrame({
                    'y_true': y_true,
                    f'{model1}_prob': self.data[f"{model1}_prob"].values,
                    f'{model2}_prob': self.data[f"{model2}_prob"].values
                })
                
                # 删除任何包含NaN的行
                temp_df = temp_df.dropna()
                
                if len(temp_df) == 0:
                    self.logger.warning(f"{model1} 和 {model2} 没有足够的共同有效样本进行DeLong检验")
                    continue
                
                self.logger.info(f"执行DeLong检验: {model1} vs {model2}，有效样本数: {len(temp_df)}")
                
                # 获取清理后的数据
                clean_y_true = temp_df['y_true'].values
                clean_y_pred1 = temp_df[f'{model1}_prob'].values
                clean_y_pred2 = temp_df[f'{model2}_prob'].values
                
                # 计算AUC和置信区间
                auc1, ci1 = delong_roc_ci(clean_y_true, clean_y_pred1)
                auc2, ci2 = delong_roc_ci(clean_y_true, clean_y_pred2)
                
                # 计算p值
                p_value = delong_roc_test(clean_y_true, clean_y_pred1, clean_y_pred2)
                
                # 创建比较结果
                comparison_result = {
                    'comparison': f"{model1} vs {model2}",
                    f'{model1}_auc': float(auc1),
                    f'{model1}_ci_lower': float(ci1[0]),
                    f'{model1}_ci_upper': float(ci1[1]),
                    f'{model2}_auc': float(auc2),
                    f'{model2}_ci_lower': float(ci2[0]),
                    f'{model2}_ci_upper': float(ci2[1]),
                    'p_value': float(p_value),
                    'significant_difference': bool(p_value < 0.05),
                    'conclusion': f"{model1} and {model2} have significantly different AUCs (p<0.05)" if p_value < 0.05 else f"{model1} and {model2} do not have significantly different AUCs (p≥0.05)"
                }
                results.append(comparison_result)
        
        # 输出结果
        self.logger.info("DeLong Test Results:")
        self.logger.info("=" * 50)
        for result in results:
            self.logger.info(f"{result['comparison']}")
            self.logger.info(f"P-value: {result['p_value']:.4f}")
            self.logger.info(f"Conclusion: {result['conclusion']}")
            self.logger.info("AUCs with 95% CI:")
            model1, model2 = result['comparison'].split(" vs ")
            self.logger.info(f"{model1}: {result[f'{model1}_auc']:.3f} ({result[f'{model1}_ci_lower']:.3f}-{result[f'{model1}_ci_upper']:.3f})")
            self.logger.info(f"{model2}: {result[f'{model2}_auc']:.3f} ({result[f'{model2}_ci_lower']:.3f}-{result[f'{model2}_ci_upper']:.3f})")
        
        # 保存结果
        if output_json:
            import json
            output_path = os.path.join(self.output_dir, output_json)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            self.logger.info(f"Results saved to {output_path}")
        
        return results
    
