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

class ModelEvaluator:
    def __init__(self, output_dir: str):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir (str): Directory where evaluation results and plots will be saved
        """
        self.output_dir = output_dir
        self.plotter = Plotter(self.output_dir)
    
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
            
        print("\n" + "="*80)
        print(" "*30 + "Model AUC Comparison (DeLong test)")
        print("="*80)
        
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
                
                # Print results
                print(f"Comparison: {model1} vs {model2}")
                print(f"{model1} AUC: {auc1:.4f} (95% CI: {ci1[0]:.4f}-{ci1[1]:.4f})")
                print(f"{model2} AUC: {auc2:.4f} (95% CI: {ci2[0]:.4f}-{ci2[1]:.4f})")
                print(f"DeLong test p-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    print(f"Conclusion: Significant difference in AUC between {model1} and {model2} (p<0.05)")
                else:
                    print(f"Conclusion: No significant difference in AUC between {model1} and {model2} (p≥0.05)")
                
                print("-"*80)
        
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
        print("\n" + "="*80)
        print(" "*30 + "Model Performance Evaluation Table")
        print("="*80)
        
        # Define metric name mapping
        metric_names = {
            'accuracy': 'Accuracy',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'ppv': 'Positive Predictive Value',
            'npv': 'Negative Predictive Value',
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
            print("No model results available")
            return
            
        # Print table header
        header = ["Metric"]
        for model_name in sorted(available_models):
            if 'train' in results:
                header.append(f"{model_name} (Train)")
            if 'test' in results:
                header.append(f"{model_name} (Test)")
        
        print(" | ".join([f"{h:^15}" for h in header]))
        print("-"*80)
        
        # Print values for each metric
        for metric in ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'auc', 
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
            
            print(" | ".join([f"{cell:^15}" for cell in row]))
        
        print("="*80)

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
            print("No model results available for saving")
            return
        
        # Create DataFrame for saving
        performance_data = []
        
        # Define all metrics to include
        all_metrics = ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv', 'auc', 
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
        
        print(f"Performance table saved to: {output_path}")
        
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
        
        print(f"Detailed performance summary saved to: {output_path}")


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
        merged_df = None
        all_labels = {}  # 存储所有样本的标签值
        first_subject_id_col = None  # 记录第一个subject_id_col
        first_label_col = None  # 记录第一个标签列名
        
        print(f"Reading data from multiple files: {len(files_config)} files")
        
        for idx, file_config in enumerate(files_config):
            file_path = file_config['path']
            name = file_config.get('model_name', file_config.get('name', f"model{idx+1}")) # 兼容老格式
            subject_id_col = file_config.get('subject_id_col')
            label_col = file_config.get('label_col')
            prob_col = file_config.get('prob_col')
            pred_col = file_config.get('pred_col')  # 获取预测列名称（可选）
            
            # 检查必要参数
            if subject_id_col is None:
                raise ValueError(f"Subject ID column must be specified for file {file_path}")
            if label_col is None:
                raise ValueError(f"Label column must be specified for file {file_path}")
            if prob_col is None:
                raise ValueError(f"Probability column must be specified for file {file_path}")
            
            print(f"  Reading file: {file_path}")
            print(f"  Model name: {name}")
            print(f"  Subject ID column: {subject_id_col}")
            print(f"  Label column: {label_col}")
            print(f"  Probability column: {prob_col}")
            if pred_col:
                print(f"  Prediction column: {pred_col}")
            
            # 读取文件
            df = pd.read_csv(file_path)
            
            # 检查列是否存在
            required_cols = [subject_id_col, label_col, prob_col]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in {file_path}")
            
            # 如果指定了预测列，检查是否存在
            if pred_col and pred_col not in df.columns:
                print(f"Warning: Prediction column '{pred_col}' not found in {file_path}, it will be ignored")
                pred_col = None
            
            # 将subject ID转换为字符串以便一致合并
            df[subject_id_col] = df[subject_id_col].astype(str)
            
            # 记录第一个文件的标签列名作为最终标签列名
            if first_label_col is None:
                first_label_col = label_col
                self.label_col = first_label_col
                label_values = df[label_col]
            
            # 记录第一个文件的subject_id_col作为最终subject_id_col
            if first_subject_id_col is None:
                first_subject_id_col = subject_id_col
                self.subject_id_col = first_subject_id_col
            
            # 提取并保存标签数据
            all_labels[file_path] = df.loc[:, [subject_id_col, label_col]]
            all_labels[file_path].set_index(subject_id_col, inplace=True)
            
            # 为模型结果创建新的列名
            prob_column_name = f"{name}_prob"
            pred_column_name = f"{name}_{pred_col}" if pred_col else None
            
            # 选择要保留的列
            if pred_col:
                df_subset = df[[subject_id_col, prob_col, pred_col]].copy()
                df_subset.rename(columns={prob_col: prob_column_name, pred_col: pred_column_name}, inplace=True)
            else:
                df_subset = df[[subject_id_col, prob_col]].copy()
                df_subset.rename(columns={prob_col: prob_column_name}, inplace=True)
            
            # 首次处理初始化合并数据框
            if merged_df is None:
                merged_df = df_subset
                # 设置索引用于后续合并
                merged_df.set_index(subject_id_col, inplace=True)
                # 索引转为字符串
                merged_df.index = merged_df.index.astype(str)
            else:
                # 为合并设置临时索引
                df_subset.set_index(subject_id_col, inplace=True)
                # 索引转为字符串
                df_subset.index = df_subset.index.astype(str)
                
                # 与现有数据合并
                merged_df = merged_df.join(df_subset, how='outer')
                
                # 更新索引名称
                merged_df.index.name = self.subject_id_col
        
        # 将标签数据添加回合并后的数据框
        print(f"Adding unified label column: {self.label_col}")
        merged_df[self.label_col] = label_values.values
        
        # 重排列，把标签放到最前面
        cols = [self.label_col] + [col for col in merged_df.columns if col != self.label_col]
        merged_df = merged_df[cols]
        
        # 保存合并后的数据
        self.data = merged_df
        
        # 准备plotting模块所需的models_data字典
        # 键是模型名称，值是(y_true, y_pred_proba)元组
        for idx, file_config in enumerate(files_config):
            name = file_config.get('model_name', file_config.get('name', f"model{idx+1}"))
            prob_column_name = f"{name}_prob"
            
            if prob_column_name in self.data.columns:
                self.models_data[name] = (
                    self.data[self.label_col].values,
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
            print("\nNaN 值统计:")
            for col, count in nan_counts.items():
                if count > 0:
                    total = len(output_df)
                    percent = (count / total) * 100
                    print(f"  {col}: {count}/{total} ({percent:.2f}%)")
            
            output_path = os.path.join(self.output_dir, filename)
            output_df.to_csv(output_path, index=False)
            print(f"Merged data saved to {output_path}")
        else:
            print("No data to save. Please read prediction files first.")
    
    def plot_roc(self, save_name: str = "ROC.pdf", title: str = "evaluation") -> None:
        """
        为所有模型绘制ROC曲线
        
        Args:
            save_name (str): 保存文件名
            title (str): 图表标题
        """
        if not self.models_data:
            print("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_roc_v2(self.models_data, save_name=save_name, title=title)
        print(f"ROC curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def plot_dca(self, save_name: str = "DCA.pdf", title: str = "evaluation") -> None:
        """
        为所有模型绘制决策曲线分析(DCA)
        
        Args:
            save_name (str): 保存文件名
            title (str): 图表标题
        """
        if not self.models_data:
            print("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_dca_v2(self.models_data, save_name=save_name, title=title)
        print(f"DCA curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def plot_calibration(self, save_name: str = "Calibration.pdf", n_bins: int = 5, title: str = "evaluation") -> None:
        """
        为所有模型绘制校准曲线
        
        Args:
            save_name (str): 保存文件名
            n_bins (int): 校准曲线的分箱数
            title (str): 图表标题
        """
        if not self.models_data:
            print("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_calibration_v2(self.models_data, save_name=save_name, n_bins=n_bins, title=title)
        print(f"Calibration curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def plot_pr_curve(self, save_name: str = "PR_curve.pdf", title: str = "evaluation") -> None:
        """
        为所有模型绘制精确率-召回率曲线
        
        Args:
            save_name (str): 保存文件名
            title (str): 图表标题
        """
        if not self.models_data:
            print("No models data available. Please read prediction files first.")
            return
        
        self.plotter.plot_pr_curve(self.models_data, save_name=save_name, title=title)
        print(f"PR curve saved to {os.path.join(self.output_dir, save_name)}")
    
    def run_delong_test(self, output_json: Optional[str] = "delong_test_results.json") -> List[Dict]:
        """
        对所有模型对执行DeLong检验
        
        Args:
            output_json (Optional[str]): 输出JSON文件名，如不需要保存设为None
            
        Returns:
            List[Dict]: DeLong检验结果列表
        """
        if not self.models_data or len(self.models_data) < 2:
            print("Need at least two models for DeLong test.")
            return []
        
        results = []
        model_names = list(self.models_data.keys())
        
        # 获取真实标签
        y_true = self.data[self.label_col].values
        
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
                    print(f"警告: {model1} 和 {model2} 没有足够的共同有效样本进行DeLong检验")
                    continue
                
                print(f"执行DeLong检验: {model1} vs {model2}，有效样本数: {len(temp_df)}")
                
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
        print("\nDeLong Test Results:")
        print("=" * 50)
        for result in results:
            print(f"\n{result['comparison']}")
            print(f"P-value: {result['p_value']:.4f}")
            print(f"Conclusion: {result['conclusion']}")
            print(f"AUCs with 95% CI:")
            model1, model2 = result['comparison'].split(" vs ")
            print(f"{model1}: {result[f'{model1}_auc']:.3f} ({result[f'{model1}_ci_lower']:.3f}-{result[f'{model1}_ci_upper']:.3f})")
            print(f"{model2}: {result[f'{model2}_auc']:.3f} ({result[f'{model2}_ci_lower']:.3f}-{result[f'{model2}_ci_upper']:.3f})")
        
        # 保存结果
        if output_json:
            import json
            output_path = os.path.join(self.output_dir, output_json)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nResults saved to {output_path}")
        
        return results
    
