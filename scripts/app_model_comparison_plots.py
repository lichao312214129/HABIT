"""
MultifileEvaluator使用示例
展示如何使用多文件评估工具评估多个模型的性能
"""

import os
import argparse
import yaml
import pandas as pd
import json
import numpy as np
from habit.core.machine_learning.evaluation.model_evaluation import MultifileEvaluator
from habit.core.machine_learning.visualization.plotting import Plotter
from habit.core.machine_learning.evaluation.metrics import (
    calculate_metrics, 
    calculate_metrics_youden, 
    calculate_metrics_at_target,
    apply_youden_threshold,
    apply_target_threshold
)

class ModelComparisonTool:
    """
    Tool for comparing and evaluating multiple machine learning models
    """
    def __init__(self, config_path):
        """
        Initialize the model comparison tool
        
        Args:
            config_path (str): Path to configuration YAML file
        """
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    # 设置输出目录
        self.output_dir = self.config.get('output_dir', './results/model_comparison')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化评估器
        self.evaluator = MultifileEvaluator(output_dir=self.output_dir)
        
        # 初始化分组数据存储
        self.split_groups = {}
        self.split_column = None
        
        # 用于存储离散预测列的映射
        self.pred_col_mapping = {}
        
        # 初始化指标存储字典
        self.all_metrics = {}
    
    def setup(self):
        """
        Setup the tool by reading prediction files and preparing data
        """
        # 配置多个预测文件
        files_config = self.config.get('files_config', [])
    
        # 读取所有预测文件并获取split列信息
        split_cols = []
        for file_conf in files_config:
            if "split_col" in file_conf:
                split_cols.append(file_conf["split_col"])
        
            # 添加split列到合并数据中
            self._add_split_columns(files_config, split_cols)
            
            # 创建离散预测列的映射
            for file_config in files_config:
                if 'pred_col' in file_config and 'model_name' in file_config:
                    model_name = file_config['model_name']
                    pred_col = file_config['pred_col']
                    self.pred_col_mapping[model_name] = pred_col
            
            # 获取split配置
            split_config = self.config.get('split', {})
            use_split = split_config.get('enabled', False)
            
            # 选择要使用的分组列
            # 检查所有split列是否存在于数据中，如果存在则使用第一个
            for col in split_cols:
                if col in self.evaluator.data.columns:
                    self.split_column = col
                    break
            
            # 如果启用分组，创建分组数据
            if use_split and self.split_column:
                self._create_split_groups()
    
    def _add_split_columns(self, files_config, split_cols):
        """
        Add split columns from original data to merged data
        
        Args:
            files_config (list): List of file configurations
            split_cols (list): List of split column names
        """
        # 需要保存的原始数据，包含split列
        original_data_frames = []
        dataset_values_by_id = {}  # 用于存储每个患者的split值
    
        # 首先读取所有原始数据，并保存包含split列的数据框
        for idx, file_config in enumerate(files_config):
            file_path = file_config['path']
            subject_id_col = file_config.get('subject_id_col')
            split_col = file_config.get('split_col')
            
            if split_col and os.path.exists(file_path):
                # 读取原始文件
                df = pd.read_csv(file_path)
                
                # 检查split列是否存在
                if split_col in df.columns:
                    # 保存id和split列
                    df_subset = df[[subject_id_col, split_col]].copy()
                    df_subset[subject_id_col] = df_subset[subject_id_col].astype(str)
                    original_data_frames.append(df_subset)
                    
                    # 检查split值的一致性
                    for _, row in df_subset.iterrows():
                        subj_id = row[subject_id_col]
                        dataset_value = row[split_col]
                        
                        if subj_id in dataset_values_by_id:
                            # 已经存在这个患者的split值，检查是否一致
                            if dataset_values_by_id[subj_id] != dataset_value:
                                raise ValueError(f"数据不一致: 患者 {subj_id} 在不同文件中的split值不同 "
                                                f"({dataset_values_by_id[subj_id]} vs {dataset_value})")
                        else:
                            # 添加新的患者split值
                            dataset_values_by_id[subj_id] = dataset_value
        
        # 读取预测文件（使用MultifileEvaluator的标准功能）
        self.evaluator.read_prediction_files(files_config)
        
        # 如果找到了split列，手动将其添加到合并后的数据中
        if original_data_frames and self.evaluator.data is not None:
            # 创建一个包含索引的副本
            merged_data = self.evaluator.data.copy()
            merged_data.reset_index(inplace=True)  # 将索引变为常规列
            
            # 对于每个包含split列的原始数据框
            for df_subset in original_data_frames:
                subject_id_col = df_subset.columns[0]
                split_col = df_subset.columns[1]
                
                # 创建一个映射字典，用于快速查找
                split_dict = dict(zip(df_subset[subject_id_col], df_subset[split_col]))
                
                # 如果split列尚未存在，则添加它
                if split_col not in merged_data.columns:
                    # 根据ID填充split值
                    merged_data[split_col] = merged_data[subject_id_col].map(split_dict)
            
            # 更新evaluator的data属性
            if subject_id_col in merged_data.columns:
                merged_data.set_index(subject_id_col, inplace=True)
            self.evaluator.data = merged_data
            
            # 输出包含split列的信息
            existing_split_cols = [col for col in merged_data.columns if col in split_cols]
            if existing_split_cols:
                print(f"成功添加split列到合并数据中: {existing_split_cols}")
    
    def _create_split_groups(self):
        """
        Create data groups based on split column
        """
        if not self.split_column:
            return
        
        merged_df = self.evaluator.data.copy()
        
        # 直接从数据中获取所有唯一的分组值
        dataset_values = merged_df[self.split_column].dropna().unique().tolist()
        
        print(f"按照 {self.split_column} 列进行分组，识别到的分组：{dataset_values}")
        
        # 为每个分组创建模型数据
        for dataset_value in dataset_values:
            group_df = merged_df[merged_df[self.split_column] == dataset_value]
            
            if not group_df.empty:
                group_models_data = {}
                
                for model_name in self.evaluator.models_data.keys():
                    prob_column_name = f"{model_name}_prob"
                    
                    if prob_column_name in group_df.columns:
                        group_models_data[model_name] = (
                            group_df[self.evaluator.label_col].values,
                            group_df[prob_column_name].values
                        )
                
                self.split_groups[dataset_value] = group_models_data
    
    def save_merged_data(self):
        """
        Save merged data to a file
        """
        merged_data_config = self.config.get('merged_data', {})
        if merged_data_config.get('enabled', True):
            self.evaluator.save_merged_data(merged_data_config.get('save_name', 'combined_predictions.csv'))
    
    def run_evaluation(self):
        """
        Run the entire evaluation process
        """
        # 获取split配置
        split_config = self.config.get('split', {})
        use_split = split_config.get('enabled', False)
        
        # 根据是否有分组分别处理
        if use_split and self.split_column and self.split_groups:
            # 按分组处理数据
            self._run_evaluation_by_group()
        else:
            # 不分组，处理所有数据
            self._run_evaluation_all_data()
    
    def _run_evaluation_by_group(self):
        """
        Run evaluation for each data group
        """
        for dataset_value, group_models_data in self.split_groups.items():
            # 创建分组子目录
            group_output_dir = os.path.join(self.output_dir, str(dataset_value))
            os.makedirs(group_output_dir, exist_ok=True)
            
            # 重新获取当前dataset_value对应的数据框
            group_df = self.evaluator.data[self.evaluator.data[self.split_column] == dataset_value]
            
            # 创建该分组的专用plotter
            group_plotter = Plotter(group_output_dir)
            
            # 绘制各种可视化图表
            self._generate_visualizations(group_plotter, group_models_data, group_output_dir, dataset_value)
            
            # 执行DeLong检验
            self._run_delong_test(group_models_data, group_df, group_output_dir, dataset_value)
            
            # 计算Youden指数
            self._calculate_youden_metrics(dataset_value)
            
            # 计算目标指标
            self._calculate_target_metrics(dataset_value)
        
        # 集中计算所有分组的基本指标并保存
        self._calculate_all_basic_metrics()
    
    def _run_evaluation_all_data(self):
        """
        Run evaluation for all data without grouping
        """
        # 绘制各种可视化图表
        self._generate_visualizations(self.evaluator.plotter, self.evaluator.models_data, self.output_dir)
        
        # 执行DeLong检验
        self._run_delong_test(self.evaluator.models_data, self.evaluator.data, self.output_dir)
        
        # 计算基本metrics
        self._calculate_basic_metrics(self.evaluator.models_data, self.evaluator.data, self.output_dir)
        
        # 计算Youden指数
        self._calculate_youden_metrics()
        
        # 计算目标指标
        self._calculate_target_metrics()
    
    def _generate_visualizations(self, plotter, models_data, output_dir, dataset_value=None):
        """
        Generate visualization plots
        
        Args:
            plotter (Plotter): Plotter object
            models_data (dict): Models data
            output_dir (str): Output directory
            dataset_value (str, optional): Split value for title. Defaults to None.
        """
        viz_config = self.config.get('visualization', {})
        
        # 绘制ROC曲线
        roc_config = viz_config.get('roc', {})
        if roc_config.get('enabled', True):
            save_name = roc_config.get('save_name', 'roc_curves.pdf')
            title = roc_config.get('title', 'ROC Curves Comparison')
            if dataset_value:
                title = f"{title} - {dataset_value}"
            plotter.plot_roc_v2(models_data, save_name=save_name, title=title)
            print(f"{dataset_value+'组 ' if dataset_value else ''}ROC曲线已保存到 {os.path.join(output_dir, save_name)}")
        
        # 绘制决策曲线
        dca_config = viz_config.get('dca', {})
        if dca_config.get('enabled', True):
            save_name = dca_config.get('save_name', 'decision_curves.pdf')
            title = dca_config.get('title', 'Decision Curve Analysis')
            if dataset_value:
                title = f"{title} - {dataset_value}"
            plotter.plot_dca_v2(models_data, save_name=save_name, title=title)
            print(f"{dataset_value+'组 ' if dataset_value else ''}决策曲线已保存到 {os.path.join(output_dir, save_name)}")
        
        # 绘制校准曲线
        cal_config = viz_config.get('calibration', {})
        if cal_config.get('enabled', True):
            save_name = cal_config.get('save_name', 'calibration_curves.pdf')
            title = cal_config.get('title', 'Calibration Curves')
            if dataset_value:
                title = f"{title} - {dataset_value}"
            n_bins = cal_config.get('n_bins', 10)
            plotter.plot_calibration_v2(models_data, save_name=save_name, n_bins=n_bins, title=title)
            print(f"{dataset_value+'组 ' if dataset_value else ''}校准曲线已保存到 {os.path.join(output_dir, save_name)}")
        
        # 绘制精确率-召回率曲线
        pr_config = viz_config.get('pr_curve', {})
        if pr_config.get('enabled', True):
            save_name = pr_config.get('save_name', 'precision_recall_curves.pdf')
            title = pr_config.get('title', 'Precision-Recall Curves')
            if dataset_value:
                title = f"{title} - {dataset_value}"
            plotter.plot_pr_curve(models_data, save_name=save_name, title=title)
            print(f"{dataset_value+'组 ' if dataset_value else ''}精确率-召回率曲线已保存到 {os.path.join(output_dir, save_name)}")
    
    def _run_delong_test(self, models_data, data_df, output_dir, dataset_value=None):
        """
        Run DeLong test for comparing AUCs
        
        Args:
            models_data (dict): Models data
            data_df (pd.DataFrame): Data DataFrame
            output_dir (str): Output directory
            dataset_value (str, optional): Split value for logging. Defaults to None.
        """
        delong_config = self.config.get('delong_test', {})
        if delong_config.get('enabled', True) and len(models_data) >= 2:
            save_name = delong_config.get('save_name', 'delong_results.json')
            
            # 创建临时评估器
            temp_evaluator = MultifileEvaluator(output_dir=output_dir)
            temp_evaluator.data = data_df
            temp_evaluator.models_data = models_data
            temp_evaluator.label_col = self.evaluator.label_col
            temp_evaluator.subject_id_col = self.evaluator.subject_id_col
            
            # 调用评估器自带的DeLong检验方法
            temp_evaluator.run_delong_test(save_name)
            print(f"{dataset_value+'组 ' if dataset_value else ''}DeLong检验结果已保存到 {os.path.join(output_dir, save_name)}")
    
    def _calculate_all_basic_metrics(self):
        """
        Calculate and save basic metrics for all datasets in a single file
        """
        basic_metrics_config = self.config.get('metrics', {}).get('basic_metrics', {})
        if not basic_metrics_config.get('enabled', False):
            return
            
        print("开始计算所有数据集的基本指标...")
        
        # 为每个数据集计算指标
        for dataset_value, dataset_models_data in self.split_groups.items():
            print(f"计算 {dataset_value} 数据集的基本指标...")
            group_df = self.evaluator.data[self.evaluator.data[self.split_column] == dataset_value]
            
            # 确保该数据集在结果中有条目
            if dataset_value not in self.all_metrics:
                self.all_metrics[dataset_value] = {}
                
                # 为每个模型计算metrics
            for model_name, (y_true, y_pred_proba) in dataset_models_data.items():
                # 确保该模型在结果中有条目
                if model_name not in self.all_metrics[dataset_value]:
                    self.all_metrics[dataset_value][model_name] = {}
                    
                # 处理数据并计算metrics
                model_metrics = self._compute_model_metrics(model_name, y_true, y_pred_proba, group_df, dataset_value)
                if model_metrics:
                    self.all_metrics[dataset_value][model_name]['basic_metrics'] = model_metrics
        
        print("所有数据集的基本指标计算完成")
    
    def _calculate_basic_metrics(self, models_data: dict, data_df: pd.DataFrame, output_dir: str, dataset_value: str = None):
        """
        Calculate basic metrics for each model
        
        Args:
            models_data (dict): Models data
            data_df (pd.DataFrame): Data DataFrame
            output_dir (str): Output directory
            dataset_value (str, optional): Split value for logging. Defaults to None.
        """
        basic_metrics_config = self.config.get('metrics', {}).get('basic_metrics', {})
        if not basic_metrics_config.get('enabled', False):
            print("基本指标计算未启用，跳过计算")
            return
        
        print("开始计算基本指标...")
        
        # 如果是分组评估，将结果添加到分组下
        if dataset_value is not None:
            # 确保该数据集在结果中有条目
            if dataset_value not in self.all_metrics:
                self.all_metrics[dataset_value] = {}
                print(f"创建新的数据集分组: {dataset_value}")
            
            # 为每个模型计算metrics
            for model_name, (y_true, y_pred_proba) in models_data.items():
                print(f"计算模型 {model_name} 的基本指标...")
                # 确保该模型在结果中有条目
                if model_name not in self.all_metrics[dataset_value]:
                    self.all_metrics[dataset_value][model_name] = {}
                
                # 处理数据并计算metrics
                model_metrics = self._compute_model_metrics(model_name, y_true, y_pred_proba, data_df, dataset_value)
                if model_metrics:
                    self.all_metrics[dataset_value][model_name]['basic_metrics'] = model_metrics
                    print(f"模型 {model_name} 的基本指标计算完成")
                else:
                    print(f"警告: 模型 {model_name} 的基本指标计算失败")
            
            print(f"{dataset_value}组基本指标计算完成")
        else:
            # 如果不是分组评估，创建一个"all"分组
            if 'all' not in self.all_metrics:
                self.all_metrics['all'] = {}
                print("创建'all'分组")
            
            # 为每个模型计算metrics
            for model_name, (y_true, y_pred_proba) in models_data.items():
                print(f"计算模型 {model_name} 的基本指标...")
                # 确保该模型在结果中有条目
                if model_name not in self.all_metrics['all']:
                    self.all_metrics['all'][model_name] = {}
                
                # 处理数据并计算metrics
                model_metrics = self._compute_model_metrics(model_name, y_true, y_pred_proba, data_df, dataset_value)
                if model_metrics:
                    self.all_metrics['all'][model_name]['basic_metrics'] = model_metrics
                    print(f"模型 {model_name} 的基本指标计算完成")
                else:
                    print(f"警告: 模型 {model_name} 的基本指标计算失败")
            
            print("基本指标计算完成")
    
    def _calculate_youden_metrics(self, dataset_value=None):
        """
        Calculate Youden metrics
        
        Args:
            dataset_value (str, optional): Split value. Defaults to None.
        """
        youden_config = self.config.get('metrics', {}).get('youden_metrics', {})
        if not youden_config.get('enabled', False):
            print("Youden指标计算未启用，跳过计算")
            return
        
        print("开始计算Youden指标...")
        
        # 检查是否有分组
        if self.split_column and self.split_groups and dataset_value is not None:
            self._calculate_youden_metrics_by_split()
        else:
            # 不使用分组，在全部数据上计算Youden指数
            print("没有启用数据分割，在全部数据上计算Youden指数...")
            
            # 确保'all'分组存在
            if 'all' not in self.all_metrics:
                self.all_metrics['all'] = {}
            
            # 为每个模型计算Youden指标
            for model_name, (y_true, y_pred_proba) in self.evaluator.models_data.items():
                # 创建DataFrame以便处理可能的NaN值
                temp_df = pd.DataFrame({
                    'y_true': y_true,
                    'y_pred_proba': y_pred_proba
                })
                # 删除任何包含NaN的行
                temp_df = temp_df.dropna()
                
                if len(temp_df) > 0:
                    try:
                        # 调用metrics.py中的calculate_metrics_youden函数
                        model_metrics = calculate_metrics_youden(
                            temp_df['y_true'].values,
                            temp_df['y_pred_proba'].values
                        )
                        print(f"{model_name} 计算Youden指数，有效样本数: {len(temp_df)}")
                        
                        # 确保模型条目存在
                        if model_name not in self.all_metrics['all']:
                            self.all_metrics['all'][model_name] = {}
                        
                        # 添加Youden指标
                        self.all_metrics['all'][model_name]['youden_metrics'] = model_metrics
                        
                        # 保存阈值信息
                        if 'threshold' in model_metrics:
                            if 'thresholds' not in self.all_metrics['all'][model_name]:
                                self.all_metrics['all'][model_name]['thresholds'] = {}
                            self.all_metrics['all'][model_name]['thresholds']['youden'] = model_metrics['threshold']
                    except Exception as e:
                        print(f"警告: {model_name} 计算Youden指数时出错: {str(e)}")
                else:
                    print(f"警告: {model_name} 没有有效的预测数据")
            
            print("Youden指标计算完成")
    
    def _calculate_youden_metrics_by_split(self):
        """
        Calculate Youden metrics for train/test split scenario
        """
        print(f"根据 {self.split_column} 分组计算Youden指数...")
        
        # 在训练集上确定阈值
        if 'train' in self.split_groups:
            train_models_data = self.split_groups['train']
            print("使用训练集确定Youden指数最优阈值...")
            
            # 为每个模型计算Youden指数阈值
            train_thresholds = {}
            
            for model_name, (y_true, y_pred_proba) in train_models_data.items():
                # 创建DataFrame以便处理可能的NaN值
                temp_df = pd.DataFrame({
                    'y_true': y_true,
                    'y_pred_proba': y_pred_proba
                })
                # 删除任何包含NaN的行
                temp_df = temp_df.dropna()
                
                if len(temp_df) > 0:
                    try:
                        # 调用metrics.py中的calculate_metrics_youden函数
                        model_metrics = calculate_metrics_youden(
                            temp_df['y_true'].values,
                            temp_df['y_pred_proba'].values
                        )
                        print(f"{model_name} 训练集计算Youden指数，有效样本数: {len(temp_df)}")
                        
                        # 保存每个模型的最优阈值，用于后续应用到测试集
                        if 'threshold' in model_metrics:
                            train_thresholds[model_name] = model_metrics['threshold']
                            print(f"模型 {model_name} Youden指数最优阈值: {train_thresholds[model_name]}")
                            
                            # 确保train分组存在
                            if 'train' not in self.all_metrics:
                                self.all_metrics['train'] = {}
                            
                            # 确保模型条目存在
                            if model_name not in self.all_metrics['train']:
                                self.all_metrics['train'][model_name] = {}
                            
                            # 添加Youden指标
                            self.all_metrics['train'][model_name]['youden_metrics'] = model_metrics
                            
                            # 保存阈值信息
                            if 'thresholds' not in self.all_metrics['train'][model_name]:
                                self.all_metrics['train'][model_name]['thresholds'] = {}
                            self.all_metrics['train'][model_name]['thresholds']['youden'] = model_metrics['threshold']
                    except Exception as e:
                        print(f"警告: {model_name} 计算Youden指数时出错: {str(e)}")
                else:
                    print(f"警告: {model_name} 训练集没有有效的预测数据")
            
            # 如果找到了阈值，将其应用到所有数据集
            if train_thresholds:
                # 将阈值应用到所有数据集
                for dataset_value, dataset_models_data in self.split_groups.items():
                    print(f"将Youden指数最优阈值应用到 {dataset_value} 数据集...")
                    
                    # 确保该数据集在结果中有条目
                    if dataset_value not in self.all_metrics:
                        self.all_metrics[dataset_value] = {}
                    
                    # 为每个模型应用阈值
                    dataset_metrics_results = self._apply_thresholds_to_test(
                        dataset_models_data, train_thresholds, 'apply_youden_threshold'
                    )
                    
                    # 将结果添加到all_results中
                    for model_name, metrics_data in dataset_metrics_results.items():
                        # 确保该模型在结果中有条目
                        if model_name not in self.all_metrics[dataset_value]:
                            self.all_metrics[dataset_value][model_name] = {}
                        
                        # 添加Youden指标结果
                        self.all_metrics[dataset_value][model_name]['youden_metrics'] = metrics_data
                        # 添加使用的阈值
                        if model_name in train_thresholds:
                            if 'thresholds' not in self.all_metrics[dataset_value][model_name]:
                                self.all_metrics[dataset_value][model_name]['thresholds'] = {}
                            self.all_metrics[dataset_value][model_name]['thresholds']['youden'] = train_thresholds[model_name]
                
                print("所有数据集的Youden指数评估完成")
            else:
                print("警告: 没有找到任何Youden指数最优阈值，无法评估数据集")
    
    def _calculate_target_metrics(self, dataset_value=None):
        """
        Calculate metrics based on target sensitivity/specificity
        
        Args:
            dataset_value (str, optional): Split value. Defaults to None.
        """
        target_metrics_config = self.config.get('metrics', {}).get('target_metrics', {})
        if not target_metrics_config.get('enabled', False):
            print("目标指标计算未启用，跳过计算")
            return
        
        print("开始计算目标指标...")
        
        # 检查是否有有效的目标指标配置
        if not target_metrics_config.get('targets', {}):
            print("警告: 目标指标配置为空，跳过目标指标计算")
            return
        
        # 检查是否有分组
        if self.split_column and self.split_groups and dataset_value is not None:
            # 这是按组分析的入口点，需要先在训练集确定阈值，再应用到测试集
            self._calculate_target_metrics_by_split(target_metrics_config.get('targets', {}))
        else:
            # 不使用分组，在全部数据上计算目标指标
            print("没有启用数据分割，在全部数据上计算目标指标阈值...")
            
            # 确保'all'分组存在
            if 'all' not in self.all_metrics:
                self.all_metrics['all'] = {}
            
            # 为每个模型计算目标指标
            for model_name, (y_true, y_pred_proba) in self.evaluator.models_data.items():
                # 创建DataFrame以便处理可能的NaN值
                temp_df = pd.DataFrame({
                    'y_true': y_true,
                    'y_pred_proba': y_pred_proba
                })
                # 删除任何包含NaN的行
                temp_df = temp_df.dropna()
                
                if len(temp_df) > 0:
                    try:
                        # 调用metrics.py中的calculate_metrics_at_target函数
                        model_metrics = calculate_metrics_at_target(
                            temp_df['y_true'].values,
                            temp_df['y_pred_proba'].values,
                            target_metrics_config.get('targets', {})
                        )
                        print(f"{model_name} 计算目标指标，有效样本数: {len(temp_df)}")
                        
                        # 确保模型条目存在
                        if model_name not in self.all_metrics['all']:
                            self.all_metrics['all'][model_name] = {}
                        
                        # 添加目标指标
                        self.all_metrics['all'][model_name]['target_metrics'] = model_metrics
                        
                        # 保存阈值信息
                        if 'combined_results' in model_metrics:
                            combined_key = ' & '.join(target_metrics_config.get('targets', {}).keys())
                            if combined_key in model_metrics['combined_results']:
                                combined_thresholds = model_metrics['combined_results'][combined_key]
                                if combined_thresholds:
                                    first_threshold_key = next(iter(combined_thresholds))
                                    target_threshold = float(first_threshold_key)
                                    
                                    if 'thresholds' not in self.all_metrics['all'][model_name]:
                                        self.all_metrics['all'][model_name]['thresholds'] = {}
                                    self.all_metrics['all'][model_name]['thresholds']['target'] = target_threshold
                    except Exception as e:
                        print(f"警告: {model_name} 计算目标指标时出错: {str(e)}")
                else:
                    print(f"警告: {model_name} 没有有效的预测数据")
            
            print("目标指标计算完成")
    
    def _calculate_target_metrics_by_split(self, targets):
        """
        Calculate target metrics for train/test split scenario
        
        Args:
            targets (dict): Target metrics
        """
        print(f"根据 {self.split_column} 分组计算目标指标阈值...")
        
        # 在训练集上确定阈值
        if 'train' in self.split_groups:
            train_models_data = self.split_groups['train']
            print("使用训练集确定目标指标阈值...")
            
            # 为训练集计算目标指标
            train_thresholds = {}
            
            for model_name, (y_true, y_pred_proba) in train_models_data.items():
                # 创建DataFrame以便处理可能的NaN值
                temp_df = pd.DataFrame({
                    'y_true': y_true,
                    'y_pred_proba': y_pred_proba
                })
                # 删除任何包含NaN的行
                temp_df = temp_df.dropna()
                
                if len(temp_df) > 0:
                    try:
                        # 计算目标指标
                        model_metrics = calculate_metrics_at_target(
                            temp_df['y_true'].values,
                            temp_df['y_pred_proba'].values,
                            targets
                        )
                        print(f"{model_name} 训练集计算目标指标，有效样本数: {len(temp_df)}")
                        
                        # 确保train分组存在
                        if 'train' not in self.all_metrics:
                            self.all_metrics['train'] = {}
                        
                        # 确保模型条目存在
                        if model_name not in self.all_metrics['train']:
                            self.all_metrics['train'][model_name] = {}
                        
                        # 添加目标指标
                        self.all_metrics['train'][model_name]['target_metrics'] = model_metrics
                        
                        # 检查是否有同时满足所有目标的阈值
                        combined_key = ' & '.join(targets.keys())
                        if 'combined_results' in model_metrics and combined_key in model_metrics['combined_results']:
                            # 找到第一个同时满足所有目标的阈值
                            combined_thresholds = model_metrics['combined_results'][combined_key]
                            if combined_thresholds:
                                # 获取第一个阈值
                                first_threshold_key = next(iter(combined_thresholds))
                                train_thresholds[model_name] = float(first_threshold_key)
                                print(f"模型 {model_name} 同时满足所有目标的阈值: {train_thresholds[model_name]}")
                                
                                # 保存阈值信息
                                if 'thresholds' not in self.all_metrics['train'][model_name]:
                                    self.all_metrics['train'][model_name]['thresholds'] = {}
                                self.all_metrics['train'][model_name]['thresholds']['target'] = train_thresholds[model_name]
                            else:
                                print(f"警告: {model_name} 没有找到同时满足所有目标的阈值")
                        else:
                            print(f"警告: {model_name} 没有找到同时满足所有目标的阈值")
                    except Exception as e:
                        print(f"警告: {model_name} 计算目标指标时出错: {str(e)}")
                else:
                    print(f"警告: {model_name} 训练集没有有效的预测数据")
            
            # 如果找到了阈值，将其应用到所有数据集
            if train_thresholds:
                # 将阈值应用到所有数据集
                for dataset_value, dataset_models_data in self.split_groups.items():
                    print(f"将目标指标阈值应用到 {dataset_value} 数据集...")
                    
                    # 确保该数据集在结果中有条目
                    if dataset_value not in self.all_metrics:
                        self.all_metrics[dataset_value] = {}
                    
                    # 为每个模型应用阈值
                    dataset_metrics_results = self._apply_thresholds_to_test(
                        dataset_models_data, train_thresholds, 'apply_target_threshold'
                    )
                    
                    # 将结果添加到all_metrics中
                    for model_name, metrics_data in dataset_metrics_results.items():
                        # 确保该模型在结果中有条目
                        if model_name not in self.all_metrics[dataset_value]:
                            self.all_metrics[dataset_value][model_name] = {}
                        
                        # 添加目标指标结果
                        self.all_metrics[dataset_value][model_name]['target_metrics'] = metrics_data
                        # 添加使用的阈值
                        if model_name in train_thresholds:
                            if 'thresholds' not in self.all_metrics[dataset_value][model_name]:
                                self.all_metrics[dataset_value][model_name]['thresholds'] = {}
                            self.all_metrics[dataset_value][model_name]['thresholds']['target'] = train_thresholds[model_name]
                
                print("所有数据集的目标指标评估完成")
            else:
                print("警告: 没有找到任何同时满足所有目标的阈值，无法评估数据集")
    
    def _apply_thresholds_to_test(self, test_models_data, train_thresholds, apply_function_name):
        """
        Apply thresholds from training set to test set
        
        Args:
            test_models_data (dict): Test set models data
            train_thresholds (dict): Thresholds from training set
            apply_function_name (str): Name of the function to apply thresholds
            
        Returns:
            dict: Results of applying thresholds to test set
        """
        test_results = {}
        
        for model_name, (y_true, y_pred_proba) in test_models_data.items():
            # 检查是否有该模型的训练集阈值
            if model_name in train_thresholds:
                # 创建DataFrame以便处理可能的NaN值
                temp_df = pd.DataFrame({
                    'y_true': y_true,
                    'y_pred_proba': y_pred_proba
                })
                # 删除任何包含NaN的行
                temp_df = temp_df.dropna()
                
                if len(temp_df) > 0:
                    # 使用训练集确定的阈值评估测试集
                    threshold = train_thresholds[model_name]
                    
                    try:
                        # 根据函数名动态选择应用阈值的函数
                        if apply_function_name == 'apply_youden_threshold':
                            test_metrics = apply_youden_threshold(
                                temp_df['y_true'].values,
                                temp_df['y_pred_proba'].values,
                                threshold
                            )
                        elif apply_function_name == 'apply_target_threshold':
                            test_metrics = apply_target_threshold(
                                temp_df['y_true'].values,
                                temp_df['y_pred_proba'].values,
                                threshold
                            )
                        else:
                            raise ValueError(f"未知的应用阈值函数: {apply_function_name}")
                        
                        test_results[model_name] = test_metrics
                        print(f"{model_name} 测试集应用训练集阈值评估完成，有效样本数: {len(temp_df)}")
                    except Exception as e:
                        print(f"警告: {model_name} 测试集应用阈值时出错: {str(e)}")
                else:
                    print(f"警告: {model_name} 测试集没有有效的预测数据")
            else:
                print(f"警告: {model_name} 测试集评估失败，没有找到训练集阈值")
        
        return test_results

    def _compute_model_metrics(self, model_name: str, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             data_df: pd.DataFrame, dataset_value: str = None) -> dict:
        """
        Compute basic metrics for a single model
        
        Args:
            model_name (str): Name of the model
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            data_df (pd.DataFrame): Data DataFrame
            dataset_value (str, optional): Split value for logging. Defaults to None.
            
        Returns:
            dict: Dictionary containing basic metrics
        """
        try:
            # Create a temporary DataFrame to handle NaN values
            temp_df = pd.DataFrame({
                'y_true': y_true,
                'y_pred_proba': y_pred_proba
            })
            # Remove any rows containing NaN
            temp_df = temp_df.dropna()
            
            if len(temp_df) > 0:
                # Calculate basic metrics using the metrics module
                # 确保y_true和y_pred_proba都是numpy数组
                y_true_array = temp_df['y_true'].values
                y_pred_proba_array = temp_df['y_pred_proba'].values
                
                # 计算预测标签（使用0.5作为阈值）
                y_pred_array = (y_pred_proba_array >= 0.5).astype(int)
                
                # 打印调试信息
                print(f"调试信息 - {model_name}:")
                print(f"y_true shape: {y_true_array.shape}")
                print(f"y_pred_proba shape: {y_pred_proba_array.shape}")
                print(f"y_pred shape: {y_pred_array.shape}")
                print(f"y_true unique values: {np.unique(y_true_array)}")
                print(f"y_pred unique values: {np.unique(y_pred_array)}")
                print(f"y_pred_proba range: [{np.min(y_pred_proba_array)}, {np.max(y_pred_proba_array)}]")
                
                metrics = calculate_metrics(
                    y_true=y_true_array,
                    y_pred_proba=y_pred_proba_array,
                    y_pred=y_pred_array
                )
                
                print(f"{model_name} {'(' + dataset_value + '组) ' if dataset_value else ''}计算基本指标，有效样本数: {len(temp_df)}")
                return metrics
            else:
                print(f"警告: {model_name} {'(' + dataset_value + '组) ' if dataset_value else ''}没有有效的预测数据")
                return None
                
        except Exception as e:
            print(f"警告: {model_name} {'(' + dataset_value + '组) ' if dataset_value else ''}计算基本指标时出错: {str(e)}")
            return None

    def save_all_metrics(self):
        """
        Save all metrics to a single JSON file
        """
        # 创建metrics目录
        metrics_dir = os.path.join(self.output_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # 设置保存路径
        save_path = os.path.join(metrics_dir, 'metrics.json')
        
        # 如果文件已存在，先加载它
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
                # 合并现有指标和新计算的指标
                for group_key in self.all_metrics:
                    if group_key in existing_metrics:
                        for model_key in self.all_metrics[group_key]:
                            if model_key in existing_metrics[group_key]:
                                existing_metrics[group_key][model_key].update(self.all_metrics[group_key][model_key])
                            else:
                                existing_metrics[group_key][model_key] = self.all_metrics[group_key][model_key]
                    else:
                        existing_metrics[group_key] = self.all_metrics[group_key]
                # 更新本地指标字典
                self.all_metrics = existing_metrics
            except Exception as e:
                print(f"警告: 无法读取现有metrics文件: {str(e)}")
        
        # 保存结果
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_metrics, f, indent=4)
            print(f"所有指标已保存到 {save_path}")
            
            # 打印结果统计
            print(f"保存的指标分组数量: {len(self.all_metrics)}")
            for key, value in self.all_metrics.items():
                print(f"分组 {key} 中的模型数量: {len(value)}")
                for model_name, metrics in value.items():
                    metric_types = list(metrics.keys())
                    print(f"  模型 {model_name} 的指标类型: {', '.join(metric_types)}")
        except Exception as e:
            print(f"保存指标时出错: {str(e)}")

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Model comparison and evaluation tool")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建模型比较工具
    tool = ModelComparisonTool(args.config)
    
    # 读取和准备数据
    tool.setup()
    
    # 保存合并数据
    tool.save_merged_data()
    
    # 执行评估
    tool.run_evaluation()
    
    # 保存所有指标
    tool.save_all_metrics()
    
    print("评估完成！所有结果已保存到", tool.output_dir)

if __name__ == "__main__":
    import sys
    # 调试模式：如果没有提供命令行参数，使用默认配置文件
    if len(sys.argv) == 1:
        print("调试模式：使用默认配置文件 config_model_comparison.yaml")
        sys.argv = [sys.argv[0], "--config", "config_model_comparison.yaml"]
    main() 