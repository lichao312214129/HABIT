"""
MultifileEvaluator使用示例
展示如何使用多文件评估工具评估多个模型的性能
"""

import os
import yaml
import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Optional, List, Union
from ..evaluation.model_evaluation import MultifileEvaluator
from ..visualization.plotting import Plotter
from ..evaluation.metrics import (
    calculate_metrics,
    calculate_metrics_youden,
    calculate_metrics_at_target,
    apply_youden_threshold,
    apply_target_threshold
)
from ..evaluation.prediction_container import (
    PredictionContainer,
    create_prediction_container,
    from_tuple,
    convert_models_data_to_containers,
    convert_containers_to_models_data
)
from ..evaluation.threshold_manager import ThresholdManager
from ..reporting.report_exporter import ReportExporter, MetricsStore
from ..visualization.plot_manager import PlotManager
from ..config_schemas import ModelComparisonConfig
from habit.utils.log_utils import setup_output_logger, setup_logger, get_module_logger, LoggerManager

class ModelComparison:
    """
    Tool for comparing and evaluating multiple machine learning models.
    
    Note: Dependencies should be provided via ServiceConfigurator or explicitly.
    """
    def __init__(
        self, 
        config: Union[Dict[str, Any], ModelComparisonConfig],
        evaluator: MultifileEvaluator,
        reporter: ReportExporter,
        threshold_manager: ThresholdManager,
        plot_manager: PlotManager,
        metrics_store: MetricsStore,
        logger: Any,
    ) -> None:
        """
        Initialize the model comparison tool.
        
        Args:
            config: Parsed config dict or validated config object.
            evaluator: MultifileEvaluator instance (required).
            reporter: ReportExporter instance (required).
            threshold_manager: ThresholdManager instance (required).
            plot_manager: PlotManager instance (required).
            metrics_store: MetricsStore instance (required).
            logger: Logger instance (required).
        """
        if isinstance(config, ModelComparisonConfig):
            self.config = config
        else:
            self.config = ModelComparisonConfig(**config)

        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.evaluator = evaluator
        self.reporter = reporter
        self.threshold_manager = threshold_manager
        self.plot_manager = plot_manager
        self.metrics_store = metrics_store
        self.logger = logger

        self.split_groups = {}
        self.split_column = None
        self.pred_col_mapping = {}
    
    def setup(self) -> None:
        """
        Setup the tool by reading prediction files and preparing data
        """
        # 配置多个预测文件
        files_config = [self._model_to_dict(file_config) for file_config in self.config.files_config]
    
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
            split_config = self.config.split
            use_split = split_config.enabled
            
            # 选择要使用的分组列
            # 检查所有split列是否存在于数据中，如果存在则使用第一个
            for col in split_cols:
                if col in self.evaluator.data.columns:
                    self.split_column = col
                    break
            
            # 如果启用分组，创建分组数据
            if use_split and self.split_column:
                self._create_split_groups()
    
    def _add_split_columns(self, files_config: List[Dict[str, Any]], split_cols: List[str]) -> None:
        """
        Add split columns from original data to merged data
        
        Args:
            files_config (List[Dict[str, Any]]): List of file configurations
            split_cols (List[str]): List of split column names
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
                        dataset_group_name = row[split_col]
                        
                        if subj_id in dataset_values_by_id:
                            # 已经存在这个患者的split值，检查是否一致
                            if dataset_values_by_id[subj_id] != dataset_group_name:
                                self.logger.error(f"数据不一致: 患者 {subj_id} 在不同文件中的split值不同 "
                                                f"({dataset_values_by_id[subj_id]} vs {dataset_group_name})")
                        else:
                            # 添加新的患者split值
                            dataset_values_by_id[subj_id] = dataset_group_name
        
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
                    merged_data[split_col] = merged_data['subject_id'].map(split_dict)
            
            # 更新evaluator的data属性
            if 'subject_id' in merged_data.columns:
                merged_data.set_index('subject_id', inplace=True)
            self.evaluator.data = merged_data
            
            # 输出包含split列的信息
            existing_split_cols = [col for col in merged_data.columns if col in split_cols]
            if existing_split_cols:
                self.logger.info(f"成功添加split列到合并数据中: {existing_split_cols}")
    
    def _create_split_groups(self) -> None:
        """
        Create data groups based on split column
        """
        if not self.split_column:
            return
        
        merged_df = self.evaluator.data.copy()
        
        # 直接从数据中获取所有唯一的分组值
        dataset_values = merged_df[self.split_column].dropna().unique().tolist()
        
        self.logger.info(f"按照 {self.split_column} 列进行分组，识别到的分组：{dataset_values}")
        
        # 为每个分组创建模型数据
        for dataset_group_name in dataset_values:
            group_df = merged_df[merged_df[self.split_column] == dataset_group_name]
            
            if not group_df.empty:
                group_models_data = {}
                
                for model_name, data_tuple in self.evaluator.models_data.items():
                    prob_column_name = f"{model_name}_prob"
                    pred_column_name = f"{model_name}_pred"
                    
                    if prob_column_name in group_df.columns:
                        # Always have true labels and probabilities
                        y_true = group_df['label'].values
                        y_pred_proba = group_df[prob_column_name].values
                        
                        # Check for prediction labels
                        if pred_column_name in group_df.columns:
                            y_pred = group_df[pred_column_name].values
                            group_models_data[model_name] = (y_true, y_pred_proba, y_pred)
                        else:
                            # Fallback to tuple with None for y_pred
                            group_models_data[model_name] = (y_true, y_pred_proba, None)
                
                self.split_groups[dataset_group_name] = group_models_data
    
    def save_merged_data(self) -> None:
        """
        Save merged data to a file
        """
        merged_data_config = self.config.merged_data
        if merged_data_config.enabled:
            self.evaluator.save_merged_data(merged_data_config.save_name)
    
    def run_evaluation(self) -> None:
        """
        Run the entire evaluation process
        """
        # 获取split配置
        split_config = self.config.split
        use_split = split_config.enabled
        
        # 根据是否有分组分别处理
        if use_split and self.split_column and self.split_groups:
            # 按分组处理数据
            self._run_evaluation_by_group()
        else:
            # 不分组，处理所有数据
            self._run_evaluation_all_data()
    
    def _run_evaluation_by_group(self) -> None:
        """
        Run evaluation for each data group
        """
        for dataset_group_name, group_models_data in self.split_groups.items():
            # 创建分组子目录
            group_output_dir = os.path.join(self.output_dir, str(dataset_group_name))
            os.makedirs(group_output_dir, exist_ok=True)
            
            # 重新获取当前dataset_value对应的数据框
            group_df = self.evaluator.data[self.evaluator.data[self.split_column] == dataset_group_name]
            # 创建该分组的专用plotter
            group_plotter = Plotter(group_output_dir)
            
            # 绘制各种可视化图表
            self._generate_visualizations(group_plotter, group_models_data, group_output_dir, dataset_group_name)
            
            # 执行DeLong检验
            self._run_delong_test(group_models_data, group_df, group_output_dir, dataset_group_name)
            
            # 计算Youden指数
            self._calculate_youden_metrics(dataset_group_name)
            
            # 计算目标指标
            self._calculate_target_metrics(dataset_group_name)
        
        # 集中计算所有分组的基本指标并保存
        self._calculate_all_basic_metrics()
    
    def _run_evaluation_all_data(self) -> None:
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

    def _convert_models_data_for_plotting(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert models_data from tuple format to dict format for plotting.

        Args:
            models_data: Dict mapping model_name -> (y_true, y_prob)

        Returns:
            Dict mapping model_name -> (y_true_array, y_prob_array)
        """
        plotting_data = {}
        for model_name, (y_true, y_prob, *_) in models_data.items():
            plotting_data[model_name] = (np.array(y_true), np.array(y_prob))
        return plotting_data

    def _model_to_dict(self, model: Any) -> Dict[str, Any]:
        """
        Convert a pydantic model to a plain dict with compatibility for v1/v2.

        Args:
            model (Any): Pydantic model instance to convert

        Returns:
            Dict[str, Any]: Serialized model data as a dictionary
        """
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def _generate_visualizations(
        self,
        plotter: Plotter,
        models_data: Dict[str, Any],
        output_dir: str,
        dataset_group_name: Optional[str] = None
    ) -> None:
        """
        Generate visualization plots using PlotManager

        Args:
            plotter (Plotter): Plotter object (kept for backward compatibility)
            models_data (dict): Models data
            output_dir (str): Output directory
            dataset_group_name (str, optional): Split value for title. Defaults to None.
        """
        viz_config = self.config.visualization

        plotting_data = self._convert_models_data_for_plotting(models_data)

        title_suffix = f"{dataset_group_name}组 " if dataset_group_name else ""
        prefix = f"{dataset_group_name}_" if dataset_group_name else ""

        if viz_config.roc.enabled:
            save_name = viz_config.roc.save_name or f'{prefix}roc_curves.pdf'
            title = viz_config.roc.title or f'{title_suffix}ROC Curves Comparison'
            plotter.plot_roc_v2(plotting_data, save_name=save_name, title=title)
            self.logger.info(f"{title_suffix}ROC曲线已保存到 {os.path.join(output_dir, save_name)}")

        if viz_config.dca.enabled:
            save_name = viz_config.dca.save_name or f'{prefix}decision_curves.pdf'
            title = viz_config.dca.title or f'{title_suffix}Decision Curve'
            plotter.plot_dca_v2(plotting_data, save_name=save_name, title=title)
            self.logger.info(f"{title_suffix}决策曲线已保存到 {os.path.join(output_dir, save_name)}")

        if viz_config.calibration.enabled:
            save_name = viz_config.calibration.save_name or f'{prefix}calibration_curves.pdf'
            title = viz_config.calibration.title or f'{title_suffix}Calibration Curves'
            n_bins = viz_config.calibration.n_bins or 10
            plotter.plot_calibration_v2(plotting_data, save_name=save_name, n_bins=n_bins, title=title)
            self.logger.info(f"{title_suffix}校准曲线已保存到 {os.path.join(output_dir, save_name)}")

        if viz_config.pr_curve.enabled:
            save_name = viz_config.pr_curve.save_name or f'{prefix}precision_recall_curves.pdf'
            title = viz_config.pr_curve.title or f'{title_suffix}Precision-Recall Curves'
            plotter.plot_pr_curve(plotting_data, save_name=save_name, title=title)
            self.logger.info(f"{title_suffix}精确率-召回率曲线已保存到 {os.path.join(output_dir, save_name)}")
    
    def _run_delong_test(
        self,
        models_data: Dict[str, Any],
        data_df: pd.DataFrame,
        output_dir: str,
        dataset_group_name: Optional[str] = None
    ) -> None:
        """
        Run DeLong test for comparing AUCs
        
        Args:
            models_data (dict): Models data
            data_df (pd.DataFrame): Data DataFrame
            output_dir (str): Output directory
            dataset_group_name (str, optional): Split value for logging. Defaults to None.
        """
        delong_config = self.config.delong_test
        if delong_config.enabled and len(models_data) >= 2:
            save_name = delong_config.save_name
            
            # 创建临时评估器
            temp_evaluator = MultifileEvaluator(output_dir=output_dir)
            temp_evaluator.data = data_df
            temp_evaluator.models_data = models_data
            temp_evaluator.label_col = self.evaluator.label_col
            temp_evaluator.subject_id_col = self.evaluator.subject_id_col
            
            # 调用评估器自带的DeLong检验方法
            temp_evaluator.run_delong_test(save_name)
            self.logger.info(f"{dataset_group_name+'组 ' if dataset_group_name else ''}DeLong检验结果已保存到 {os.path.join(output_dir, save_name)}")
    
    def _calculate_all_basic_metrics(self) -> None:
        """
        Calculate and save basic metrics for all datasets in a single file
        """
        basic_metrics_config = self.config.metrics.basic_metrics
        if not basic_metrics_config.enabled:
            return

        self.logger.info("开始计算所有数据集的基本指标...")

        for dataset_group_name, dataset_models_data in self.split_groups.items():
            self.logger.info(f"计算 {dataset_group_name} 数据集的基本指标...")
            group_df = self.evaluator.data[self.evaluator.data[self.split_column] == dataset_group_name]

            for model_name, (y_true, y_pred_proba, y_pred) in dataset_models_data.items():
                model_metrics = self._compute_model_metrics(model_name, y_true, y_pred_proba, group_df, dataset_group_name, y_pred)
                if model_metrics:
                    self.metrics_store.add_metrics(dataset_group_name, model_name, 'basic_metrics', model_metrics)

        self.logger.info("所有数据集的基本指标计算完成")
    
    def _calculate_basic_metrics(
        self,
        models_data: Dict[str, Any],
        data_df: pd.DataFrame,
        output_dir: str,
        dataset_group_name: Optional[str] = None
    ) -> None:
        """
        Calculate basic metrics for each model

        Args:
            models_data (dict): Models data
            data_df (pd.DataFrame): Data DataFrame
            output_dir (str): Output directory
            dataset_group_name (str, optional): Split value for logging. Defaults to None.
        """
        basic_metrics_config = self.config.metrics.basic_metrics
        if not basic_metrics_config.enabled:
            self.logger.info("基本指标计算未启用，跳过计算")
            return

        self.logger.info("开始计算基本指标...")
        group = dataset_group_name or 'all'

        for model_name, data_tuple in models_data.items():
            y_true, y_pred_proba, y_pred = data_tuple if len(data_tuple) == 3 else (data_tuple[0], data_tuple[1], None)
            model_metrics = self._compute_model_metrics(model_name, y_true, y_pred_proba, data_df, dataset_group_name, y_pred)
            if model_metrics:
                self.metrics_store.add_metrics(group, model_name, 'basic_metrics', model_metrics)
                self.logger.info(f"{model_name} 基本指标计算完成")
            else:
                self.logger.warning(f"{model_name} 基本指标计算失败")

        self.logger.info(f"{group}组基本指标计算完成")
    
    def _calculate_youden_metrics(self, dataset_group_name: Optional[str] = None) -> None:
        """
        Calculate Youden metrics

        Args:
            dataset_group_name (str, optional): Split value. Defaults to None.
        """
        youden_config = self.config.metrics.youden_metrics
        if not youden_config.enabled:
            self.logger.info("Youden指标计算未启用，跳过计算")
            return

        self.logger.info("开始计算Youden指标...")

        if self.split_column and self.split_groups and dataset_group_name is not None:
            self._calculate_youden_metrics_by_split()
        else:
            self.logger.info("没有启用数据分割，在全部数据上计算Youden指数...")

            for model_name, (y_true, y_pred_proba) in self.evaluator.models_data.items():
                try:
                    container = self._create_prediction_container(y_true, y_pred_proba)
                    model_metrics = calculate_metrics_youden(container.y_true, container.y_prob)
                    threshold = model_metrics.get('threshold')

                    self.logger.info(f"{model_name} 计算Youden指数，有效样本数: {len(container.y_true)}")

                    self.metrics_store.add_metrics('all', model_name, 'youden_metrics', model_metrics)
                    if threshold is not None:
                        self.metrics_store.add_threshold('all', model_name, 'youden', threshold)
                except Exception as e:
                    self.logger.warning(f"{model_name} 计算Youden指数时出错: {str(e)}")

            self.logger.info("Youden指标计算完成")
    
    def _calculate_youden_metrics_by_split(self) -> None:
        """
        Calculate Youden metrics for train/test split scenario using ThresholdManager and MetricsStore
        """
        if 'Training set' not in self.split_groups:
            self.logger.warning("没有找到训练集数据，跳过Youden指数计算")
            return

        self.logger.info(f"根据 {self.split_column} 分组计算Youden指数...")
        train_models_data = self.split_groups['Training set']

        for model_name, (y_true, y_pred_proba) in train_models_data.items():
            try:
                container = self._create_prediction_container(y_true, y_pred_proba)
                self.threshold_manager.find_and_store(model_name, container, method='youden')
                threshold = self.threshold_manager.get_threshold(model_name, 'youden')
                model_metrics = calculate_metrics_youden(container.y_true, container.y_prob)

                self.logger.info(f"{model_name} 训练集计算Youden指数，有效样本数: {len(container.y_true)}, 阈值: {threshold:.4f}")

                self.metrics_store.add_metrics('Training set', model_name, 'youden_metrics', model_metrics)
                self.metrics_store.add_threshold('Training set', model_name, 'youden', threshold)
            except Exception as e:
                self.logger.warning(f"{model_name} 计算Youden指数时出错: {str(e)}")

        train_thresholds = {
            m: self.threshold_manager.get_threshold(m, 'youden')
            for m in self.threshold_manager.store.keys()
        }

        if not train_thresholds:
            self.logger.warning("没有找到任何Youden指数最优阈值")
            return

        for dataset_group_name, dataset_models_data in self.split_groups.items():
            self.logger.info(f"将阈值应用到 {dataset_group_name} 数据集...")
            dataset_results = self._apply_thresholds_to_test(dataset_models_data, train_thresholds, 'apply_youden_threshold')

            for model_name, metrics_data in dataset_results.items():
                self.metrics_store.add_metrics(dataset_group_name, model_name, 'youden_metrics', metrics_data)
                if model_name in train_thresholds:
                    self.metrics_store.add_threshold(dataset_group_name, model_name, 'youden', train_thresholds[model_name])

        self.logger.info("所有数据集的Youden指数评估完成")
    
    def _calculate_target_metrics(self, dataset_group_name: Optional[str] = None) -> None:
        """
        Calculate metrics based on target sensitivity/specificity

        Args:
            dataset_group_name (str, optional): Split value. Defaults to None.
        """
        target_metrics_config = self.config.metrics.target_metrics
        if not target_metrics_config.enabled:
            self.logger.info("目标指标计算未启用，跳过计算")
            return

        self.logger.info("开始计算目标指标...")

        targets = target_metrics_config.targets
        if not targets:
            self.logger.warning("目标指标配置为空，跳过目标指标计算")
            return

        if self.split_column and self.split_groups and dataset_group_name is not None:
            self._calculate_target_metrics_by_split(targets)
        else:
            self.logger.info("没有启用数据分割，在全部数据上计算目标指标阈值...")

            for model_name, (y_true, y_pred_proba) in self.evaluator.models_data.items():
                try:
                    container = self._create_prediction_container(y_true, y_pred_proba)
                    model_metrics = calculate_metrics_at_target(container.y_true, container.y_prob, targets)

                    self.logger.info(f"{model_name} 计算目标指标，有效样本数: {len(container.y_true)}")

                    self.metrics_store.add_metrics('all', model_name, 'target_metrics', model_metrics)

                    combined_key = ' & '.join(targets.keys())
                    if 'combined_results' in model_metrics and combined_key in model_metrics['combined_results']:
                        combined_thresholds = model_metrics['combined_results'][combined_key]
                        if combined_thresholds:
                            first_threshold_key = next(iter(combined_thresholds))
                            target_threshold = float(first_threshold_key)
                            self.metrics_store.add_threshold('all', model_name, 'target', target_threshold)
                except Exception as e:
                    self.logger.warning(f"{model_name} 计算目标指标时出错: {str(e)}")

            self.logger.info("目标指标计算完成")
    
    def _calculate_target_metrics_by_split(self, targets: Dict[str, float]) -> None:
        """
        Calculate target metrics for train/test split scenario

        Args:
            targets (dict): Target metrics
        """
        if 'Training set' not in self.split_groups:
            self.logger.warning("没有找到训练集数据，跳过目标指标计算")
            return

        self.logger.info(f"根据 {self.split_column} 分组计算目标指标阈值...")
        train_models_data = self.split_groups['Training set']
        train_thresholds = {}
        combined_key = ' & '.join(targets.keys())

        for model_name, (y_true, y_pred_proba) in train_models_data.items():
            try:
                container = self._create_prediction_container(y_true, y_pred_proba)
                model_metrics = calculate_metrics_at_target(container.y_true, container.y_prob, targets)

                self.logger.info(f"{model_name} 训练集计算目标指标，有效样本数: {len(container.y_true)}")

                self.metrics_store.add_metrics('Training set', model_name, 'target_metrics', model_metrics)

                if 'combined_results' in model_metrics and combined_key in model_metrics['combined_results']:
                    combined_thresholds = model_metrics['combined_results'][combined_key]
                    if combined_thresholds:
                        first_threshold_key = next(iter(combined_thresholds))
                        train_thresholds[model_name] = float(first_threshold_key)
                        self.metrics_store.add_threshold('Training set', model_name, 'target', train_thresholds[model_name])
                        self.logger.info(f"{model_name} 同时满足所有目标的阈值: {train_thresholds[model_name]:.4f}")
            except Exception as e:
                self.logger.warning(f"{model_name} 计算目标指标时出错: {str(e)}")

        if not train_thresholds:
            self.logger.warning("没有找到任何同时满足所有目标的阈值")
            return

        for dataset_group_name, dataset_models_data in self.split_groups.items():
            self.logger.info(f"将目标阈值应用到 {dataset_group_name} 数据集...")
            dataset_results = self._apply_thresholds_to_test(dataset_models_data, train_thresholds, 'apply_target_threshold')

            for model_name, metrics_data in dataset_results.items():
                self.metrics_store.add_metrics(dataset_group_name, model_name, 'target_metrics', metrics_data)
                if model_name in train_thresholds:
                    self.metrics_store.add_threshold(dataset_group_name, model_name, 'target', train_thresholds[model_name])

        self.logger.info("所有数据集的目标指标评估完成")

    def _apply_thresholds_to_test(
        self,
        test_models_data: Dict[str, Any],
        train_thresholds: Dict[str, float],
        apply_function_name: str
    ) -> Dict[str, Any]:
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
                # Clean data using PredictionContainer
                container = create_prediction_container(y_true, y_pred_proba)
                
                if len(container) > 0:
                    # 使用训练集确定的阈值评估测试集
                    threshold = train_thresholds[model_name]
                    
                    try:
                        # 根据函数名动态选择应用阈值的函数
                        if apply_function_name == 'apply_youden_threshold':
                            test_metrics = apply_youden_threshold(
                                container.y_true,
                                container.get_eval_probs(),
                                threshold
                            )
                        elif apply_function_name == 'apply_target_threshold':
                            test_metrics = apply_target_threshold(
                                container.y_true,
                                container.get_eval_probs(),
                                threshold
                            )
                        else:
                            raise ValueError(f"未知的应用阈值函数: {apply_function_name}")
                        
                        test_results[model_name] = test_metrics
                        self.logger.info(f"{model_name} 测试集应用训练集阈值评估完成，有效样本数: {len(container)}")
                    except Exception as e:
                        self.logger.warning(f"警告: {model_name} 测试集应用阈值时出错: {str(e)}")
                else:
                    self.logger.warning(f"警告: {model_name} 测试集没有有效的预测数据")
            else:
                self.logger.warning(f"警告: {model_name} 测试集评估失败，没有找到训练集阈值")
        
        return test_results

    def _create_prediction_container(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> PredictionContainer:
        """
        Create a PredictionContainer from raw arrays, handling NaN values.

        Args:
            y_true: True labels array
            y_pred_proba: Predicted probabilities array

        Returns:
            PredictionContainer instance
        """
        return create_prediction_container(y_true, y_pred_proba)

    def _compute_model_metrics(self, model_name: str, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             data_df: pd.DataFrame, dataset_group_name: str = None, y_pred: np.ndarray = None) -> dict:
        """
        Compute basic metrics for a single model

        Args:
            model_name (str): Name of the model
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            data_df (pd.DataFrame): Data DataFrame
            dataset_group_name (str, optional): Split value for logging. Defaults to None.
            y_pred (np.ndarray, optional): Predicted labels. If None, will be generated from probabilities.

        Returns:
            dict: Dictionary containing basic metrics
        """
        try:
            container = create_prediction_container(y_true, y_pred_proba, y_pred)
            
            if len(container) == 0:
                self.logger.warning(f"警告: {model_name} {'(' + dataset_group_name + '组) ' if dataset_group_name else ''}没有有效的预测数据")
                return None
            
            # Use provided prediction labels if available, otherwise compute from probability
            if y_pred is not None:
                self.logger.debug(f"Using provided prediction labels for model '{model_name}'.")
            else:
                self.logger.warning(
                    f"Prediction labels for model '{model_name}' not found. "
                    f"Falling back to generating labels from probabilities using a 0.5 threshold. "
                    f"This may not accurately reflect of model's original predictions."
                )
            
            # 打印调试信息
            self.logger.debug(f"调试信息 - {model_name}:")
            self.logger.debug(f"y_true shape: {container.y_true.shape}")
            self.logger.debug(f"y_pred_proba shape: {container.get_eval_probs().shape}")
            self.logger.debug(f"y_pred shape: {container.y_pred.shape}")
            self.logger.debug(f"y_true unique values: {np.unique(container.y_true)}")
            self.logger.debug(f"y_pred unique values: {np.unique(container.y_pred)}")
            self.logger.debug(f"y_pred_proba range: [{np.min(container.get_eval_probs())}, {np.max(container.get_eval_probs())}]")
            
            metrics = calculate_metrics(
                y_true=container.y_true,
                y_pred_proba=container.get_eval_probs(),
                y_pred=container.y_pred
            )
            
            self.logger.info(f"{model_name} {'(' + dataset_group_name + '组) ' if dataset_group_name else ''}计算基本指标，有效样本数: {len(container)}")
            return metrics
                
        except Exception as e:
            self.logger.warning(f"警告: {model_name} {'(' + dataset_group_name + '组) ' if dataset_group_name else ''}计算基本指标时出错: {str(e)}")
            return None

    def save_all_metrics(self) -> None:
        """
        Save all metrics to a single JSON file
        """
        all_metrics = self.metrics_store.get()
        self.reporter.merge_and_save_metrics(all_metrics)

        self.logger.info(f"保存的指标分组数量: {len(all_metrics)}")
        for key, value in all_metrics.items():
            self.logger.info(f"  分组 {key} 中的模型数量: {len(value)}")
            for model_name, metrics in value.items():
                metric_types = list(metrics.keys())
                self.logger.info(f"  模型 {model_name} 的指标类型: {', '.join(metric_types)}")

    def run(self) -> None:    
        
        # 读取和准备数据
        self.setup()
        
        # 保存合并数据
        self.save_merged_data()
        
        # 执行评估
        self.run_evaluation()
        
        # 保存所有指标
        self.save_all_metrics()
        
        self.logger.info("评估完成！所有结果已保存到" + self.output_dir)
