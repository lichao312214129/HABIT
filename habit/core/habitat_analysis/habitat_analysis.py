"""
Habitat Clustering Analysis Module
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pickle
import warnings
import multiprocessing
from glob import glob
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from habit.utils.io_utils import (get_image_and_mask_paths,
                            detect_image_names,
                            check_data_structure,
                            save_habitat_image)

from .features.feature_expression_parser import FeatureExpressionParser
from .features.feature_extractor_factory import create_feature_extractor
from habit.utils.visualization import plot_cluster_scores
from habit.core.habitat_analysis.clustering.base_clustering import get_clustering_algorithm
from habit.core.habitat_analysis.clustering.cluster_validation_methods import (
    get_validation_methods,
    is_valid_method_for_algorithm,
    get_default_methods
)
from habit.core.habitat_analysis.features.feature_preprocessing import preprocess_features

# Ignore warnings
warnings.simplefilter('ignore')

# Import progress utilities
from habit.utils.progress_utils import CustomTqdm


class HabitatAnalysis:
    """
    Habitat Analysis class for performing two-step clustering analysis:
    1. Individual-level clustering: Divide each tumor into supervoxels
    2. Population-level clustering: Cluster the average values of all sample supervoxels to obtain habitats
    """

    def __init__(self,
                 root_folder: str,
                 out_folder: Optional[str] = None,
                 feature_config: Optional[Dict[str, Any]] = None,
                 clustering_mode: str = "two_step",
                 supervoxel_clustering_method: str = "kmeans",
                 habitat_clustering_method: str = "kmeans",
                 mode: str = "training",
                 n_clusters_supervoxel: int = 50,
                 n_clusters_habitats_max: int = 10,
                 n_clusters_habitats_min: int = 2,
                 habitat_cluster_selection_method: Optional[Union[str, List[str]]] = None,
                 best_n_clusters: Optional[int] = None,
                 one_step_settings: Optional[Dict[str, Any]] = None,
                 n_processes: int = 1,
                 random_state: int = 42,
                 verbose: bool = True,
                 images_dir: str = "images",
                 masks_dir: str = "masks",
                 plot_curves: bool = True,
                 progress_callback: Optional[Callable] = None,
                 save_intermediate_results: bool = False,
                 config_file: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize HabitatAnalysis class

        Args:
            root_folder (str): Root directory of data
            out_folder (str, optional): Output directory
            feature_config (dict, optional): Complete feature configuration dictionary that includes:
                - image_names (list): List of image names, e.g. ['pre_contrast', 'LAP', 'PVP']
                - extractor (str or dict): Feature extractor name or configuration
                - params (dict): Feature extractor parameters, used when extractor is a string
                - preprocessing (dict): Feature preprocessing parameters
            clustering_mode (str, optional): Clustering strategy, either 'one_step' or 'two_step', default is two_step
                - one_step: Individual-level clustering only (each tumor gets its own habitats)
                - two_step: Individual + Population clustering (supervoxels then habitats across patients)
            supervoxel_clustering_method (str, optional): Supervoxel clustering method, default is kmeans
            habitat_clustering_method (str, optional): Habitat clustering method, default is kmeans
            mode (str, optional): Analysis mode, either 'training' or 'testing', default is training
            n_clusters_supervoxel (int, optional): Number of supervoxel clusters for two_step, default is 50
            n_clusters_habitats_max (int, optional): Maximum number of habitat clusters, default is 10
            n_clusters_habitats_min (int, optional): Minimum number of habitat clusters, default is 2
            habitat_cluster_selection_method (str, optional): Method for selecting number of habitat clusters
            best_n_clusters (int, optional): Directly specify the best number of clusters
            one_step_settings (dict, optional): Settings for one_step mode cluster selection:
                - min_clusters: Minimum number of clusters to test
                - max_clusters: Maximum number of clusters to test
                - selection_method: Method to determine optimal clusters
                - plot_validation_curves: Whether to plot validation curves for each tumor
            n_processes (int, optional): Number of parallel processes, default is 1
            random_state (int, optional): Random seed, default is 42
            verbose (bool, optional): Whether to output detailed information, default is True
            images_dir (str, optional): Image directory name, default is images
            masks_dir (str, optional): Mask directory name, default is masks
            plot_curves (bool, optional): Whether to plot evaluation curves, default is True
            progress_callback (callable, optional): Progress callback function
            save_intermediate_results (bool, optional): Whether to save intermediate results, default is False
            config_file (str, optional): Path to the original configuration file
        """
        # Basic parameters
        self.data_dir = os.path.abspath(root_folder)
        self.out_dir = os.path.abspath(out_folder or os.path.join(self.data_dir, "habitats_output"))
        self.config_file = config_file  # 保存原始配置文件路径
        self._setup_logging(log_level)

        # Clustering mode: one_step or two_step
        self.clustering_mode = clustering_mode.lower()
        if self.clustering_mode not in ['one_step', 'two_step']:
            raise ValueError(f"clustering_mode must be 'one_step' or 'two_step', got '{clustering_mode}'")
        
        # Clustering parameters
        self.supervoxel_method = supervoxel_clustering_method
        self.habitat_method = habitat_clustering_method
        self.mode = mode
        self.n_clusters_supervoxel = n_clusters_supervoxel
        self.n_clusters_habitats_max = n_clusters_habitats_max
        self.n_clusters_habitats_min = n_clusters_habitats_min

        # One-step mode settings
        if self.clustering_mode == 'one_step':
            default_one_step_settings = {
                'min_clusters': 2,
                'max_clusters': 10,
                'selection_method': 'silhouette',
                'plot_validation_curves': True
            }
            if one_step_settings is None:
                self.one_step_settings = default_one_step_settings
            else:
                self.one_step_settings = {**default_one_step_settings, **one_step_settings}
            
            if self.verbose:
                self.logger.info(f"One-step clustering mode enabled")
                self.logger.info(f"  Cluster range: [{self.one_step_settings['min_clusters']}, {self.one_step_settings['max_clusters']}]")
                self.logger.info(f"  Selection method: {self.one_step_settings['selection_method']}")
        else:
            self.one_step_settings = None

        self.verbose = verbose

        # Get validation methods supported by the clustering method using cluster_validation_methods module
        validation_info = get_validation_methods(habitat_clustering_method)
        valid_selection_methods = list(validation_info['methods'].keys())
        default_methods = get_default_methods(habitat_clustering_method)

        if self.verbose:
            self.logger.info(f"Validation methods supported by clustering method '{habitat_clustering_method}': {', '.join(valid_selection_methods)}")
            self.logger.info(f"Default validation methods: {', '.join(default_methods)}")

        # Check validity of habitat_cluster_selection_method parameter
        if habitat_cluster_selection_method is None:
            # Use default methods
            self.habitat_cluster_selection_method = default_methods
            if self.verbose:
                self.logger.info(f"No clustering evaluation method specified, using default methods: {', '.join(default_methods)}")
        elif isinstance(habitat_cluster_selection_method, str):
            # Check if single method is valid
            if is_valid_method_for_algorithm(habitat_clustering_method, habitat_cluster_selection_method.lower()):
                self.habitat_cluster_selection_method = habitat_cluster_selection_method.lower()
            else:
                # If specified method is invalid for current clustering algorithm, use default methods
                original_method = habitat_cluster_selection_method
                self.habitat_cluster_selection_method = default_methods
                if self.verbose:
                    self.logger.warning(f"Validation method '{original_method}' is invalid for clustering method '{habitat_clustering_method}'")
                    self.logger.info(f"Available validation methods: {', '.join(valid_selection_methods)}")
                    self.logger.info(f"Using default methods: {', '.join(default_methods)}")
        elif isinstance(habitat_cluster_selection_method, list):
            # Check if multiple methods are valid
            valid_methods = []
            invalid_methods = []

            for method in habitat_cluster_selection_method:
                if is_valid_method_for_algorithm(habitat_clustering_method, method.lower()):
                    valid_methods.append(method.lower())
                else:
                    invalid_methods.append(method)

            if valid_methods:
                self.habitat_cluster_selection_method = valid_methods
                if invalid_methods and self.verbose:
                    self.logger.warning(f"The following validation methods are invalid for clustering method '{habitat_clustering_method}': {', '.join(invalid_methods)}")
                    self.logger.info(f"Only using valid methods: {', '.join(valid_methods)}")
            else:
                # If no valid methods, use default methods
                self.habitat_cluster_selection_method = default_methods
                if self.verbose:
                    self.logger.warning(f"All specified validation methods are invalid for clustering method '{habitat_clustering_method}'")
                    self.logger.info(f"Available validation methods: {', '.join(valid_selection_methods)}")
                    self.logger.info(f"Using default methods: {', '.join(default_methods)}")
        else:
            # Parameter type error, use default methods
            self.habitat_cluster_selection_method = default_methods
            if self.verbose:
                self.logger.warning("Clustering evaluation method parameter type error, should be string or list")
                self.logger.info(f"Using default methods: {', '.join(default_methods)}")


        # Parameters
        self.best_n_clusters = best_n_clusters
        self.random_state = random_state
        self.verbose = verbose
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.plot_curves = plot_curves
        self.n_processes = n_processes
        self.progress_callback = progress_callback
        self.save_intermediate_results = save_intermediate_results

        # Initialize feature configuration dictionary
        self.feature_config = feature_config or {}

        # 检查feature_config格式
        if 'voxel_level' not in self.feature_config:
            raise ValueError("voxel_level configuration is required")

        # 检查supervoxel_level配置（可选）
        if 'supervoxel_level' in self.feature_config and self.verbose:
            self.logger.info("Note: supervoxel_level feature configuration is detected but not used in the current implementation.")

        # Create output directory
        os.makedirs(self.out_dir, exist_ok=True)

        # Image paths
        self.images_paths, self.mask_paths = get_image_and_mask_paths(
            self.data_dir, keyword_of_raw_folder=self.images_dir,
            keyword_of_mask_folder=self.masks_dir
        )

        # 检查voxel_level中是否有image_names
        voxel_config = self.feature_config['voxel_level']
        if isinstance(voxel_config, dict) and 'image_names' not in voxel_config:
            if self.verbose:
                self.logger.info("Image names not provided in voxel_level config, automatically detecting from data directory...")
            image_names = detect_image_names(self.images_paths)
            self.feature_config['voxel_level']['image_names'] = image_names
            if self.verbose:
                self.logger.info(f"Detected image names: {image_names}")

        # Initialize feature extractor
        self._init_feature_extractor()

        # Initialize clustering algorithms
        self.voxel2supervoxel_clustering = get_clustering_algorithm(
            self.supervoxel_method,
            n_clusters=self.n_clusters_supervoxel,
            random_state=self.random_state
        )

        self.supervoxel2habitat_clustering = get_clustering_algorithm(
            self.habitat_method,
            n_clusters=self.n_clusters_habitats_max,
            random_state=self.random_state
        )

        # Results storage
        self.X = None  # Feature matrix
        self.supervoxel_labels = {}  # Supervoxel labels
        self.habitat_labels = None  # Habitat labels
        self.results_df = None  # Results DataFrame

        # Validate data structure
        voxel_config = self.feature_config['voxel_level']
        if isinstance(voxel_config, dict) and 'image_names' in voxel_config:
            check_data_structure(self.images_paths, self.mask_paths,
                               voxel_config['image_names'], None)
        else:
            raise ValueError("voxel_level configuration must contain 'image_names' field")
    
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration.
        
        Args:
            log_level (str): Logging level.
        """
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler
        log_dir = os.path.join(self.out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "habitat_analysis.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def _init_feature_extractor(self):
        """Initialize feature extractor based on configuration"""


        # 创建表达式解析器实例
        self.expression_parser = FeatureExpressionParser()

        # 提取voxel_level配置
        voxel_config = self.feature_config['voxel_level']

        # 确保voxel_config是字典且包含method字段
        if not isinstance(voxel_config, dict) or 'method' not in voxel_config:
            raise ValueError("voxel_level must be a dictionary with 'method' field")

        # 解析voxel_level表达式
        self.voxel_method, self.voxel_params, self.voxel_processing_steps = self.expression_parser.parse(voxel_config)

        # 检查是否提供了supervoxel_level配置
        self.has_supervoxel_config = 'supervoxel_level' in self.feature_config
        if self.has_supervoxel_config:
            # 提取supervoxel_level配置
            supervoxel_config = self.feature_config['supervoxel_level']

            # 确保supervoxel_config是字典且包含method字段
            if not isinstance(supervoxel_config, dict) or 'method' not in supervoxel_config:
                raise ValueError("supervoxel_level must be a dictionary with 'method' field")

            # 解析supervoxel_level表达式
            self.supervoxel_method_name, self.supervoxel_params, self.supervoxel_processing_steps = self.expression_parser.parse(supervoxel_config)

            if self.verbose:
                self.logger.info("supervoxel_level feature configuration detected but not used in current implementation")

        # 创建跨图像特征提取器参数
        # 准备跨图像参数
        cross_image_kwargs = {}

        # 检查voxel_params是否为空
        if self.voxel_params:
            # 处理跨图像参数
            for param_name, param_value in self.voxel_params.items():
                # 从voxel_level.params中获取参数值
                voxel_params = self.feature_config['voxel_level'].get('params', {})
                if param_value == param_name and param_name in voxel_params:
                    cross_image_kwargs[param_name] = voxel_params[param_name]
                elif isinstance(param_value, str) and param_value in voxel_params:
                    cross_image_kwargs[param_name] = voxel_params[param_value]
                else:
                    # 否则直接使用参数值
                    cross_image_kwargs[param_name] = param_value
                    
        self.cross_image_kwargs = cross_image_kwargs

    def _determine_optimal_clusters_for_subject(self, subject: str, features: np.ndarray, settings: Dict[str, Any]) -> int:
        """
        Determine optimal number of clusters for a single subject using validation methods
        
        Args:
            subject (str): Subject ID
            features (np.ndarray): Feature matrix for the subject
            settings (dict): One-step settings containing min_clusters, max_clusters, selection_method, etc.
            
        Returns:
            int: Optimal number of clusters
        """
        min_k = settings['min_clusters']
        max_k = settings['max_clusters']
        selection_method = settings['selection_method']
        plot_curves = settings.get('plot_validation_curves', False)
        
        self.logger.info(f"Determining optimal clusters for {subject} using {selection_method}")
        
        # Test different numbers of clusters
        cluster_range = range(min_k, max_k + 1)
        scores = []
        
        from .clustering.base_clustering import get_clustering_algorithm
        from .clustering.cluster_validation_methods import get_validation_methods
        
        for n_clusters in cluster_range:
            try:
                # Create clustering algorithm
                clusterer = get_clustering_algorithm(
                    self.supervoxel_method,
                    n_clusters=n_clusters,
                    random_state=self.random_state
                )
                
                # Fit and predict
                clusterer.fit(features)
                labels = clusterer.predict(features)
                
                # Calculate validation score
                validation_info = get_validation_methods(self.supervoxel_method)
                validation_func = validation_info['methods'][selection_method]['function']
                
                if selection_method in ['aic', 'bic']:
                    # For model-based methods that need the model
                    score = validation_func(clusterer.model, features, labels)
                else:
                    # For distance/density-based methods
                    score = validation_func(features, labels)
                
                scores.append(score)
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate {n_clusters} clusters for {subject}: {e}")
                scores.append(np.nan)
        
        # Determine optimal number of clusters based on selection method
        scores = np.array(scores)
        valid_indices = ~np.isnan(scores)
        
        if not np.any(valid_indices):
            self.logger.warning(f"All validation scores failed for {subject}, using default min_clusters={min_k}")
            return min_k
        
        # Different methods have different optimization directions
        validation_info = get_validation_methods(self.supervoxel_method)
        higher_is_better = validation_info['methods'][selection_method]['higher_is_better']
        
        if higher_is_better:
            # For silhouette, calinski_harabasz: higher is better
            optimal_idx = np.nanargmax(scores)
        else:
            # For davies_bouldin, inertia, aic, bic: lower is better
            optimal_idx = np.nanargmin(scores)
        
        optimal_n_clusters = cluster_range[optimal_idx]
        
        # Plot validation curves if requested
        if plot_curves:
            try:
                subject_plot_dir = os.path.join(self.out_dir, f"{subject}_validation_plots")
                os.makedirs(subject_plot_dir, exist_ok=True)
                
                plot_data = {selection_method: (list(cluster_range), scores.tolist())}
                plot_file = os.path.join(subject_plot_dir, f"{subject}_cluster_validation.png")
                
                plot_cluster_scores(
                    plot_data,
                    save_path=plot_file,
                    title=f"{subject} - Cluster Validation ({selection_method})"
                )
                
                self.logger.info(f"Validation plot saved to: {plot_file}")
            except Exception as e:
                self.logger.warning(f"Failed to plot validation curves for {subject}: {e}")
        
        return optimal_n_clusters
    
    def _voxel2supervoxel_clustering(self, subject: str) -> Tuple[str, Union[Tuple[str, pd.DataFrame], Exception]]:
        """
        Process a single subject

        Args:
            subject (str): Subject ID to process

        Returns:
            Tuple[str, Union[Tuple[str, pd.DataFrame], Exception]]: Tuple containing:
                - subject ID
                - Either a tuple of (subject ID, mean_features_df) or an Exception
        """
        try:
            # log add subject
            self.logger.info(f"_voxel2supervoxel_clustering subject: {subject}")
            print(f"_voxel2supervoxel_clustering subject: {subject}")
            # Extract features
            _, feature_df, raw_df, mask_info = self.extract_voxel_features(subject)
            
            # Apply preprocessing if enabled
            if self.feature_config.get('preprocessing_for_subject_level', False):
                # 检查是否使用新的多方法串联机制
                if 'methods' in self.feature_config['preprocessing_for_subject_level']:
                    X_subj_ = preprocess_features(
                        feature_df.values,
                        methods=self.feature_config['preprocessing_for_subject_level']['methods']
                    )
                    feature_df = pd.DataFrame(X_subj_, columns=feature_df.columns)

            # Perform clustering for voxel to supervoxel
            try:
                if self.clustering_mode == 'one_step':
                    # One-step mode: determine optimal number of clusters for this individual tumor
                    optimal_n_clusters = self._determine_optimal_clusters_for_subject(
                        subject, 
                        feature_df.values,
                        self.one_step_settings
                    )
                    self.logger.info(f"Subject {subject}: optimal clusters = {optimal_n_clusters}")
                    print(f"Subject {subject}: optimal clusters = {optimal_n_clusters}")
                    
                    # Update clustering algorithm with optimal number of clusters
                    from .clustering.base_clustering import get_clustering_algorithm
                    self.voxel2supervoxel_clustering = get_clustering_algorithm(
                        self.supervoxel_method,
                        n_clusters=optimal_n_clusters,
                        random_state=self.random_state
                    )
                
                self.voxel2supervoxel_clustering.fit(feature_df.values)
                supervoxel_labels = self.voxel2supervoxel_clustering.predict(feature_df.values)
                supervoxel_labels += 1  # Start numbering from 1
            except Exception as e:
                self.logger.error(f"Error performing supervoxel clustering for subject {subject}: {str(e)}")
                raise e

            # Get feature names
            feature_names = feature_df.columns.tolist()

            # Get original feature names (image names)
            original_feature_names = raw_df.columns.tolist()

            # Calculate average features for each supervoxel
            unique_labels = np.arange(1, self.n_clusters_supervoxel + 1)
            data_rows = []
            for i in unique_labels:
                indices = supervoxel_labels == i
                if np.any(indices):
                    # Calculate average features for current supervoxel
                    mean_features = np.mean(feature_df[indices], axis=0)
                    mean_original_features = np.mean(raw_df.values[indices], axis=0)
                    count = np.sum(indices)
                    # Create data row
                    data_row = {
                        "Subject": subject,
                        "Supervoxel": i,
                        "Count": count,
                    }
                    # Add feature means
                    for j, name in enumerate(feature_names):
                        data_row[name] = mean_features[j]

                    # Add original feature means
                    for j, name in enumerate(original_feature_names):
                        data_row[f"{name}-original"] = mean_original_features[j]

                    data_rows.append(data_row)
            mean_features_df = pd.DataFrame(data_rows)

            # Save supervoxel image
            if isinstance(mask_info, dict) and 'mask_array' in mask_info and 'mask' in mask_info:
                # Create supervoxel map
                supervoxel_map = np.zeros_like(mask_info['mask_array'])
                mask_indices = mask_info['mask_array'] > 0
                supervoxel_map[mask_indices] = supervoxel_labels

                # Convert to SimpleITK image and save
                supervoxel_img = sitk.GetImageFromArray(supervoxel_map)
                supervoxel_img.CopyInformation(mask_info['mask'])

                # Save as nrrd file
                sitk.WriteImage(supervoxel_img, os.path.join(self.out_dir, f"{subject}_supervoxel.nrrd"))

                # Clean up memory
                del supervoxel_map, mask_indices

            # Clean up memory
            del feature_df, raw_df, supervoxel_labels

            return subject, mean_features_df
        except Exception as e:
            # 记录错误并返回异常对象
            self.logger.error(f"Error in _voxel2supervoxel_clustering for subject {subject}: {str(e)}")
            return subject, Exception(str(e))

    def extract_voxel_features(self, subject: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
        """
        Extract features for a single subject

        Args:
            subject (str): Subject ID

        Returns:
            Tuple[str, pd.DataFrame, pd.DataFrame, np.ndarray, dict]: Tuple containing:
                - subject ID
                - feature dataframe
                - original image dataframe
                - mask array
                - dictionary with image information
        """
        
        # Get image and mask paths
        img_paths = self.images_paths[subject]
        mask_paths = self.mask_paths[subject]

        # 先处理单图像
        processed_images = []
        # 遍历处理步骤，处理每个单图像
        for step in self.voxel_processing_steps:
            method = step['method']
            img_name = step['image']
            step_params = step['params'].copy()  # 复制参数，避免修改原始参数

            # 检查step_params是否为空
            if step_params:
                # 处理参数，将参数名替换为实际值
                for param_name, param_value in list(step_params.items()):
                    # 检查voxel_level.params中的参数
                    voxel_params = self.feature_config['voxel_level'].get('params', {})
                    if param_value == param_name and param_name in voxel_params:
                        step_params[param_name] = voxel_params[param_name]
                    elif isinstance(param_value, str) and param_value in voxel_params:
                        step_params[param_name] = voxel_params[param_value]

            # 按需创建单图像特征提取器并处理图像
            step_params.update({'subject': subject, 'image': img_name})
            extractor = create_feature_extractor(method, **step_params)
            processed_df = extractor.extract_features(img_paths.get(img_name), mask_paths.get(img_name), **step_params)
            processed_images.append(processed_df)

        # 按需创建跨图像特征提取器
        cross_image_kwargs = self.cross_image_kwargs.copy()
        cross_image_kwargs.update({'subject': subject})
        cross_image_extractor = create_feature_extractor(self.voxel_method, **cross_image_kwargs)
        features = cross_image_extractor.extract_features(processed_images, **cross_image_kwargs)

        # get raw data - 在连接前重命名列
        raw_df = pd.concat(processed_images, axis=1)

        # Save mask information for later image reconstruction
        mask = self.mask_paths[subject]
        mask = list(mask.values())[0]
        mask_img = sitk.ReadImage(mask)
        mask_array = sitk.GetArrayFromImage(mask_img)
        mask_info = {
            'mask': mask_img,
            'mask_array': mask_array
        }

        # Clean up memory
        del processed_images, mask_img, mask_array

        return subject, features, raw_df, mask_info

    def extract_supervoxel_features(self, subject: str) -> Tuple[str, np.ndarray]:
        """
        Extract supervoxel-level features from supervoxel maps and original images

        Returns:
            np.ndarray: Supervoxel features for clustering
        """
        try:
            # log and print
            self.logger.info(f"Extracting supervoxel-level features for subject {subject}...")
            print(f"Extracting supervoxel-level features for subject {subject}...")

            # Get image and mask paths
            img_paths = self.images_paths[subject]
            mask_path = self.supervoxel_file_dict[subject]

            # 获取outdir中的所有supervoxel文件
            if self.verbose:
                self.logger.info("Extracting supervoxel-level features...")

            # 先处理单图像
            processed_images = []
            # 遍历处理步骤，处理每个单图像
            for step in self.supervoxel_processing_steps:
                method = step['method']
                img_name = step['image']
                step_params = step['params'].copy()  # 复制参数，避免修改原始参数

                # 使用单图像特征提取器处理图像
                step_params.update({'subject': subject, 'image': img_name})
                # 按需创建单图像特征提取器
                single_image_extractor = create_feature_extractor(method, **step_params)
                processed_df = single_image_extractor.extract_features(img_paths.get(img_name), mask_path, **step_params)
                processed_images.append(processed_df)

            # 按需创建跨图像特征提取器
            supervoxel_params = self.supervoxel_params.copy()
            cross_image_extractor = create_feature_extractor(self.supervoxel_method_name, **supervoxel_params)
            features = cross_image_extractor.extract_features(processed_images, **supervoxel_params)

            return subject, features
        
        except Exception as e:
            return subject, Exception(str(e))

    def _save_habitat_for_subject(self, subject):
        """
        保存单个主体的栖息地图像

        Args:
            args (tuple): 包含主体索引和名称的元组 (i, subject)

        Returns:
            int: 主体索引
        """
        try:
            supervoxel_path = os.path.join(self.out_dir, f"{subject}_supervoxel.nrrd")
            save_habitat_image(subject, self.results_df, supervoxel_path, self.out_dir)
            return subject, None
        
        except Exception as e:
            return subject, Exception(str(e))

    def run(self, subjects: Optional[List[str]] = None, save_results_csv: bool = True) -> pd.DataFrame:
        """
        Run the habitat clustering pipeline

        Args:
            subjects (Optional[List[str]], optional): List of subjects to process. If None, all subjects will be processed.
            save_results_csv (bool, optional): Whether to save results as CSV files. Defaults to True.

        Returns:
            pd.DataFrame: Habitat clustering results
        """
        if subjects is None:
            subjects = list(self.images_paths.keys())

        # Extract features and perform supervoxel clustering for each subject
        if self.verbose:
            self.logger.info("Extracting features and performing supervoxel clustering...")

        # Results storage
        mean_features_all = pd.DataFrame()
        failed_subjects = []  # Track failed subjects

        # Use multiprocessing to process subjects
        if self.n_processes > 1 and len(subjects) > 1:
            if self.verbose:
                self.logger.info(f"Using {self.n_processes} processes for parallel processing...")

            with multiprocessing.Pool(processes=self.n_processes) as pool:
                if self.verbose:
                    self.logger.info("Starting parallel processing of all subjects...")
                results_iter = pool.imap_unordered(self._voxel2supervoxel_clustering, subjects)

                # 使用自定义进度条显示进度
                progress_bar = CustomTqdm(total=len(subjects), desc="Processing subjects")
                for result in results_iter:
                    if isinstance(result[1], Exception):
                        # 如果结果是异常，记录错误并继续
                        failed_subjects.append(result[0])
                        if self.verbose:
                            self.logger.error(f"Error processing subject {result[0]}: {str(result[1])}")
                    else:
                        subject, mean_features_df = result
                        mean_features_all = pd.concat([mean_features_all, mean_features_df], ignore_index=True)

                    progress_bar.update(1)

                if self.verbose:
                    if failed_subjects:
                        self.logger.warning(f"Failed to process {len(failed_subjects)}")
                    self.logger.info(f"\nAll {len(subjects)} subjects have been processed. Proceeding to clustering...")
        
        # Single process processing, use custom progress bar
        else:
            progress_bar = CustomTqdm(total=len(subjects), desc="Processing subjects")
            results = []
            for subject in subjects:
                result = self._voxel2supervoxel_clustering(subject)
                results.append(result)
            for subject in subjects:
                if isinstance(result[1], Exception):
                    failed_subjects.append(subject)
                    if self.verbose:
                        self.logger.error(f"Error processing subject {subject}: {str(result[1])}")
                else:
                    subject, mean_features_df = result
                    mean_features_all = pd.concat([mean_features_all, mean_features_df], ignore_index=True)
                
                progress_bar.update(1)

            if self.verbose and failed_subjects:
                self.logger.warning(f"Failed to process {len(failed_subjects)}")

        # Check if there is enough data for clustering
        if len(mean_features_all) == 0:
            raise ValueError("No valid features for analysis")

        # ===============================================
        # Prepare features for population-level clustering
        # feature_names = mean_features_all中不以-original结尾的列
        feature_names = [col for col in mean_features_all.columns if not col.endswith('-original')]
        feature_names = feature_names[3:]  # 去掉Subject, Supervoxel, Count FIXME: 需要根据实际情况调整
        features_of_all_subjects = mean_features_all[feature_names]
        supervoxel_file_keyword = self.feature_config['supervoxel_level'].get('supervoxel_file_keyword', '*_supervoxel.nrrd')
        supervoxel_files = glob(os.path.join(self.out_dir, supervoxel_file_keyword))
        # if subject in subjects, then add to supervoxel_file_keyword_dict
        self.supervoxel_file_dict = {}
        for subject in subjects:
            for supervoxel_file in supervoxel_files:
                if subject in supervoxel_file:
                    self.supervoxel_file_dict[subject] = supervoxel_file
                    break
            else:
                if subject not in failed_subjects:  # Only warn for non-failed subjects
                    if self.verbose:
                        self.logger.warning(f"No supervoxel file found for subject {subject}")

        if not self.supervoxel_file_dict:
            raise ValueError(f"No supervoxel files found in {self.out_dir}")

        # 检查是否使用mean_voxel_features()
        is_mean_voxel_features = 'mean_voxel_features' in self.feature_config['supervoxel_level']['method']
        if not is_mean_voxel_features:
            # Use multiprocessing to extract supervoxel features for all subjects
            if self.n_processes > 1 and len(subjects) > 1:
                if self.verbose:
                    self.logger.info(f"Using {self.n_processes} processes for supervoxel feature extraction...")

                with multiprocessing.Pool(processes=self.n_processes) as pool:
                    if self.verbose:
                        self.logger.info("Starting parallel supervoxel feature extraction...")

                    # Use imap_unordered to process subjects in parallel
                    results_iter = pool.imap_unordered(self.extract_supervoxel_features, subjects)

                    # Collect features from all subjects
                    features_of_all_subjects = []
                    progress_bar = CustomTqdm(total=len(subjects), desc="Extracting supervoxel features")
                    for result in results_iter:
                        if isinstance(result[1], Exception):
                            failed_subjects.append(result[0])
                            if self.verbose:
                                self.logger.error(f"Error extracting supervoxel features for subject {result[0]}: {str(result[1])}")
                        else:
                            features_of_all_subjects.append(result[1])
                        progress_bar.update(1)

                    if self.verbose and failed_subjects:
                        self.logger.warning(f"Failed to extract supervoxel features for {len(failed_subjects)}")

                    # concat
                    if features_of_all_subjects:
                        features_of_all_subjects = pd.concat(features_of_all_subjects, ignore_index=True)
                    else:
                        raise ValueError("No valid supervoxel features extracted")
            else:
                # Single process processing
                features_of_all_subjects = []
                progress_bar = CustomTqdm(total=len(subjects), desc="Extracting supervoxel features")
                for subject in subjects:
                    result = self.extract_supervoxel_features(subject)
                    if isinstance(result[1], Exception):
                        failed_subjects.append(result[0])
                        if self.verbose:
                            self.logger.error(f"Error extracting supervoxel features for subject {result[0]}: {str(result[1])}")
                    else:
                        features_of_all_subjects.append(result[1])
                    progress_bar.update(1)

                if self.verbose and failed_subjects:
                    self.logger.warning(f"Failed to extract supervoxel features for {len(failed_subjects)}")

                # concat
                if features_of_all_subjects:
                    features_of_all_subjects = pd.concat(features_of_all_subjects, ignore_index=True)
                else:
                    raise ValueError("No valid supervoxel features extracted")

        # ===============================================
        #  TODO: 把mean fill也整合到preprocess_features中
        #  TODO: 且preprocess_features要返回model，方便测试集使用
        features_of_all_subjects = features_of_all_subjects.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        features_of_all_subjects = features_of_all_subjects.applymap(lambda x: x.item() if hasattr(x, 'item') else x)
        features_of_all_subjects = features_of_all_subjects.replace([np.inf, -np.inf], np.nan)
        features_of_all_subjects = features_of_all_subjects.fillna(features_of_all_subjects.mean())

        # Apply preprocessing if enabled
        if self.feature_config.get('preprocessing_for_group_level', False):
            # 检查是否使用新的多方法串联机制
            if 'methods' in self.feature_config['preprocessing_for_group_level']:
                # 兼容多方法串联机制
                farray = preprocess_features(
                    features_of_all_subjects.values,
                    methods=self.feature_config['preprocessing_for_group_level']['methods']
                )
                features_of_all_subjects = pd.DataFrame(farray, columns=features_of_all_subjects.columns)

        #  Save mean values for unseen test subjects if is training model  FIXME: 需要根据实际情况调整
        if self.mode == 'training':
            mean_values = features_of_all_subjects.mean()
            mean_values.name = 'mean_values'
            mean_values.to_csv(os.path.join(self.out_dir, 'mean_values_of_all_supervoxels_features.csv'), index=True, header=True)
        elif self.mode == 'testing':
            #  Load mean values for unseen test subjects
            mean_values = pd.read_csv(os.path.join(self.out_dir, 'mean_values_of_all_supervoxels_features.csv'))
            features_of_all_subjects = features_of_all_subjects.fillna(mean_values)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        # ===============================================
        
        # For one_step mode, skip population-level clustering
        if self.clustering_mode == 'one_step':
            if self.verbose:
                self.logger.info("One-step clustering mode: skipping population-level clustering")
                self.logger.info("Using individual-level clusters as final habitats")
            
            # In one-step mode, supervoxels ARE the habitats
            # Rename Supervoxel column to Habitats
            mean_features_all['Habitats'] = mean_features_all['Supervoxel']
            self.results_df = mean_features_all.copy()
            
            # Save results
            if save_results_csv:
                if self.verbose:
                    self.logger.info("Saving one-step clustering results...")
                
                os.makedirs(self.out_dir, exist_ok=True)
                
                # Save main results
                results_path = os.path.join(self.out_dir, 'results_all_samples.csv')
                self.results_df.to_csv(results_path, index=False)
                if self.verbose:
                    self.logger.info(f"Results saved to {results_path}")
                
                # Save summary statistics
                summary = self.results_df.groupby('Subject')['Habitats'].agg(['nunique', 'count'])
                summary.columns = ['Num_Habitats', 'Total_Voxels']
                summary_path = os.path.join(self.out_dir, 'clustering_summary.csv')
                summary.to_csv(summary_path)
                if self.verbose:
                    self.logger.info(f"Summary saved to {summary_path}")
            
            # In one-step mode, supervoxel images already represent final habitats
            # No need to generate separate habitat images
            if self.verbose:
                self.logger.info("Habitat images (supervoxel maps) already saved during Step 1")
            
            return self.results_df
        
        # For two_step mode, continue with population-level clustering
        if self.mode == 'training':
            # Determine optimal number of clusters
            if self.best_n_clusters is not None:
                # If best number of clusters is already specified, use it directly
                optimal_n_clusters = self.best_n_clusters
                scores = None
                if self.verbose:
                    self.logger.info(f"Using specified best number of clusters: {optimal_n_clusters}")
            else:
                # Otherwise find optimal number of clusters
                if self.verbose:
                    self.logger.info("Finding optimal number of clusters...")

                try:
                    # Ensure cluster number range is reasonable
                    min_clusters = max(2, self.n_clusters_habitats_min)
                    max_clusters = min(self.n_clusters_habitats_max, len(features_of_all_subjects) - 1)

                    if max_clusters <= min_clusters:
                        # If range is invalid, use default minimum value
                        if self.verbose:
                            self.logger.warning(f"Invalid cluster number range [{min_clusters}, {max_clusters}], using default value")
                        optimal_n_clusters = min_clusters
                        scores = None
                    else:
                        # Try to find optimal number of clusters
                        try:
                            cluster_for_best_n = get_clustering_algorithm(self.habitat_method)
                            optimal_n_clusters, scores = cluster_for_best_n.find_optimal_clusters(
                                features_of_all_subjects,
                                min_clusters=min_clusters,
                                max_clusters=max_clusters,
                                methods=self.habitat_cluster_selection_method,
                                show_progress=True
                            )

                            # If plotting is needed, plot score graphs
                            if self.plot_curves and scores is not None:
                                try:
                                    # 确保输出目录存在
                                    os.makedirs(self.out_dir, exist_ok=True)

                                    # 构造保存路径
                                    save_path = os.path.join(self.out_dir, 'habitat_clustering_scores.png')

                                    # 使用新的可视化函数绘图
                                    plot_cluster_scores(
                                        scores_dict=scores,
                                        cluster_range=cluster_for_best_n.cluster_range,
                                        methods=self.habitat_cluster_selection_method,
                                        clustering_algorithm=self.habitat_method,
                                        figsize=(12, 8),
                                        save_path=save_path,
                                        show=False
                                    )

                                    if self.verbose:
                                        self.logger.info(f"聚类评分图已保存至 {save_path}")
                                except Exception as e:
                                    # 捕获绘图过程中的错误
                                    if self.verbose:
                                        self.logger.error(f"绘制聚类评分图时出错: {str(e)}")
                                        self.logger.info("继续执行其他流程...")
                        except Exception as e:
                            # If optimal number of clusters cannot be found, use default value and warn user
                            if self.verbose:
                                self.logger.warning(f"Failed to find optimal number of clusters: {str(e)}")
                                self.logger.info("Using default number of clusters")
                            optimal_n_clusters = min(3, max_clusters)  # Use a reasonable default value
                            scores = None
                except Exception as e:
                    # Catch all exceptions, use default value
                    if self.verbose:
                        self.logger.error(f"Exception occurred when determining optimal number of clusters: {str(e)}")
                        self.logger.info("Using default number of clusters")
                    optimal_n_clusters = 3  # Default to 3 clusters
                    scores = None

                if self.verbose:
                    self.logger.info(f"Optimal number of clusters: {optimal_n_clusters}")

                # Perform population-level clustering using optimal number of clusters
                if self.verbose:
                    self.logger.info("Performing population-level clustering...")

            # Reinitialize clustering algorithm with optimal number of clusters
            self.supervoxel2habitat_clustering.n_clusters = optimal_n_clusters
            self.supervoxel2habitat_clustering.fit(features_of_all_subjects)
            habitat_labels = self.supervoxel2habitat_clustering.predict(features_of_all_subjects) + 1  # Start numbering from 1

            # Save clustering model of supervoxel to habitat
            model_path = os.path.join(self.out_dir, 'supervoxel2habitat_clustering_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.supervoxel2habitat_clustering, f)

        elif self.mode == 'testing':
            # Load clustering model from config file FIXME: 需要根据实际情况调整
            model_path = os.path.join(self.out_dir, 'supervoxel2habitat_clustering_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.supervoxel2habitat_clustering = pickle.load(f)
            else:
                raise ValueError(f"No clustering model found at {model_path}")
            
            if self.verbose:
                self.logger.info(f"Performing population-level clustering using already trained model: {model_path}...")

            # Perform population-level clustering using already trained model
            habitat_labels = self.supervoxel2habitat_clustering.predict(features_of_all_subjects) + 1  # Start numbering from 1
            optimal_n_clusters = self.supervoxel2habitat_clustering.n_clusters
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Add habitat labels to results
        mean_features_all['Habitats'] = habitat_labels

        # Create results DataFrame
        self.results_df = mean_features_all.copy()

        # Save results
        if save_results_csv:
            if self.verbose:
                self.logger.info("Saving results...")

            # 确保目录存在
            os.makedirs(self.out_dir, exist_ok=True)

            # 如果有原始配置文件，则直接复制到输出目录
            if self.config_file and os.path.exists(self.config_file):
                import shutil
                config_out_path = os.path.join(self.out_dir, 'config.yaml')
                shutil.copy2(self.config_file, config_out_path)
                if self.verbose:
                    self.logger.info(f"原始配置文件已复制到: {config_out_path}")
            else:
                # 没有原始配置文件时，仍然保存当前配置为JSON
                if self.verbose:
                    self.logger.info("未提供原始配置文件路径，将以JSON格式保存当前配置")
                
                # 创建包含当前分析参数的配置字典
                config = {
                    'data_dir': self.data_dir,
                    'out_folder': self.out_dir,
                    'feature_config': self.feature_config,
                    'supervoxel_clustering_method': self.supervoxel_method,
                    'habitat_clustering_method': self.habitat_method,
                    'mode': self.mode,
                    'n_clusters_supervoxel': self.n_clusters_supervoxel,
                    'n_clusters_habitats_max': self.n_clusters_habitats_max,
                    'n_clusters_habitats_min': self.n_clusters_habitats_min,
                    'habitat_cluster_selection_method': self.habitat_cluster_selection_method,
                    'best_n_clusters': self.best_n_clusters,
                    'n_processes': self.n_processes,
                    'random_state': self.random_state,
                    'verbose': self.verbose,
                    'images_dir': self.images_dir,
                    'masks_dir': self.masks_dir,
                    'plot_curves': self.plot_curves,
                    'save_intermediate_results': self.save_intermediate_results,
                    'optimal_n_clusters_habitat': int(optimal_n_clusters)
                }

                # 保存配置
                with open(os.path.join(self.out_dir, 'config.json'), 'w') as f:
                    json.dump(config, f, indent=4)

            # Save CSV
            self.results_df.to_csv(os.path.join(self.out_dir, 'habitats.csv'), index=False)

            # Save habitat images for each subject
            # Set Subject as index
            self.results_df.set_index('Subject', inplace=True)
            
            # 创建参数列表
            args_list = list(set(self.results_df.index))

            # 使用多进程保存所有主体的栖息地图像
            with multiprocessing.Pool(processes=self.n_processes) as pool:
                # 使用imap_unordered并配合自定义进度条，添加容错机制
                progress_bar = CustomTqdm(total=len(args_list), desc="Save habitat images")
                failed_subjects = []
                results_iter = pool.imap_unordered(self._save_habitat_for_subject, args_list)
                for result in results_iter:
                    if isinstance(result[1], Exception):
                        failed_subjects.append(result[0])
                        if self.verbose:
                            self.logger.error(f"Error saving habitat image for subject {result[0]}: {str(result[1])}")
                    else:
                        progress_bar.update(1)
                
                if failed_subjects and self.verbose:
                    self.logger.warning(f"Failed to save habitat images for {len(failed_subjects)}")

        return self.results_df

