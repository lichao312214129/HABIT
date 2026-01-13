"""
Habitat Clustering Analysis Module

This module implements a two-step (or one-step) clustering approach for 
tumor habitat analysis:
1. Individual-level clustering: Divide each tumor into supervoxels
2. Population-level clustering: Cluster supervoxels across patients to obtain habitats
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from typing import Dict, List, Any, Tuple, Optional, Union

# Suppress warnings
warnings.simplefilter('ignore')

# Internal imports
from habit.utils.io_utils import (
    get_image_and_mask_paths,
    detect_image_names,
    check_data_structure,
    save_habitat_image
)
from habit.utils.log_utils import setup_logger, get_module_logger, LoggerManager
from habit.utils.parallel_utils import parallel_map, ProcessingResult

# Local imports
from .config import HabitatConfig, ResultColumns, ClusteringConfig, IOConfig, RuntimeConfig
from .pipeline import create_pipeline, BasePipeline
from .features.feature_expression_parser import FeatureExpressionParser
from .features.feature_extractor_factory import create_feature_extractor
from .features.feature_preprocessing import preprocess_features
from .clustering.base_clustering import get_clustering_algorithm
from .clustering.cluster_validation_methods import (
    get_validation_methods,
    is_valid_method_for_algorithm,
    get_default_methods
)

# Visualization imports (optional)
try:
    from habit.utils.visualization import plot_cluster_scores, plot_cluster_results
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False


class HabitatAnalysis:
    """
    Habitat Analysis class for performing clustering analysis on medical images.
    
    Supports two clustering strategies:
    - one_step: Individual-level clustering only (each tumor gets its own habitats)
    - two_step: Individual + Population clustering (supervoxels then habitats across patients)
    
    Example:
        >>> # Using new config objects
        >>> config = HabitatConfig(
        ...     clustering=ClusteringConfig(strategy='two_step', n_clusters_supervoxel=50),
        ...     io=IOConfig(root_folder='/path/to/data'),
        ...     runtime=RuntimeConfig(n_processes=4),
        ...     feature_config={'voxel_level': {...}}
        ... )
        >>> analyzer = HabitatAnalysis(config=config)
        >>> results = analyzer.run()
        
        >>> # Using legacy parameters (backward compatible)
        >>> analyzer = HabitatAnalysis(
        ...     root_folder='/path/to/data',
        ...     n_clusters_supervoxel=50,
        ...     n_processes=4,
        ...     feature_config={'voxel_level': {...}}
        ... )
    """

    def __init__(
        self,
        config: Optional[HabitatConfig] = None,
        # Legacy parameters for backward compatibility
        root_folder: Optional[str] = None,
        out_folder: Optional[str] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        clustering_strategy: str = "two_step",
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
        progress_callback: Optional[callable] = None,
        save_intermediate_results: bool = False,
        config_file: Optional[str] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize HabitatAnalysis.
        
        Args:
            config: HabitatConfig object containing all configuration.
                    If provided, legacy parameters are ignored.
            
            Legacy parameters (used only if config is None):
                root_folder: Root directory of data
                out_folder: Output directory for results
                feature_config: Feature extraction configuration
                clustering_strategy: 'one_step' or 'two_step'
                supervoxel_clustering_method: Method for supervoxel clustering
                habitat_clustering_method: Method for habitat clustering
                mode: 'training' or 'testing'
                n_clusters_supervoxel: Number of supervoxel clusters
                n_clusters_habitats_max: Maximum habitat clusters to test
                n_clusters_habitats_min: Minimum habitat clusters to test
                habitat_cluster_selection_method: Method(s) for cluster selection
                best_n_clusters: Explicitly specify number of clusters
                one_step_settings: Settings for one-step mode
                n_processes: Number of parallel processes
                random_state: Random seed
                verbose: Enable verbose output
                images_dir: Name of images subdirectory
                masks_dir: Name of masks subdirectory
                plot_curves: Generate evaluation plots
                progress_callback: Progress callback function
                save_intermediate_results: Save intermediate results
                config_file: Path to original config file
                log_level: Logging level
        """
        # Build config from parameters if not provided
        if config is None:
            config = HabitatConfig.from_dict({
                'root_folder': root_folder,
                'out_folder': out_folder,
                'feature_config': feature_config or {},
                'clustering_strategy': clustering_strategy,
                'supervoxel_clustering_method': supervoxel_clustering_method,
                'habitat_clustering_method': habitat_clustering_method,
                'mode': mode,
                'n_clusters_supervoxel': n_clusters_supervoxel,
                'n_clusters_habitats_max': n_clusters_habitats_max,
                'n_clusters_habitats_min': n_clusters_habitats_min,
                'habitat_cluster_selection_method': habitat_cluster_selection_method,
                'best_n_clusters': best_n_clusters,
                'one_step_settings': one_step_settings,
                'n_processes': n_processes,
                'random_state': random_state,
                'verbose': verbose,
                'images_dir': images_dir,
                'masks_dir': masks_dir,
                'plot_curves': plot_curves,
                'progress_callback': progress_callback,
                'save_intermediate_results': save_intermediate_results,
                'config_file': config_file,
                'log_level': log_level,
            })
        
        self.config = config
        self._setup_logging()  # 设置日志
        self._validate_feature_config()  # 验证特征配置
        self._setup_data_paths()  # 设置数据路径
        self._init_feature_extractor()  # 初始化特征提取器
        self._init_clustering_algorithms()  # 初始化聚类算法
        self._init_selection_methods()  # 初始化聚类选择方法
        self._init_pipeline()  # 初始化管道
        self._init_results_storage()  # 初始化结果存储

    # =========================================================================
    # Initialization Methods
    # =========================================================================
    
    def _setup_logging(self) -> None:
        """
        Setup logging configuration.
        
        If logging has already been configured by CLI, use existing logger.
        Otherwise, create a new logger with file output.
        Also stores log configuration for child processes (multiprocessing support).
        """
        manager = LoggerManager()
        
        if manager.get_log_file() is not None:
            # Logging already configured by CLI
            self.logger = get_module_logger('habitat')
            self.logger.info("Using existing logging configuration from CLI entry point")
            self._log_file_path = manager.get_log_file()
            self._log_level = (
                manager._root_logger.getEffectiveLevel() 
                if manager._root_logger else logging.INFO
            )
        else:
            # Setup new logging
            level = self.config.runtime.get_log_level_int()
            self.logger = setup_logger(
                name='habitat',
                output_dir=self.config.io.out_folder,
                log_filename='habitat_analysis.log',
                level=level
            )
            self._log_file_path = manager.get_log_file()
            self._log_level = level
    
    def _ensure_logging_in_subprocess(self) -> None:
        """
        Ensure logging is properly configured in child processes.
        
        In Windows spawn mode, child processes don't inherit logging configuration.
        This method restores it.
        """
        from habit.utils.log_utils import restore_logging_in_subprocess
        
        if hasattr(self, '_log_file_path') and self._log_file_path:
            restore_logging_in_subprocess(self._log_file_path, self._log_level)
    
    def _validate_feature_config(self) -> None:
        """Validate feature configuration."""
        if 'voxel_level' not in self.config.feature_config:
            raise ValueError("voxel_level configuration is required")
        
        if 'supervoxel_level' in self.config.feature_config and self.config.runtime.verbose:
            self.logger.info(
                "Note: supervoxel_level feature configuration detected but "
                "not used in current implementation."
            )
    
    def _setup_data_paths(self) -> None:
        """Setup data paths and create output directory."""
        os.makedirs(self.config.io.out_folder, exist_ok=True)
        
        # Get image and mask paths
        self.images_paths, self.mask_paths = get_image_and_mask_paths(
            self.config.io.root_folder,
            keyword_of_raw_folder=self.config.io.images_dir,
            keyword_of_mask_folder=self.config.io.masks_dir
        )
        
        # Auto-detect image names if not provided
        voxel_config = self.config.feature_config['voxel_level']
        if isinstance(voxel_config, dict) and 'image_names' not in voxel_config:
            if self.config.runtime.verbose:
                self.logger.info(
                    "Image names not provided in voxel_level config, "
                    "automatically detecting from data directory..."
                )
            image_names = detect_image_names(self.images_paths)
            self.config.feature_config['voxel_level']['image_names'] = image_names
            if self.config.runtime.verbose:
                self.logger.info(f"Detected image names: {image_names}")
        
        # Validate data structure
        voxel_config = self.config.feature_config['voxel_level']
        if isinstance(voxel_config, dict) and 'image_names' in voxel_config:
            check_data_structure(
                self.images_paths, 
                self.mask_paths,
                voxel_config['image_names'], 
                None
            )
        else:
            raise ValueError("voxel_level configuration must contain 'image_names' field")
    
    def _init_feature_extractor(self) -> None:
        """Initialize feature extractor based on configuration."""
        self.expression_parser = FeatureExpressionParser()
        
        voxel_config = self.config.feature_config['voxel_level']
        if not isinstance(voxel_config, dict) or 'method' not in voxel_config:
            raise ValueError("voxel_level must be a dictionary with 'method' field")
        
        # Parse voxel_level expression
        (self.voxel_method, 
         self.voxel_params, 
         self.voxel_processing_steps) = self.expression_parser.parse(voxel_config)
        
        # Check for supervoxel_level configuration
        self.has_supervoxel_config = 'supervoxel_level' in self.config.feature_config
        if self.has_supervoxel_config:
            supervoxel_config = self.config.feature_config['supervoxel_level']
            if not isinstance(supervoxel_config, dict) or 'method' not in supervoxel_config:
                raise ValueError(
                    "supervoxel_level must be a dictionary with 'method' field"
                )
            
            (self.supervoxel_method_name,
             self.supervoxel_params,
             self.supervoxel_processing_steps) = self.expression_parser.parse(supervoxel_config)
            
            if self.config.runtime.verbose:
                self.logger.info(
                    "supervoxel_level feature configuration detected but "
                    "not used in current implementation"
                )
        
        # Prepare cross-image parameters
        self.cross_image_kwargs = self._prepare_cross_image_params()
    
    def _prepare_cross_image_params(self) -> Dict[str, Any]:
        """
        Prepare cross-image feature extractor parameters.
        
        Returns:
            Dictionary of cross-image parameters
        """
        cross_image_kwargs = {}
        
        if self.voxel_params:
            voxel_params = self.config.feature_config['voxel_level'].get('params', {})
            for param_name, param_value in self.voxel_params.items():
                if param_value == param_name and param_name in voxel_params:
                    cross_image_kwargs[param_name] = voxel_params[param_name]
                elif isinstance(param_value, str) and param_value in voxel_params:
                    cross_image_kwargs[param_name] = voxel_params[param_value]
                else:
                    cross_image_kwargs[param_name] = param_value
        
        return cross_image_kwargs
    
    def _init_clustering_algorithms(self) -> None:
        """Initialize clustering algorithm instances."""
        self.voxel2supervoxel_clustering = get_clustering_algorithm(
            self.config.clustering.supervoxel_method,
            n_clusters=self.config.clustering.n_clusters_supervoxel,
            random_state=self.config.clustering.random_state
        )
        
        self.supervoxel2habitat_clustering = get_clustering_algorithm(
            self.config.clustering.habitat_method,
            n_clusters=self.config.clustering.n_clusters_habitats_max,
            random_state=self.config.clustering.random_state
        )
    
    def _init_selection_methods(self) -> None:
        """Initialize and validate cluster selection methods."""
        validation_info = get_validation_methods(self.config.clustering.habitat_method)
        valid_methods = list(validation_info['methods'].keys())
        default_methods = get_default_methods(self.config.clustering.habitat_method)
        
        if self.config.runtime.verbose:
            self.logger.info(
                f"Validation methods supported by '{self.config.clustering.habitat_method}': "
                f"{', '.join(valid_methods)}"
            )
            self.logger.info(f"Default validation methods: {', '.join(default_methods)}")
        
        # Validate and set selection methods
        selection_methods = self.config.clustering.selection_methods
        
        if selection_methods is None:
            self.selection_methods = default_methods
            if self.config.runtime.verbose:
                self.logger.info(
                    f"No clustering evaluation method specified, "
                    f"using defaults: {', '.join(default_methods)}"
                )
        elif isinstance(selection_methods, str):
            if is_valid_method_for_algorithm(
                self.config.clustering.habitat_method, 
                selection_methods.lower()
            ):
                self.selection_methods = selection_methods.lower()
            else:
                self.selection_methods = default_methods
                if self.config.runtime.verbose:
                    self.logger.warning(
                        f"Validation method '{selection_methods}' is invalid for "
                        f"'{self.config.clustering.habitat_method}'"
                    )
                    self.logger.info(f"Using default methods: {', '.join(default_methods)}")
        elif isinstance(selection_methods, list):
            valid = [
                m.lower() for m in selection_methods 
                if is_valid_method_for_algorithm(
                    self.config.clustering.habitat_method, m.lower()
                )
            ]
            invalid = [
                m for m in selection_methods 
                if not is_valid_method_for_algorithm(
                    self.config.clustering.habitat_method, m.lower()
                )
            ]
            
            if valid:
                self.selection_methods = valid
                if invalid and self.config.runtime.verbose:
                    self.logger.warning(
                        f"Invalid methods for '{self.config.clustering.habitat_method}': "
                        f"{', '.join(invalid)}"
                    )
            else:
                self.selection_methods = default_methods
                if self.config.runtime.verbose:
                    self.logger.warning("All specified methods are invalid")
                    self.logger.info(f"Using default methods: {', '.join(default_methods)}")
        else:
            self.selection_methods = default_methods
    
    def _init_pipeline(self) -> None:
        """Initialize the appropriate pipeline based on mode."""
        self.pipeline = create_pipeline(self.config, self.logger)
    
    def _init_results_storage(self) -> None:
        """Initialize results storage containers."""
        self.X = None
        self.supervoxel_labels = {}
        self.habitat_labels = None
        self.results_df = None

    # =========================================================================
    # Feature Extraction Methods
    # =========================================================================
    
    def extract_voxel_features(
        self, 
        subject: str
    ) -> Tuple[str, pd.DataFrame, pd.DataFrame, dict]:
        """
        Extract voxel-level features for a single subject.
        
        Args:
            subject: Subject ID to process
            
        Returns:
            Tuple of (subject_id, feature_df, raw_df, mask_info)
        """
        img_paths = self.images_paths[subject]
        mask_paths = self.mask_paths[subject]
        
        # Process each image according to processing steps
        processed_images = []
        for step in self.voxel_processing_steps:
            method = step['method']
            img_name = step['image']
            step_params = step['params'].copy()
            
            # Resolve parameter values
            if step_params:
                voxel_params = self.config.feature_config['voxel_level'].get('params', {})
                for param_name, param_value in list(step_params.items()):
                    if param_value == param_name and param_name in voxel_params:
                        step_params[param_name] = voxel_params[param_name]
                    elif isinstance(param_value, str) and param_value in voxel_params:
                        step_params[param_name] = voxel_params[param_value]
            
            # Create extractor and process
            step_params.update({'subject': subject, 'image': img_name})
            extractor = create_feature_extractor(method, **step_params)
            processed_df = extractor.extract_features(
                img_paths.get(img_name), 
                mask_paths.get(img_name), 
                **step_params
            )
            processed_images.append(processed_df)
        
        # Create cross-image feature extractor
        cross_image_kwargs = self.cross_image_kwargs.copy()
        cross_image_kwargs.update({'subject': subject})
        cross_image_extractor = create_feature_extractor(
            self.voxel_method, **cross_image_kwargs
        )
        features = cross_image_extractor.extract_features(
            processed_images, **cross_image_kwargs
        )
        
        # Get raw data
        raw_df = pd.concat(processed_images, axis=1)
        
        # Save mask information for image reconstruction
        mask = self.mask_paths[subject]
        mask = list(mask.values())[0]
        mask_img = sitk.ReadImage(mask)
        mask_array = sitk.GetArrayFromImage(mask_img)
        mask_info = {
            'mask': mask_img,
            'mask_array': mask_array
        }
        
        del processed_images, mask_img, mask_array
        
        return subject, features, raw_df, mask_info
    
    def extract_supervoxel_features(
        self, 
        subject: str
    ) -> Tuple[str, Union[pd.DataFrame, Exception]]:
        """
        Extract supervoxel-level features from supervoxel maps.
        
        Args:
            subject: Subject ID to process
            
        Returns:
            Tuple of (subject_id, features_df or Exception)
        """
        self._ensure_logging_in_subprocess()
        
        try:
            self.logger.info(f"Extracting supervoxel-level features for subject {subject}...")
            print(f"Extracting supervoxel-level features for subject {subject}...")
            
            img_paths = self.images_paths[subject]
            mask_path = self.supervoxel_file_dict[subject]
            
            # Process each image
            processed_images = []
            for step in self.supervoxel_processing_steps:
                method = step['method']
                img_name = step['image']
                step_params = step['params'].copy()
                step_params.update({'subject': subject, 'image': img_name})
                
                single_image_extractor = create_feature_extractor(method, **step_params)
                processed_df = single_image_extractor.extract_features(
                    img_paths.get(img_name), mask_path, **step_params
                )
                processed_images.append(processed_df)
            
            # Create cross-image extractor
            supervoxel_params = self.supervoxel_params.copy()
            cross_image_extractor = create_feature_extractor(
                self.supervoxel_method_name, **supervoxel_params
            )
            features = cross_image_extractor.extract_features(
                processed_images, **supervoxel_params
            )
            
            return subject, features
            
        except Exception as e:
            return subject, Exception(str(e))

    # =========================================================================
    # Clustering Methods
    # =========================================================================
    
    def _voxel2supervoxel_clustering(
        self, 
        subject: str
    ) -> Tuple[str, Union[pd.DataFrame, Exception]]:
        """
        Process a single subject: extract features and cluster to supervoxels.
        
        Args:
            subject: Subject ID to process
            
        Returns:
            Tuple of (subject_id, mean_features_df or Exception)
        """
        self._ensure_logging_in_subprocess()
        
        try:
            self.logger.info(f"Processing subject: {subject}")
            print(f"Processing subject: {subject}")
            
            # Extract features
            _, feature_df, raw_df, mask_info = self.extract_voxel_features(subject)
            
            # Apply subject-level preprocessing if enabled
            feature_df = self._apply_subject_preprocessing(feature_df)
            
            # Perform clustering
            supervoxel_labels = self._cluster_subject_voxels(subject, feature_df)
            
            # Calculate mean features per supervoxel
            mean_features_df = self._calculate_supervoxel_means(
                subject, feature_df, raw_df, supervoxel_labels
            )
            
            # Save supervoxel image
            self._save_supervoxel_image(subject, supervoxel_labels, mask_info)
            
            # Visualize if enabled
            if self.config.runtime.plot_curves and HAS_VISUALIZATION:
                self._visualize_supervoxel_clustering(subject, feature_df, supervoxel_labels)
            
            # Cleanup
            del feature_df, raw_df, supervoxel_labels
            
            return subject, mean_features_df
            
        except Exception as e:
            self.logger.error(
                f"Error in _voxel2supervoxel_clustering for subject {subject}: {e}"
            )
            return subject, Exception(str(e))
    
    def _apply_subject_preprocessing(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to subject-level features.
        
        Args:
            feature_df: Feature DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        preprocessing_config = self.config.feature_config.get(
            'preprocessing_for_subject_level', False
        )
        
        if preprocessing_config and 'methods' in preprocessing_config:
            X_preprocessed = preprocess_features(
                feature_df.values,
                methods=preprocessing_config['methods']
            )
            return pd.DataFrame(X_preprocessed, columns=feature_df.columns)
        
        return feature_df
    
    def _cluster_subject_voxels(
        self, 
        subject: str, 
        feature_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Cluster voxels to supervoxels for a single subject.
        
        Args:
            subject: Subject ID
            feature_df: Feature DataFrame
            
        Returns:
            Array of supervoxel labels (1-indexed)
        """
        try:
            if self.config.clustering.strategy == 'one_step':
                optimal_n_clusters = self._get_one_step_optimal_clusters(
                    subject, feature_df
                )
                
                # Update clustering algorithm
                self.voxel2supervoxel_clustering = get_clustering_algorithm(
                    self.config.clustering.supervoxel_method,
                    n_clusters=optimal_n_clusters,
                    random_state=self.config.clustering.random_state
                )
            
            self.voxel2supervoxel_clustering.fit(feature_df.values)
            supervoxel_labels = self.voxel2supervoxel_clustering.predict(feature_df.values)
            supervoxel_labels += 1  # 1-indexed
            
            return supervoxel_labels
            
        except Exception as e:
            self.logger.error(
                f"Error performing supervoxel clustering for subject {subject}: {e}"
            )
            raise
    
    def _get_one_step_optimal_clusters(
        self, 
        subject: str, 
        feature_df: pd.DataFrame
    ) -> int:
        """
        Determine optimal cluster number for one-step mode.
        
        Args:
            subject: Subject ID
            feature_df: Feature DataFrame
            
        Returns:
            Optimal number of clusters
        """
        one_step = self.config.one_step
        
        # Use fixed number if specified
        if one_step and one_step.best_n_clusters is not None:
            self.logger.info(
                f"Subject {subject}: Using fixed cluster number {one_step.best_n_clusters}"
            )
            print(f"Subject {subject}: Using fixed cluster number {one_step.best_n_clusters}")
            return one_step.best_n_clusters
        
        # Find optimal using validation methods
        self.logger.info(
            f"Determining optimal clusters for {subject} using {one_step.selection_method}"
        )
        
        clusterer = get_clustering_algorithm(
            self.config.clustering.supervoxel_method,
            n_clusters=one_step.max_clusters,
            random_state=self.config.clustering.random_state
        )
        
        optimal_n_clusters, scores_dict = clusterer.find_optimal_clusters(
            X=feature_df.values,
            min_clusters=one_step.min_clusters,
            max_clusters=one_step.max_clusters,
            methods=[one_step.selection_method],
            show_progress=False
        )
        
        # Plot validation curves if requested
        if one_step.plot_validation_curves and self.config.runtime.plot_curves:
            self._plot_one_step_validation(subject, scores_dict, clusterer)
        
        self.logger.info(f"Subject {subject}: optimal clusters = {optimal_n_clusters}")
        print(f"Subject {subject}: optimal clusters = {optimal_n_clusters}")
        
        return optimal_n_clusters
    
    def _plot_one_step_validation(
        self, 
        subject: str, 
        scores_dict: Dict, 
        clusterer: Any
    ) -> None:
        """Plot validation curves for one-step clustering."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            viz_dir = os.path.join(
                self.config.io.out_folder, 'visualizations', 'optimal_clusters'
            )
            os.makedirs(viz_dir, exist_ok=True)
            
            plot_file = os.path.join(viz_dir, f'{subject}_cluster_validation.png')
            plot_cluster_scores(
                scores_dict=scores_dict,
                cluster_range=clusterer.cluster_range,
                methods=[self.config.one_step.selection_method],
                clustering_algorithm=self.config.clustering.supervoxel_method,
                figsize=(8, 6),
                save_path=plot_file,
                show=False,
                dpi=300
            )
            self.logger.info(f"Validation plot saved to: {plot_file}")
        except Exception as e:
            self.logger.warning(f"Failed to plot validation curves for {subject}: {e}")
    
    def _calculate_supervoxel_means(
        self,
        subject: str,
        feature_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        supervoxel_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate mean features for each supervoxel.
        
        Args:
            subject: Subject ID
            feature_df: Processed feature DataFrame
            raw_df: Original feature DataFrame
            supervoxel_labels: Array of supervoxel labels
            
        Returns:
            DataFrame with mean features per supervoxel
        """
        feature_names = feature_df.columns.tolist()
        original_feature_names = raw_df.columns.tolist()
        
        unique_labels = np.arange(1, self.config.clustering.n_clusters_supervoxel + 1)
        data_rows = []
        
        for label in unique_labels:
            indices = supervoxel_labels == label
            if np.any(indices):
                mean_features = np.mean(feature_df[indices], axis=0)
                mean_original = np.mean(raw_df.values[indices], axis=0)
                count = np.sum(indices)
                
                data_row = {
                    ResultColumns.SUBJECT: subject,
                    ResultColumns.SUPERVOXEL: label,
                    ResultColumns.COUNT: count,
                }
                
                # Add processed feature means
                for j, name in enumerate(feature_names):
                    data_row[name] = mean_features[j]
                
                # Add original feature means
                for j, name in enumerate(original_feature_names):
                    data_row[f"{name}{ResultColumns.ORIGINAL_SUFFIX}"] = mean_original[j]
                
                data_rows.append(data_row)
        
        return pd.DataFrame(data_rows)
    
    def _save_supervoxel_image(
        self,
        subject: str,
        supervoxel_labels: np.ndarray,
        mask_info: dict
    ) -> None:
        """Save supervoxel labels as an image file."""
        if not isinstance(mask_info, dict):
            return
        if 'mask_array' not in mask_info or 'mask' not in mask_info:
            return
        
        supervoxel_map = np.zeros_like(mask_info['mask_array'])
        mask_indices = mask_info['mask_array'] > 0
        supervoxel_map[mask_indices] = supervoxel_labels
        
        supervoxel_img = sitk.GetImageFromArray(supervoxel_map)
        supervoxel_img.CopyInformation(mask_info['mask'])
        
        output_path = os.path.join(
            self.config.io.out_folder, f"{subject}_supervoxel.nrrd"
        )
        sitk.WriteImage(supervoxel_img, output_path)
        
        del supervoxel_map, mask_indices
    
    def _visualize_supervoxel_clustering(
        self,
        subject: str,
        feature_df: pd.DataFrame,
        supervoxel_labels: np.ndarray
    ) -> None:
        """Create visualizations for supervoxel clustering results."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            viz_dir = os.path.join(
                self.config.io.out_folder, 'visualizations', 'supervoxel_clustering'
            )
            os.makedirs(viz_dir, exist_ok=True)
            
            centers = None
            if hasattr(self.voxel2supervoxel_clustering, 'cluster_centers_'):
                centers = self.voxel2supervoxel_clustering.cluster_centers_
            
            title = (
                f'Supervoxel Clustering: {subject}\n'
                f'(n_clusters={self.config.clustering.n_clusters_supervoxel})'
            )
            
            # 2D scatter
            plot_cluster_results(
                X=feature_df.values,
                labels=supervoxel_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, f'{subject}_supervoxel_clustering_2D.png'),
                show=False,
                dpi=300,
                plot_3d=False
            )
            
            # 3D scatter
            plot_cluster_results(
                X=feature_df.values,
                labels=supervoxel_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, f'{subject}_supervoxel_clustering_3D.png'),
                show=False,
                dpi=300,
                plot_3d=True
            )
            
            if self.config.runtime.verbose:
                self.logger.info(f"Saved supervoxel clustering visualizations to {viz_dir}")
                
        except Exception as e:
            if self.config.runtime.verbose:
                self.logger.warning(f"Failed to create visualization for {subject}: {e}")

    def _save_habitat_for_subject(
        self, 
        subject: str
    ) -> Tuple[str, Optional[Exception]]:
        """
        Save habitat image for a single subject.
        
        Args:
            subject: Subject ID
            
        Returns:
            Tuple of (subject_id, None or Exception)
        """
        self._ensure_logging_in_subprocess()
        
        try:
            supervoxel_path = os.path.join(
                self.config.io.out_folder, f"{subject}_supervoxel.nrrd"
            )
            save_habitat_image(
                subject, self.results_df, supervoxel_path, self.config.io.out_folder
            )
            return subject, None
        except Exception as e:
            return subject, Exception(str(e))

    # =========================================================================
    # Main Pipeline Methods
    # =========================================================================
    
    def run(
        self, 
        subjects: Optional[List[str]] = None, 
        save_results_csv: bool = True
    ) -> pd.DataFrame:
        """
        Run the habitat clustering pipeline.
        
        Args:
            subjects: List of subjects to process (None = all subjects)
            save_results_csv: Whether to save results as CSV files
            
        Returns:
            DataFrame with habitat clustering results
        """
        if subjects is None:
            subjects = list(self.images_paths.keys())
        
        # Step 1: Extract features and perform supervoxel clustering
        mean_features_all, failed_subjects = self._process_all_subjects(subjects)
        
        if len(mean_features_all) == 0:
            raise ValueError("No valid features for analysis")
        
        # Step 2: Prepare population-level features
        features_for_clustering = self._prepare_population_features(
            mean_features_all, subjects, failed_subjects
        )
        
        # Step 3: Perform population-level clustering (or skip for one-step)
        self.results_df = self._perform_population_clustering(
            mean_features_all, features_for_clustering
        )
        
        # Step 4: Save results
        if save_results_csv:
            self._save_results(subjects, failed_subjects)
        
        return self.results_df
    
    def _process_all_subjects(
        self, 
        subjects: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Process all subjects: extract features and perform supervoxel clustering.
        
        Args:
            subjects: List of subject IDs
            
        Returns:
            Tuple of (combined_features_df, failed_subjects_list)
        """
        if self.config.runtime.verbose:
            self.logger.info("Extracting features and performing supervoxel clustering...")
        
        results, failed_subjects = parallel_map(
            func=self._voxel2supervoxel_clustering,
            items=subjects,
            n_processes=self.config.runtime.n_processes,
            desc="Processing subjects",
            logger=self.logger,
            show_progress=True,
            log_file_path=self._log_file_path,
            log_level=self._log_level,
        )
        
        # Combine results
        mean_features_all = pd.DataFrame()
        for result in results:
            if result.success and result.result is not None:
                mean_features_all = pd.concat(
                    [mean_features_all, result.result], 
                    ignore_index=True
                )
        
        if self.config.runtime.verbose:
            if failed_subjects:
                self.logger.warning(f"Failed to process {len(failed_subjects)} subject(s)")
            self.logger.info(
                f"All {len(subjects)} subjects have been processed. "
                "Proceeding to clustering..."
            )
        
        return mean_features_all, failed_subjects
    
    def _prepare_population_features(
        self,
        mean_features_all: pd.DataFrame,
        subjects: List[str],
        failed_subjects: List[str]
    ) -> pd.DataFrame:
        """
        Prepare features for population-level clustering.
        
        Args:
            mean_features_all: Combined supervoxel features
            subjects: List of all subject IDs
            failed_subjects: List of failed subject IDs
            
        Returns:
            DataFrame ready for population-level clustering
        """
        # Get feature columns (exclude metadata columns)
        feature_columns = [
            col for col in mean_features_all.columns 
            if ResultColumns.is_feature_column(col)
        ]
        features = mean_features_all[feature_columns]
        
        # Setup supervoxel file dictionary
        self._setup_supervoxel_files(subjects, failed_subjects)
        
        # Check if we need to extract supervoxel-level features
        if self._should_extract_supervoxel_features():
            features = self._extract_all_supervoxel_features(subjects, failed_subjects)
        
        # Clean and preprocess features
        features = self._clean_features(features)
        features = self._apply_group_preprocessing(features)
        
        # Handle mean values for training/testing
        self._handle_mean_values(features)
        
        return features
    
    def _setup_supervoxel_files(
        self, 
        subjects: List[str], 
        failed_subjects: List[str]
    ) -> None:
        """Setup dictionary mapping subjects to supervoxel files."""
        supervoxel_keyword = self.config.feature_config['supervoxel_level'].get(
            'supervoxel_file_keyword', '*_supervoxel.nrrd'
        )
        supervoxel_files = glob(
            os.path.join(self.config.io.out_folder, supervoxel_keyword)
        )
        
        self.supervoxel_file_dict = {}
        for subject in subjects:
            for supervoxel_file in supervoxel_files:
                if subject in supervoxel_file:
                    self.supervoxel_file_dict[subject] = supervoxel_file
                    break
            else:
                if subject not in failed_subjects and self.config.runtime.verbose:
                    self.logger.warning(f"No supervoxel file found for subject {subject}")
        
        if not self.supervoxel_file_dict:
            raise ValueError(
                f"No supervoxel files found in {self.config.io.out_folder}"
            )
    
    def _should_extract_supervoxel_features(self) -> bool:
        """Check if supervoxel-level feature extraction is needed."""
        method = self.config.feature_config['supervoxel_level']['method']
        return 'mean_voxel_features' not in method
    
    def _extract_all_supervoxel_features(
        self,
        subjects: List[str],
        failed_subjects: List[str]
    ) -> pd.DataFrame:
        """Extract supervoxel-level features for all subjects."""
        if self.config.runtime.verbose:
            self.logger.info("Extracting supervoxel-level features...")
        
        results, new_failed = parallel_map(
            func=self.extract_supervoxel_features,
            items=subjects,
            n_processes=self.config.runtime.n_processes,
            desc="Extracting supervoxel features",
            logger=self.logger,
            show_progress=True,
            log_file_path=self._log_file_path,
            log_level=self._log_level,
        )
        
        failed_subjects.extend(new_failed)
        
        if self.config.runtime.verbose and new_failed:
            self.logger.warning(
                f"Failed to extract supervoxel features for {len(new_failed)} subject(s)"
            )
        
        # Combine results
        features_list = [r.result for r in results if r.success and r.result is not None]
        if features_list:
            return pd.concat(features_list, ignore_index=True)
        else:
            raise ValueError("No valid supervoxel features extracted")
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean feature DataFrame: handle types, inf, nan values."""
        features = features.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        features = features.applymap(
            lambda x: x.item() if hasattr(x, 'item') else x
        )
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.mean())
        return features
    
    def _apply_group_preprocessing(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply group-level preprocessing if configured."""
        preprocessing_config = self.config.feature_config.get(
            'preprocessing_for_group_level', False
        )
        
        if preprocessing_config and 'methods' in preprocessing_config:
            processed = preprocess_features(
                features.values,
                methods=preprocessing_config['methods']
            )
            return pd.DataFrame(processed, columns=features.columns)
        
        return features
    
    def _handle_mean_values(self, features: pd.DataFrame) -> None:
        """Handle mean values for training/testing mode."""
        if self.config.runtime.mode == 'training':
            # Save mean values for testing
            if hasattr(self.pipeline, 'save_mean_values'):
                self.pipeline.save_mean_values(features)
        elif self.config.runtime.mode == 'testing':
            # Load and apply mean values
            if hasattr(self.pipeline, 'load_mean_values'):
                mean_values = self.pipeline.load_mean_values()
                features.fillna(mean_values, inplace=True)
    
    def _perform_population_clustering(
        self,
        mean_features_all: pd.DataFrame,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform population-level clustering to determine habitats.
        
        Args:
            mean_features_all: Original combined features with metadata
            features: Cleaned features for clustering
            
        Returns:
            Results DataFrame with habitat labels
        """
        # One-step mode: supervoxels ARE habitats
        if self.config.clustering.strategy == 'one_step':
            if self.config.runtime.verbose:
                self.logger.info(
                    "One-step clustering mode: skipping population-level clustering"
                )
                self.logger.info("Using individual-level clusters as final habitats")
            
            mean_features_all[ResultColumns.HABITATS] = mean_features_all[ResultColumns.SUPERVOXEL]
            return mean_features_all.copy()
        
        # Two-step mode: perform population-level clustering
        habitat_labels, optimal_n_clusters, scores = self.pipeline.cluster_habitats(
            features, self.supervoxel2habitat_clustering
        )
        
        # Plot scores if available
        if scores and self.config.runtime.plot_curves:
            self._plot_habitat_scores(scores, optimal_n_clusters)
        
        # Visualize clustering results
        if self.config.runtime.plot_curves and HAS_VISUALIZATION:
            self._visualize_habitat_clustering(features, habitat_labels, optimal_n_clusters)
        
        # Save model for training mode
        if self.config.runtime.mode == 'training':
            self.pipeline.save_model(
                self.supervoxel2habitat_clustering,
                'supervoxel2habitat_clustering_strategy'
            )
        
        # Add habitat labels to results
        mean_features_all[ResultColumns.HABITATS] = habitat_labels
        
        return mean_features_all.copy()
    
    def _plot_habitat_scores(self, scores: Dict, optimal_n_clusters: int) -> None:
        """Plot habitat clustering validation scores."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            os.makedirs(self.config.io.out_folder, exist_ok=True)
            save_path = os.path.join(
                self.config.io.out_folder, 'habitat_clustering_scores.png'
            )
            
            # Get cluster range from the algorithm
            cluster_for_plot = get_clustering_algorithm(self.config.clustering.habitat_method)
            
            plot_cluster_scores(
                scores_dict=scores,
                cluster_range=cluster_for_plot.cluster_range,
                methods=self.selection_methods,
                clustering_algorithm=self.config.clustering.habitat_method,
                figsize=(12, 8),
                save_path=save_path,
                show=False
            )
            
            if self.config.runtime.verbose:
                self.logger.info(f"Clustering scores plot saved to {save_path}")
                
        except Exception as e:
            if self.config.runtime.verbose:
                self.logger.error(f"Error plotting clustering scores: {e}")
                self.logger.info("Continuing with other processes...")
    
    def _visualize_habitat_clustering(
        self,
        features: pd.DataFrame,
        habitat_labels: np.ndarray,
        optimal_n_clusters: int
    ) -> None:
        """Create visualizations for habitat clustering results."""
        if not HAS_VISUALIZATION:
            return
            
        try:
            viz_dir = os.path.join(
                self.config.io.out_folder, 'visualizations', 'habitat_clustering'
            )
            os.makedirs(viz_dir, exist_ok=True)
            
            centers = None
            if hasattr(self.supervoxel2habitat_clustering, 'cluster_centers_'):
                centers = self.supervoxel2habitat_clustering.cluster_centers_
            
            title = (
                f'Habitat Clustering (Population Level)\n'
                f'(n_clusters={optimal_n_clusters})'
            )
            
            # 2D scatter
            plot_cluster_results(
                X=features,
                labels=habitat_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, 'habitat_clustering_2D.png'),
                show=False,
                dpi=300,
                plot_3d=False
            )
            
            # 3D scatter
            plot_cluster_results(
                X=features,
                labels=habitat_labels,
                centers=centers,
                title=title,
                save_path=os.path.join(viz_dir, 'habitat_clustering_3D.png'),
                show=False,
                dpi=300,
                plot_3d=True
            )
            
            if self.config.runtime.verbose:
                self.logger.info(f"Saved habitat clustering visualizations to {viz_dir}")
                
        except Exception as e:
            if self.config.runtime.verbose:
                self.logger.warning(f"Failed to create habitat clustering visualization: {e}")
    
    def _save_results(self, subjects: List[str], failed_subjects: List[str]) -> None:
        """Save all results including config, CSV, and habitat images."""
        if self.config.runtime.verbose:
            self.logger.info("Saving results...")
        
        os.makedirs(self.config.io.out_folder, exist_ok=True)
        
        # Determine optimal clusters for config saving
        optimal_n_clusters = None
        if hasattr(self.supervoxel2habitat_clustering, 'n_clusters'):
            optimal_n_clusters = self.supervoxel2habitat_clustering.n_clusters
        
        # Save configuration
        self.pipeline.save_config(optimal_n_clusters)
        
        # Save results CSV
        csv_path = os.path.join(self.config.io.out_folder, 'habitats.csv')
        self.results_df.to_csv(csv_path, index=False)
        if self.config.runtime.verbose:
            self.logger.info(f"Results saved to {csv_path}")
        
        # Save habitat images for each subject
        self._save_all_habitat_images(failed_subjects)
    
    def _save_all_habitat_images(self, failed_subjects: List[str]) -> None:
        """Save habitat images for all successfully processed subjects."""
        # Set Subject as index for save function
        self.results_df.set_index(ResultColumns.SUBJECT, inplace=True)
        
        # Get unique subjects
        subjects_to_save = list(set(self.results_df.index))
        
        if self.config.runtime.verbose:
            self.logger.info(f"Saving habitat images for {len(subjects_to_save)} subjects...")
        
        results, failed = parallel_map(
            func=self._save_habitat_for_subject,
            items=subjects_to_save,
            n_processes=self.config.runtime.n_processes,
            desc="Saving habitat images",
            logger=self.logger,
            show_progress=True,
            log_file_path=self._log_file_path,
            log_level=self._log_level,
        )
        
        if failed and self.config.runtime.verbose:
            self.logger.warning(
                f"Failed to save habitat images for {len(failed)} subject(s)"
            )
