"""
Habitat Clustering Analysis Module
"""

import os
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import warnings
import multiprocessing
import matplotlib.pyplot as plt
from glob import glob
from typing import Dict, List, Any, Tuple, Optional, Union, Callable

from .utils.io_utils import (get_image_and_mask_paths,
                            detect_image_names,
                            check_data_structure,
                            save_habitat_image)

from .features.feature_extractor_factory import create_feature_extractor
from .clustering.base_clustering import get_clustering_algorithm
from .clustering.cluster_validation_methods import (
    get_validation_methods, 
    is_valid_method_for_algorithm, 
    get_default_methods
)
from .features.feature_preprocessing import preprocess_features

# Ignore warnings
warnings.simplefilter('ignore')

# Define custom progress bar function
def print_progress_bar(i: int, total: int, prefix: str = '', suffix: str = '') -> None:
    """
    Custom progress bar
    
    Args:
        i (int): Current iteration index
        total (int): Total iterations
        prefix (str): Prefix string
        suffix (str): Suffix string
    """
    progress = int((i + 1) / total * 50)  # 50 is the length of the progress bar
    bar = "█" * progress + "-" * (50 - progress)
    percent = (i + 1) / total * 100
    print(f"\r{prefix}[{bar}] {percent:.2f}% ({i+1}/{total}) {suffix}", end="")
    if i == total - 1:
        print()


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
                 supervoxel_clustering_method: str = "kmeans", 
                 habitat_clustering_method: str = "kmeans",
                 n_clusters_supervoxel: int = 50,
                 n_clusters_habitats_max: int = 10,
                 n_clusters_habitats_min: int = 2,
                 habitat_cluster_selection_method: Optional[Union[str, List[str]]] = None,
                 best_n_clusters: Optional[int] = None,
                 n_processes: int = 1,
                 random_state: int = 42,
                 verbose: bool = True,
                 images_dir: str = "images",
                 masks_dir: str = "masks",
                 plot_curves: bool = True,
                 progress_callback: Optional[Callable] = None,
                 save_intermediate_results: bool = False):
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
            supervoxel_clustering_method (str, optional): Supervoxel clustering method, default is kmeans
            habitat_clustering_method (str, optional): Habitat clustering method, default is kmeans
            n_clusters_supervoxel (int, optional): Number of supervoxel clusters, default is 50
            n_clusters_habitats_max (int, optional): Maximum number of habitat clusters, default is 10
            n_clusters_habitats_min (int, optional): Minimum number of habitat clusters, default is 2
            habitat_cluster_selection_method (str, optional): Method for selecting number of habitat clusters
            best_n_clusters (int, optional): Directly specify the best number of clusters
            n_processes (int, optional): Number of parallel processes, default is 1
            random_state (int, optional): Random seed, default is 42
            verbose (bool, optional): Whether to output detailed information, default is True
            images_dir (str, optional): Image directory name, default is images
            masks_dir (str, optional): Mask directory name, default is masks
            plot_curves (bool, optional): Whether to plot evaluation curves, default is True
            progress_callback (callable, optional): Progress callback function
            save_intermediate_results (bool, optional): Whether to save intermediate results, default is False
        """
        # Basic parameters
        self.data_dir = os.path.abspath(root_folder)
        self.out_dir = os.path.abspath(out_folder or os.path.join(self.data_dir, "habitats_output"))
        
        # Clustering parameters
        self.supervoxel_method = supervoxel_clustering_method
        self.habitat_method = habitat_clustering_method
        self.n_clusters_supervoxel = n_clusters_supervoxel
        self.n_clusters_habitats_max = n_clusters_habitats_max
        self.n_clusters_habitats_min = n_clusters_habitats_min

        self.verbose = verbose
        
        # Get validation methods supported by the clustering method using cluster_validation_methods module
        validation_info = get_validation_methods(habitat_clustering_method)
        valid_selection_methods = list(validation_info['methods'].keys())
        default_methods = get_default_methods(habitat_clustering_method)
        
        if self.verbose:
            print(f"Validation methods supported by clustering method '{habitat_clustering_method}': {', '.join(valid_selection_methods)}")
            print(f"Default validation methods: {', '.join(default_methods)}")
        
        # Check validity of habitat_cluster_selection_method parameter
        if habitat_cluster_selection_method is None:
            # Use default methods
            self.habitat_cluster_selection_method = default_methods
            if self.verbose:
                print(f"No clustering evaluation method specified, using default methods: {', '.join(default_methods)}")
        elif isinstance(habitat_cluster_selection_method, str):
            # Check if single method is valid
            if is_valid_method_for_algorithm(habitat_clustering_method, habitat_cluster_selection_method.lower()):
                self.habitat_cluster_selection_method = habitat_cluster_selection_method.lower()
            else:
                # If specified method is invalid for current clustering algorithm, use default methods
                original_method = habitat_cluster_selection_method
                self.habitat_cluster_selection_method = default_methods
                if self.verbose:
                    print(f"WarninValidation methodg: Validation method '{original_method}' is invalid for clustering method '{habitat_clustering_method}'")
                    print(f"Available validation methods: {', '.join(valid_selection_methods)}")
                    print(f"Using default methods: {', '.join(default_methods)}")
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
                    print(f"Warning: The following validation methods are invalid for clustering method '{habitat_clustering_method}': {', '.join(invalid_methods)}")
                    print(f"Only using valid methods: {', '.join(valid_methods)}")
            else:
                # If no valid methods, use default methods
                self.habitat_cluster_selection_method = default_methods
                if self.verbose:
                    print(f"Warning: All specified validation methods are invalid for clustering method '{habitat_clustering_method}'")
                    print(f"Available validation methods: {', '.join(valid_selection_methods)}")
                    print(f"Using default methods: {', '.join(default_methods)}")
        else:
            # Parameter type error, use default methods
            self.habitat_cluster_selection_method = default_methods
            if self.verbose:
                print(f"Warning: Clustering evaluation method parameter type error, should be string or list")
                print(f"Using default methods: {', '.join(default_methods)}")
        
        
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
        self.feature_config = feature_config
        
        # Create output directory
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Image paths
        self.images_paths, self.mask_paths = get_image_and_mask_paths(
            self.data_dir, keyword_of_raw_folder=self.images_dir, 
            keyword_of_mask_folder=self.masks_dir
        )
        
        # If image names not provided, detect from data directory
        if self.feature_config['image_names'] is None:
            if self.verbose:
                print("Image names not provided, automatically detecting from data directory...")
            self.feature_config['image_names'] = detect_image_names(self.images_paths)
            if self.verbose:
                print(f"Detected image names: {self.feature_config['image_names']}")
        
        # Initialize feature extractor
        self._init_feature_extractor()
        
        # Initialize clustering algorithms
        self.supervoxel_clustering = get_clustering_algorithm(
            self.supervoxel_method, 
            n_clusters=self.n_clusters_supervoxel, 
            random_state=self.random_state
        )
        
        self.habitat_clustering = get_clustering_algorithm(
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
        check_data_structure(self.images_paths, self.mask_paths, 
                            self.feature_config['image_names'], None)
    
    def _init_feature_extractor(self) -> None:
        """
        Initialize feature extractor
        """
        try:
            extractor_config = self.feature_config['method']
            feature_params = self.feature_config.get('params', {})
            
            if isinstance(extractor_config, dict):
                # If dictionary, copy configuration to avoid modifying original
                config = extractor_config.copy()
                # Get feature extractor name
                name = config.pop('name', None)
                if not name:
                    raise ValueError("Feature extractor configuration must include 'name' field")
                
                # Create feature extractor, passing all other parameters
                self.feature_extractor = create_feature_extractor(name, **config)
            else:
                # Otherwise, treat as string name
                self.feature_extractor = create_feature_extractor(
                    extractor_config, 
                    **feature_params
                )
            
            if self.verbose:
                print(f"Using feature extractor: {self.feature_extractor.__class__.__name__}")
        except Exception as e:
            raise ValueError(f"Error initializing feature extractor: {str(e)}")
    
    def _process_subject(self, subject: str) -> Tuple[str, pd.DataFrame]:
        """
        Process a single subject

        Args:
            subject (str): Subject ID to process

        Returns:
            Tuple[str, pd.DataFrame]: Tuple containing subject ID and processed data
        """
        # Extract features
        _, X_subj, image_df, mask_array, mask_info = self._extract_features_for_subject(subject)
        
        # Apply preprocessing if enabled
        if self.feature_config.get('preprocessing', False):
            # 检查是否使用新的多方法串联机制
            if 'methods' in self.feature_config['preprocessing']:
                # 使用多方法串联
                X_subj = preprocess_features(
                    X_subj, 
                    methods=self.feature_config['preprocessing']['methods']
                )
            else:
                # 使用单一方法（向后兼容）
                X_subj = preprocess_features(X_subj, **self.feature_config['preprocessing'])

        
        # Perform supervoxel clustering for each subject individually
        try:
            self.supervoxel_clustering.fit(X_subj)
            supervoxel_labels = self.supervoxel_clustering.predict(X_subj)
            supervoxel_labels += 1  # Start numbering from 1
        except Exception as e:
            print(f"Error: {str(e)}")
            raise ValueError(f"Error performing supervoxel clustering for subject {subject}")
        
        # Get feature names
        if self.feature_config['method'] == 'simple':
            feature_names = self.feature_config['image_names']
        else:
            feature_names = self.feature_extractor.get_feature_names()
        
        # Get original feature names (image names)
        original_feature_names = self.feature_config['image_names']
        
        # Calculate average features for each supervoxel
        unique_labels = np.arange(1, self.n_clusters_supervoxel + 1)
        data_rows = []
        for i in unique_labels:
            indices = supervoxel_labels == i
            if np.any(indices):
                # Calculate average features for current supervoxel
                mean_features = np.mean(X_subj[indices], axis=0)
                mean_original_features = np.mean(image_df.values[indices], axis=0)
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
                    data_row[f"{name}_original"] = mean_original_features[j]
                
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
        del X_subj, image_df, supervoxel_labels
        
        return subject, mean_features_df
    
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
            print("Extracting features and performing supervoxel clustering...")
        
        # Results storage
        mean_features_all = pd.DataFrame()
        
        # Use multiprocessing to process subjects
        if self.n_processes > 1 and len(subjects) > 1:
            if self.verbose:
                print(f"Using {self.n_processes} processes for parallel processing...")
            
            with multiprocessing.Pool(processes=self.n_processes) as pool:
                if self.verbose:
                    print("Starting parallel processing of all subjects...")
                results_iter = pool.imap_unordered(self._process_subject, subjects)
                
                # 使用自定义进度条显示进度
                total = len(subjects)
                for i, (subject, mean_features_df) in enumerate(results_iter):
                    mean_features_all = pd.concat([mean_features_all, mean_features_df], ignore_index=True)
                    print_progress_bar(i, total, prefix=f"Processing subjects:{subject}", suffix="")
                
                if self.verbose:
                    print(f"\nAll {total} subjects have been processed. Proceeding to clustering...")
        else:
            # Single process processing, use custom progress bar
            for i, subject in enumerate(subjects):
                _, mean_features_df = self._process_subject(subject)
                mean_features_all = pd.concat([mean_features_all, mean_features_df], ignore_index=True)
                print_progress_bar(i, len(subjects), prefix="Processing subjects:", suffix="")
        
        # Check if there is enough data for clustering
        if len(mean_features_all) == 0:
            raise ValueError("No valid features for analysis")
        
        # Prepare features for population-level clustering
        if self.feature_config['method'] == 'simple':
            feature_names = self.feature_config['image_names']
        else:
            feature_names = self.feature_extractor.get_feature_names()

        features_for_clustering = mean_features_all[feature_names].values

        #  TODO: if perform group level feature preprocessing, perform here
        
        # Determine optimal number of clusters
        if self.best_n_clusters is not None:
            # If best number of clusters is already specified, use it directly
            optimal_n_clusters = self.best_n_clusters
            scores = None
            if self.verbose:
                print(f"Using specified best number of clusters: {optimal_n_clusters}")
        else:
            # Otherwise find optimal number of clusters
            if self.verbose:
                print("Finding optimal number of clusters...")
            
            try:
                # Ensure cluster number range is reasonable
                min_clusters = max(2, self.n_clusters_habitats_min)
                max_clusters = min(self.n_clusters_habitats_max, len(features_for_clustering) - 1)
                
                if max_clusters <= min_clusters:
                    # If range is invalid, use default minimum value
                    if self.verbose:
                        print(f"Warning: Invalid cluster number range [{min_clusters}, {max_clusters}], using default value")
                    optimal_n_clusters = min_clusters
                    scores = None
                else:
                    # Try to find optimal number of clusters
                    try:
                        cluster_for_best_n = get_clustering_algorithm(self.habitat_method)
                        optimal_n_clusters, scores = cluster_for_best_n.find_optimal_clusters(
                            features_for_clustering, 
                            min_clusters=min_clusters, 
                            max_clusters=max_clusters,
                            methods=self.habitat_cluster_selection_method,
                            show_progress=True
                        )
                        
                        # If plotting is needed, plot score graphs
                        if self.plot_curves and scores is not None:
                            self.habitat_clustering.plot_scores(
                                scores_dict=scores,
                                min_clusters=min_clusters,
                                max_clusters=max_clusters,
                                methods=self.habitat_cluster_selection_method,
                                show=False,
                                save_path=os.path.join(self.out_dir, 'habitat_clustering_scores.png')
                            )
                    except Exception as e:
                        # If optimal number of clusters cannot be found, use default value and warn user
                        if self.verbose:
                            print(f"Warning: Failed to find optimal number of clusters: {str(e)}")
                            print("Using default number of clusters")
                        optimal_n_clusters = min(3, max_clusters)  # Use a reasonable default value
                        scores = None
            except Exception as e:
                # Catch all exceptions, use default value
                if self.verbose:
                    print(f"Error: Exception occurred when determining optimal number of clusters: {str(e)}")
                    print("Using default number of clusters")
                optimal_n_clusters = 3  # Default to 3 clusters
                scores = None
            
            if self.verbose:
                print(f"Optimal number of clusters: {optimal_n_clusters}")
        
        # Perform population-level clustering using optimal number of clusters
        if self.verbose:
            print("Performing population-level clustering...")
        
        
        # Reinitialize clustering algorithm with optimal number of clusters
        self.habitat_clustering.n_clusters = optimal_n_clusters
        self.habitat_clustering.fit(features_for_clustering)
        habitat_labels = self.habitat_clustering.predict(features_for_clustering) + 1  # Start numbering from 1
        
        # Add habitat labels to results
        mean_features_all['Habitats'] = habitat_labels
        
        # Create results DataFrame
        self.results_df = mean_features_all.copy()
        
        # Save results
        if save_results_csv:
            if self.verbose:
                print("Saving results...")
            
            # Create configuration dictionary
            config = {
                'data_dir': self.data_dir,
                'clustering_method': self.supervoxel_method,
                'optimal_n_clusters_habitat': int(optimal_n_clusters),
                'cluster_selection_method': self.habitat_cluster_selection_method,
                'best_n_clusters': self.best_n_clusters,
                'random_state': self.random_state,
                'feature_config': self.feature_config
            }
            
            # Save results
            # Ensure directory exists
            os.makedirs(self.out_dir, exist_ok=True)
            
            # Save configuration
            with open(os.path.join(self.out_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)
            
            # Save CSV
            self.results_df.to_csv(os.path.join(self.out_dir, 'habitats.csv'), index=False)
            
            # Save habitat images for each subject
            # Set Subject as index
            self.results_df.set_index('Subject', inplace=True)
            
            # 使用多进程保存所有主体的栖息地图像
            with multiprocessing.Pool(processes=self.n_processes) as pool:
                # 创建参数列表
                args_list = [(i, subject) for i, subject in enumerate(subjects)]
                
                # 使用imap_unordered并配合自定义进度条
                total = len(subjects)
                for i, _ in enumerate(pool.imap_unordered(self._save_habitat_for_subject, args_list)):
                    print_progress_bar(i, total, prefix="Save habitat images:", suffix="")
        
        return self.results_df
    
    def _save_habitat_for_subject(self, args):
        """
        保存单个主体的栖息地图像

        Args:
            args (tuple): 包含主体索引和名称的元组 (i, subject)

        Returns:
            int: 主体索引
        """
        i, subject = args
        supervoxel_path = os.path.join(self.out_dir, f"{subject}_supervoxel.nrrd")
        save_habitat_image(subject, self.results_df, supervoxel_path, self.out_dir)
        return i
    
    def _extract_features_for_subject(self, subject: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
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
        
        # Get first mask to check its shape
        first_mask_name = next(iter(mask_paths.keys())) # Get name of first mask
        mask = sitk.ReadImage(mask_paths[first_mask_name]) # Read first mask
        mask_array = sitk.GetArrayFromImage(mask) # Convert mask to numpy array
        roi_indices = np.where(mask_array > 0) # Get indices of non-zero values in mask
        
        if len(roi_indices[0]) == 0:
            raise ValueError(f"Mask for subject {subject} has no non-zero values, cannot extract features")
        
        # Optimization: Pre-calculate ROI indices to avoid multiple calculations
        roi_mask = mask_array > 0
        
        # Load image data
        # Optimization: Extract ROI region directly to reduce memory usage
        image_data = {}
        for img_name in self.feature_config['image_names']:
            if img_name not in img_paths:
                raise KeyError(f"Image name '{img_name}' does not exist in subject '{subject}'")
                
            img = sitk.ReadImage(img_paths[img_name])
            img_array = sitk.GetArrayFromImage(img)
            
            # Extract ROI region
            image_data[img_name] = img_array[roi_mask]
            
            # Release memory immediately
            del img_array
            
        # Convert image data to time series format [n_voxels, n_images]
        n_voxels = len(next(iter(image_data.values())))
        
        # Optimization: Build numpy array directly to avoid intermediate conversions
        feature_array = np.zeros((n_voxels, len(self.feature_config['image_names'])))
        for i, img_name in enumerate(self.feature_config['image_names']):
            feature_array[:, i] = image_data[img_name]
            # Release memory immediately
            del image_data[img_name]
        
        # Clean up memory
        del image_data
        
        # Create DataFrame
        image_df = pd.DataFrame(feature_array, columns=self.feature_config['image_names'])
        del feature_array  # Release memory

        # Extract features
        # If feature extractor needs timestamps, prepare relevant parameters
        feature_kwargs = {
            'subject': subject,
        }
        feature_kwargs.update(self.feature_config.get('params', {}))
        
        # Extract features
        features = self.feature_extractor.extract_features(image_df, **feature_kwargs)
        
        # Save mask information for later image reconstruction
        mask_info = {
            'mask': mask,
            'mask_array': mask_array
        }
        
        return subject, features, image_df, mask_array, mask_info 