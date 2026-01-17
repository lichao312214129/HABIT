"""
Feature Manager for Habitat Analysis.
Handles all feature extraction and preprocessing logic.
"""

import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
import os
from typing import Dict, List, Any, Tuple, Optional, Union
from glob import glob

from habit.utils.parallel_utils import parallel_map
from ..config import HabitatConfig, ResultColumns
from ..extractors.feature_expression_parser import FeatureExpressionParser
from ..extractors.feature_extractor_factory import create_feature_extractor
from ..utils.preprocessing_state import preprocess_features

class FeatureManager:
    """
    Manages feature extraction and preprocessing for habitat analysis.
    """
    
    def __init__(self, config: HabitatConfig, logger: logging.Logger):
        """
        Initialize FeatureManager.
        
        Args:
            config: Habitat analysis configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.expression_parser = FeatureExpressionParser()
        
        self._validate_feature_config()
        self._init_feature_extractor()
        
        # Will be set by set_data_paths
        self.images_paths = None
        self.mask_paths = None
        self.supervoxel_file_dict = None
        
        # Log file path for subprocesses
        self._log_file_path = None
        self._log_level = logging.INFO

    def set_data_paths(self, images_paths: Dict, mask_paths: Dict):
        """Set image and mask paths."""
        self.images_paths = images_paths
        self.mask_paths = mask_paths

    def set_logging_info(self, log_file_path: str, log_level: int):
        """Set logging info for subprocesses."""
        self._log_file_path = log_file_path
        self._log_level = log_level

    def _ensure_logging_in_subprocess(self) -> None:
        """
        Ensure logging is properly configured in child processes.
        """
        from habit.utils.log_utils import restore_logging_in_subprocess
        
        if self._log_file_path:
            restore_logging_in_subprocess(self._log_file_path, self._log_level)

    def _validate_feature_config(self) -> None:
        """Validate feature configuration."""
        if 'voxel_level' not in self.config.feature_config:
            raise ValueError("voxel_level configuration is required")
        
        if 'supervoxel_level' in self.config.feature_config and self.config.runtime.verbose:
            self.logger.info(
                "Note: supervoxel_level feature configuration detected."
            )

    def _init_feature_extractor(self) -> None:
        """Initialize feature extractor based on configuration."""
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
        if not self.images_paths or not self.mask_paths:
            raise ValueError("Data paths not set. Call set_data_paths first.")

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
            
            if not self.supervoxel_file_dict:
                raise ValueError("Supervoxel files not set up.")

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

    def _apply_preprocessing(self, feature_df: pd.DataFrame, config_key: str) -> pd.DataFrame:
        """
        Apply preprocessing based on configuration key.
        
        Args:
            feature_df: DataFrame to preprocess
            config_key: Configuration key to look up ('preprocessing_for_subject_level' etc.)
            
        Returns:
            Preprocessed DataFrame
        """
        preprocessing_config = self.config.feature_config.get(config_key, False)
        
        if preprocessing_config and 'methods' in preprocessing_config:
            processed = preprocess_features(
                feature_df.values,
                methods=preprocessing_config['methods']
            )
            return pd.DataFrame(processed, columns=feature_df.columns)
        
        return feature_df

    def apply_preprocessing(
        self, 
        feature_df: pd.DataFrame, 
        level: str,
        mode_handler: Any = None
    ) -> pd.DataFrame:
        """
        Apply preprocessing based on level.
        
        Args:
            feature_df: DataFrame to preprocess
            level: 'subject' for individual level, 'group' for population level
            mode_handler: Mode handler instance (required for group level)
            
        Returns:
            Preprocessed DataFrame
        """
        if level == 'subject':
            return self._apply_preprocessing(feature_df, 'preprocessing_for_subject_level')
            
        elif level == 'group':
            if mode_handler is None:
                raise ValueError("mode_handler is required for group-level preprocessing")
                
            config_key = 'preprocessing_for_group_level'
            preprocessing_config = self.config.feature_config.get(config_key, {})
            methods = preprocessing_config.get('methods', []) if preprocessing_config else []
            
            # Delegate to mode handler which manages PreprocessingState
            return mode_handler.process_features(feature_df, methods)
            
        else:
            raise ValueError(f"Unknown preprocessing level: {level}")

    def calculate_supervoxel_means(
        self,
        subject: str,
        feature_df: pd.DataFrame,
        raw_df: pd.DataFrame,
        supervoxel_labels: np.ndarray,
        n_clusters_supervoxel: int
    ) -> pd.DataFrame:
        """
        Calculate supervoxel-level features (aggregated from voxel features).
        """
        feature_names = feature_df.columns.tolist()
        original_feature_names = raw_df.columns.tolist()
        
        unique_labels = np.arange(1, n_clusters_supervoxel + 1)
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

    def setup_supervoxel_files(
        self, 
        subjects: List[str], 
        failed_subjects: List[str],
        out_folder: str
    ) -> None:
        """Setup dictionary mapping subjects to supervoxel files."""
        supervoxel_keyword = self.config.feature_config['supervoxel_level'].get(
            'supervoxel_file_keyword', '*_supervoxel.nrrd'
        )
        supervoxel_files = glob(
            os.path.join(out_folder, supervoxel_keyword)
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
            # Only raise if we actually need these files (checked later)
            pass

    def clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean feature DataFrame: handle types, inf, nan values."""
        features = features.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        # Replace inf with NaN first
        features = features.replace([np.inf, -np.inf], np.nan)
        # Note: We don't fillna here anymore for group level, as PreprocessingState handles it
        # But for intermediate cleaning it's safer to fill with 0 or mean to avoid crashes before that
        # For robustness, we'll leave NaNs to be handled by PreprocessingState later
        return features

    # Deprecated: Logic moved to PreprocessingState via ModeHandler
    def handle_mean_values(self, features: pd.DataFrame, mode_handler: Any) -> None:
        """
        Deprecated. Mean value handling is now integrated into PreprocessingState.
        Kept for potential backward compatibility if needed, but should not be used in new flow.
        """
        pass
