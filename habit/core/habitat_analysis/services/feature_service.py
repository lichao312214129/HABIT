"""
Feature Service for Habitat Analysis.
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
from ..config_schemas import HabitatAnalysisConfig, DROPPING_PREPROCESSING_METHODS
from ..clustering_features.feature_expression_parser import FeatureExpressionParser
from ..clustering_features.feature_extractor_factory import create_feature_extractor
from ..feature_preprocessing import apply_variance_filter, apply_correlation_filter
from ..utils.preprocessing_state import process_features_pipeline

class FeatureService:
    """
    Orchestrates feature extraction and preprocessing for habitat analysis.

    Pipeline steps call into this class instead of touching extractors and
    preprocessing utilities directly.

    The service has two distinct construction modes:

    * ``train`` — full initialisation: validates ``FeatureConstruction``,
      parses voxel/supervoxel expressions, builds extractors. Use
      :meth:`for_train` (or pass ``config.run_mode == 'train'``).
    * ``predict`` — minimal initialisation: feature extraction state lives
      inside the loaded pipeline pkl, so we only set safe defaults to keep
      attribute access from blowing up. Use :meth:`for_predict` (or pass
      ``config.run_mode == 'predict'``).

    Both factories are equivalent to the existing ``__init__`` and are
    provided to make caller intent explicit; ``HabitatConfigurator`` keeps
    using ``__init__`` so existing call sites are unaffected.
    """

    def __init__(self, config: HabitatAnalysisConfig, logger: logging.Logger):
        """
        Initialize FeatureService and dispatch on ``config.run_mode``.

        Args:
            config: Habitat analysis configuration.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        self.expression_parser = FeatureExpressionParser()

        # Per-run data paths (filled by set_data_paths). Declared here so
        # both train and predict instances expose the same attribute set.
        self.images_paths: Optional[Dict[str, Any]] = None
        self.mask_paths: Optional[Dict[str, Any]] = None
        self.supervoxel_file_dict: Optional[Dict[str, str]] = None

        # Log file path for subprocesses
        self._log_file_path: Optional[str] = None
        self._log_level: int = logging.INFO

        if config.run_mode == 'train':
            self._init_train_mode()
        else:
            self._init_predict_mode()

    @classmethod
    def for_train(
        cls,
        config: HabitatAnalysisConfig,
        logger: logging.Logger,
    ) -> 'FeatureService':
        """Explicit factory for the training run path. See class docstring."""
        if config.run_mode != 'train':
            raise ValueError(
                f"FeatureService.for_train requires config.run_mode == 'train', "
                f"got '{config.run_mode}'."
            )
        return cls(config, logger)

    @classmethod
    def for_predict(
        cls,
        config: HabitatAnalysisConfig,
        logger: logging.Logger,
    ) -> 'FeatureService':
        """Explicit factory for the prediction run path. See class docstring."""
        if config.run_mode == 'train':
            raise ValueError(
                "FeatureService.for_predict requires config.run_mode != 'train' "
                "(got 'train')."
            )
        return cls(config, logger)

    def _init_train_mode(self) -> None:
        """Full init: validate config and build feature extractors."""
        self._validate_FeatureConstruction()
        self._init_feature_extractor()

    def _init_predict_mode(self) -> None:
        """
        Minimal init for the predict path. The real feature-extraction
        state will be restored from the loaded pipeline; we only seed safe
        defaults so attribute access elsewhere does not blow up before that
        injection happens.
        """
        self.voxel_method = None
        self.voxel_params: Dict[str, Any] = {}
        self.voxel_processing_steps: List[Any] = []
        self.has_supervoxel_config = False

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

    def _validate_FeatureConstruction(self) -> None:
        """Validate feature configuration."""
        if not self.config.FeatureConstruction or not self.config.FeatureConstruction.voxel_level:
            raise ValueError("voxel_level configuration is required")
        
        if self.config.FeatureConstruction.supervoxel_level and self.config.verbose:
            self.logger.info(
                "Note: supervoxel_level feature configuration detected."
            )

    def _init_feature_extractor(self) -> None:
        """Initialize feature extractor based on configuration."""
        voxel_config = {
            "method": self.config.FeatureConstruction.voxel_level.method,
            "params": self.config.FeatureConstruction.voxel_level.params
        }
        
        # Parse voxel_level expression
        (self.voxel_method, 
         self.voxel_params, 
         self.voxel_processing_steps) = self.expression_parser.parse(voxel_config)
        
        # Check for supervoxel_level configuration
        self.has_supervoxel_config = self.config.FeatureConstruction.supervoxel_level is not None
        if self.has_supervoxel_config:
            supervoxel_config = {
                "method": self.config.FeatureConstruction.supervoxel_level.method,
                "params": self.config.FeatureConstruction.supervoxel_level.params
            }
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
            voxel_params = self.config.FeatureConstruction.voxel_level.params
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
                voxel_params = self.config.FeatureConstruction.voxel_level.params
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

    def _get_preprocessing_methods(
        self,
        preprocessing_config: Optional[Any]
    ) -> List[Any]:
        """
        Get preprocessing methods from config, returning PreprocessingMethod objects directly.
        
        Args:
            preprocessing_config: PreprocessingConfig object or None
            
        Returns:
            List of PreprocessingMethod objects
        """
        if not preprocessing_config or not hasattr(preprocessing_config, 'methods'):
            return []
        return list(preprocessing_config.methods)

    def _apply_preprocessing(self, feature_df: pd.DataFrame, config_key: str) -> pd.DataFrame:
        """
        Apply preprocessing based on configuration key.
        
        ## Purpose of Feature Preprocessing
        
        Feature preprocessing in habitat analysis serves to eliminate noise from technical 
        factors (e.g., scanner variability, acquisition protocol differences) while preserving 
        biologically meaningful tissue heterogeneity.
        
        **Subject-level preprocessing (Individual-level)**:
        - **Goal**: Eliminate within-subject outliers and scale differences
        - **Methods**: Winsorization (remove extreme outliers), Min-Max normalization (0-1 scaling)
        - **Purpose**: Ensure each subject's features are on comparable scales before pooling 
          across subjects. This prevents subjects with extreme intensity values from dominating 
          the clustering.
        - **Example**: If one subject has MRI intensities ranging [0, 1000] and another [0, 100], 
          normalization ensures both contribute equally to population-level clustering.
        
        **Group-level preprocessing (Population-level)**:
        - **Goal**: Reduce micro-noise and discretize features to capture stable patterns
        - **Methods**: Binning/Discretization (e.g., uniform bins, quantile bins)
        - **Purpose**: Transform continuous features into discrete bins, making clustering more 
          robust to small fluctuations. This helps identify stable biological patterns like 
          "high perfusion" vs "low perfusion" rather than overfitting to exact intensity values.
        - **Example**: Instead of clustering on exact ADC values (e.g., 800.1, 801.3, 799.8), 
          bin them into "low ADC" (0-600), "medium ADC" (600-1200), "high ADC" (1200+).
        
        Args:
            feature_df: DataFrame to preprocess
            config_key: Configuration key to look up ('preprocessing_for_subject_level' or 
                       'preprocessing_for_group_level')
            
        Returns:
            Preprocessed DataFrame
        """
        if config_key == 'preprocessing_for_subject_level':
            preprocessing_config = self.config.FeatureConstruction.preprocessing_for_subject_level
        else:
            preprocessing_config = self.config.FeatureConstruction.preprocessing_for_group_level

        methods = self._get_preprocessing_methods(preprocessing_config)
        if methods:
            # Guardrail: in two-step mode, subject-level feature-dropping can create
            # inconsistent columns across subjects before group concatenation.
            # Same constant set as the pydantic-level validator in
            # config_schemas.HabitatAnalysisConfig.validate_mode_dependent_fields.
            if (
                config_key == 'preprocessing_for_subject_level'
                and self.config.HabitatsSegmention.clustering_mode == 'two_step'
            ):
                dropping_methods = {
                    method.method
                    for method in methods
                    if method.method in DROPPING_PREPROCESSING_METHODS
                }
                if dropping_methods:
                    methods_text = ", ".join(sorted(dropping_methods))
                    raise ValueError(
                        "Subject-level feature-dropping methods are not allowed in two_step mode: "
                        f"{methods_text}. Please move them to preprocessing_for_group_level."
                    )

            processed_df = feature_df.copy()

            for method in methods:
                method_name = method.method

                # Column-dropping preprocessors live in feature_preprocessing/
                # because they cannot be expressed through the value-only
                # process_features_pipeline path used for the else branch.
                if method_name == 'variance_filter':
                    threshold = (
                        float(method.variance_threshold)
                        if method.variance_threshold is not None
                        else 0.0
                    )
                    processed_df = apply_variance_filter(processed_df, threshold)

                elif method_name == 'correlation_filter':
                    threshold = (
                        float(method.corr_threshold)
                        if method.corr_threshold is not None
                        else 0.95
                    )
                    corr_method = method.corr_method or 'spearman'
                    processed_df = apply_correlation_filter(
                        processed_df, threshold, corr_method
                    )

                else:
                    # Stateless value-transform methods (scaling, binning, ...).
                    transformed = process_features_pipeline(processed_df.values, methods=[method])
                    processed_df = pd.DataFrame(
                        transformed,
                        columns=processed_df.columns,
                        index=processed_df.index
                    )

            return processed_df
        
        return feature_df

    def apply_preprocessing(
        self, 
        feature_df: pd.DataFrame, 
        level: str
    ) -> pd.DataFrame:
        """
        Apply preprocessing based on level (user-facing interface).
        
        This method provides a simplified interface for applying preprocessing at different levels.
        
        Args:
            feature_df: DataFrame to preprocess
            level: 'subject' for individual level, 'group' for population level
            
        Returns:
            Preprocessed DataFrame
            
        Note:
            Group-level preprocessing is typically handled by Pipeline steps automatically.
            This method is primarily used for subject-level preprocessing.
        """
        if level == 'subject':
            return self._apply_preprocessing(feature_df, 'preprocessing_for_subject_level')

        raise ValueError(
            f"Unsupported preprocessing level: {level}. "
            "Group-level preprocessing is handled by Pipeline steps."
        )

    def setup_supervoxel_files(
        self,
        subjects: List[str],
        failed_subjects: List[str],
        out_folder: str
    ) -> None:
        """
        Map each subject to its supervoxel-label NRRD file produced by the
        upstream clustering step.

        Matching uses ``_``-delimited tokens of the file basename so that
        subject ids like ``A1`` cannot accidentally match files for ``A11``
        (the original substring match was ambiguous when one subject id is a
        prefix of another).
        """
        supervoxel_keyword = self.config.FeatureConstruction.supervoxel_level.supervoxel_file_keyword
        supervoxel_files = glob(
            os.path.join(out_folder, supervoxel_keyword)
        )

        self.supervoxel_file_dict = {}
        for subject in subjects:
            matched = self._find_supervoxel_file_for_subject(subject, supervoxel_files)
            if matched is not None:
                self.supervoxel_file_dict[subject] = matched
            elif subject not in failed_subjects and self.config.verbose:
                self.logger.warning(f"No supervoxel file found for subject {subject}")

    @staticmethod
    def _find_supervoxel_file_for_subject(
        subject: str,
        candidate_files: List[str]
    ) -> Optional[str]:
        """
        Pick the supervoxel file whose basename contains ``subject`` as a
        complete ``_``-delimited token (extension stripped first).

        Examples (subject="A1"):
            "A1_supervoxel.nrrd"             -> match
            "A11_supervoxel.nrrd"            -> NO match (token is "A11", not "A1")
            "scan_A1_supervoxel.nrrd"        -> match (subject is one of the tokens)
        """
        for path in candidate_files:
            stem = os.path.splitext(os.path.basename(path))[0]
            if subject in stem.split('_'):
                return path
        return None

    def clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean feature DataFrame: handle types, inf, nan values."""
        features = features.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        # Replace inf with NaN first
        features = features.replace([np.inf, -np.inf], np.nan)
        # Note: We don't fillna here anymore for group level, as PreprocessingState handles it
        # But for intermediate cleaning it's safer to fill with 0 or mean to avoid crashes before that
        # For robustness, we'll leave NaNs to be handled by PreprocessingState later
        return features

