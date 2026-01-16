#!/usr/bin/env python
"""
Habitat Feature Extraction Tool (Refactored Version)
This tool provides functionality for extracting features from habitat maps:
1. Radiomic features of raw images within different habitats
2. Radiomic features of habitats within the entire ROI
3. Number of disconnected regions and volume percentage for each habitat
4. MSI (Mutual Spatial Integrity) features from habitat maps
5. ITH (Intratumoral Heterogeneity) scores from habitat maps
"""

import logging
import numpy as np
import os
from pathlib import Path
import warnings
import multiprocessing
from functools import partial
import pandas as pd
from typing import Dict, List, Optional, Union

from habit.utils.io_utils import get_image_and_mask_paths
from habit.utils.progress_utils import CustomTqdm
from .basic_features import BasicFeatureExtractor
from .habitat_radiomics import HabitatRadiomicsExtractor
from .msi_features import MSIFeatureExtractor
from .ith_features import ITHFeatureExtractor
from .feature_utils import FeatureUtils
from habit.utils.io_utils import setup_logging

# Disable warnings
warnings.filterwarnings('ignore')

class HabitatFeatureExtractor:
    """Habitat Feature Extraction Class (Refactored)
    
    This class provides functionality for extracting various features from habitat maps:
    1. Radiomic features of raw images within different habitats
    2. Radiomic features of habitats within the entire ROI
    3. Number of disconnected regions and volume percentage for each habitat
    4. MSI (Mutual Spatial Integrity) features from habitat maps
    5. ITH (Intratumoral Heterogeneity) index from habitat maps
    """
    
    def __init__(self, 
                params_file_of_non_habitat=None,
                params_file_of_habitat=None,
                raw_img_folder=None, 
                habitats_map_folder=None, 
                out_dir=None,
                n_processes=None,
                habitat_pattern=None,
                voxel_cutoff=10):
        """
        Initialize the habitat feature extractor
        
        Args:
            params_file_of_non_habitat: Parameter file for extracting radiomic features from raw images
            params_file_of_habitat: Parameter file for extracting radiomic features from habitat images
            raw_img_folder: Root directory of raw images
            habitats_map_folder: Root directory of habitat maps
            out_dir: Output directory
            n_processes: Number of processes to use
            habitat_pattern: Pattern for matching habitat files
            voxel_cutoff: Voxel threshold for filtering small regions in MSI feature calculation
        """
        # Feature extraction related parameters
        self.params_file_of_non_habitat = params_file_of_non_habitat
        self.params_file_of_habitat = params_file_of_habitat
        self.raw_img_folder = raw_img_folder
        self.habitats_map_folder = habitats_map_folder
        self.out_dir = out_dir
        self._habitat_pattern = habitat_pattern
        self.n_habitats = None  # Initialize as None, will be read from file later
        self.voxel_cutoff = voxel_cutoff
        
        # Process number settings
        if n_processes is None:
            self.n_processes = max(1, multiprocessing.cpu_count() // 2)
        else:
            self.n_processes = min(n_processes, multiprocessing.cpu_count()-2)

        # Initialize feature extractors
        self.basic_extractor = BasicFeatureExtractor()
        self.radiomics_extractor = HabitatRadiomicsExtractor()
        self.msi_extractor = MSIFeatureExtractor(voxel_cutoff=voxel_cutoff)
        self.ith_extractor = ITHFeatureExtractor()
        
        # Setup logging
        self._setup_logging()
        
        # Save settings
        self.save_every_n_files = 5

    def _setup_logging(self):
        """Configure logging settings.
        
        If logging has already been configured by the CLI entry point,
        this will simply get the existing logger. Otherwise, it will
        set up a new logger with file output.
        
        Also stores log configuration for child processes (multiprocessing support).
        """
        from habit.utils.log_utils import setup_logger, get_module_logger, LoggerManager
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
        manager = LoggerManager()
        
        # Check if root logger already has handlers (configured by CLI)
        if manager.get_log_file() is not None:
            # Logging already configured by CLI, just get module logger
            self.logger = get_module_logger('habitat.new_extractor')
            self.logger.info("Using existing logging configuration from CLI entry point")
            
            # Store log configuration for child processes (Windows spawn mode)
            self._log_file_path = manager.get_log_file()
            self._log_level = manager._root_logger.getEffectiveLevel() if manager._root_logger else logging.INFO
        else:
            # Logging not configured yet (e.g., direct class usage)
            self.logger = setup_logger(
                name='habitat.new_extractor',
                output_dir=self.out_dir,
                log_filename='feature_extraction.log',
                level=logging.INFO
            )
            
            # Store log configuration for child processes
            self._log_file_path = manager.get_log_file()
            self._log_level = logging.INFO
        
        self.logger.info("Logging setup completed")
    
    def _ensure_logging_in_subprocess(self) -> None:
        """Ensure logging is properly configured in child processes."""
        from habit.utils.log_utils import restore_logging_in_subprocess
        
        if hasattr(self, '_log_file_path') and self._log_file_path:
            restore_logging_in_subprocess(self._log_file_path, self._log_level)

    def _get_n_habitats_from_csv(self):
        """Read the number of habitats from habitats.csv file"""
        n_habitats = FeatureUtils.get_n_habitats_from_csv(self.habitats_map_folder)
        
        if n_habitats is not None:
            return n_habitats
        
        # If unable to read, prompt user for input
        logging.warning("Unable to read number of habitats from habitats.csv, please enter manually")
        while True:
            try:
                user_input = input("Please enter the number of habitats (integer): ")
                n_habitats = int(user_input.strip())
                if n_habitats > 0:
                    logging.info(f"User entered number of habitats: {n_habitats}")
                    return n_habitats
                else:
                    print("Please enter a positive integer")
            except ValueError:
                print("Invalid input, please enter an integer")

    def get_mask_and_raw_files(self):
        """Get paths to all original images and habitat maps"""
        # Use the get_image_and_mask_paths function imported from io_utils
        images_paths, mask_paths = get_image_and_mask_paths(self.raw_img_folder)

        habitat_paths = {}
        for subj in Path(self.habitats_map_folder).glob(self._habitat_pattern):
            key = subj.name.replace(self._habitat_pattern.replace("*", ""), "")
            habitat_paths[key] = str(subj)

        return images_paths, habitat_paths, mask_paths

    def process_subject(self, subj, images_paths, habitat_paths, mask_paths=None, feature_types=None):
        """Process a single subject for habitat feature extraction"""
        # Restore logging configuration in child process (for multiprocessing)
        self._ensure_logging_in_subprocess()
        
        subject_features = {}
        imgs = list(images_paths[subj].keys())
        
        # 如果未指定特征类型，则默认提取所有特征
        if feature_types is None:
            feature_types = ['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi', 'ith_score']

        # Extract basic habitat features
        if 'non_radiomics' in feature_types:
            try:
                non_radiomics_features = self.basic_extractor.get_non_radiomics_features(habitat_paths[subj])
                subject_features['non_radiomics_features'] = non_radiomics_features
            except Exception as e:
                logging.error(f"Error processing basic features for subject {subj}: {str(e)}")
                subject_features['non_radiomics_features'] = {"error": str(e)}

        # Extract traditional radiomics features from original images
        if 'traditional' in feature_types:
            subject_features['tranditional_radiomics_features'] = {}
            for img in imgs:
                try:
                    subject_features['tranditional_radiomics_features'][img] = \
                        self.radiomics_extractor.extract_tranditional_radiomics(
                        images_paths[subj][img], 
                        habitat_paths[subj], 
                        subj,
                        self.params_file_of_non_habitat
                    )
                except Exception as e:
                    logging.error(f"Error processing traditional radiomics features for subject {subj}, image {img}: {str(e)}")
                    subject_features['tranditional_radiomics_features'][img] = {"error": str(e)}

        # Extract radiomics features from the whole habitat map
        if 'whole_habitat' in feature_types:
            try:
                radiomics_features_of_whole_habitat = self.radiomics_extractor.extract_radiomics_features_for_whole_habitat(
                    habitat_paths[subj], 
                    self.params_file_of_habitat
                )
                subject_features['radiomics_features_of_whole_habitat_map'] = radiomics_features_of_whole_habitat
            except Exception as e:
                logging.error(f"Error processing whole habitat radiomics features for subject {subj}: {str(e)}")
                subject_features['radiomics_features_of_whole_habitat_map'] = {"error": str(e)}
        
        # Extract radiomics features from each habitat
        if 'each_habitat' in feature_types:
            subject_features['radiomics_features_from_each_habitat'] = {}
            for img in imgs:
                try:
                    habitat_features = self.radiomics_extractor.extract_radiomics_features_from_each_habitat(
                        habitat_paths[subj], 
                        images_paths[subj][img], 
                        subj, 
                        self.params_file_of_non_habitat
                    )
                    subject_features['radiomics_features_from_each_habitat'][img] = habitat_features
                except Exception as e:
                    logging.error(f"Error processing radiomics features for subject {subj}, habitat {img}: {str(e)}")
                    subject_features['radiomics_features_from_each_habitat'][img] = {"error": str(e)}
                
        # Extract MSI features
        if 'msi' in feature_types:
            try:
                n_habitats = self._get_n_habitats_from_csv()
                msi_features = self.msi_extractor.extract_MSI_features(habitat_paths[subj], n_habitats, subj)
                subject_features['msi_features'] = msi_features
            except Exception as e:
                logging.error(f"Error processing MSI features for subject {subj}: {str(e)}")
                subject_features['msi_features'] = {"error": str(e)}
            
        # Extract ITH features if enabled and masks are available
        if 'ith_score' in feature_types:
            subject_features['ith_features'] = {}
            try:
                # Extract ITH features
                ith_features = self.ith_extractor.extract_ith_features(habitat_paths[subj])
                subject_features['ith_features'] = ith_features
            except Exception as e:
                logging.error(f"Error processing ITH features for subject {subj}: {str(e)}")
                subject_features['ith_features'] = {"error": str(e)}

        return subj, subject_features

    def extract_features(self, images_paths, habitat_paths, mask_paths=None, feature_types=None):
        """Extract habitat features for all subjects"""
        features = {}
        subjs = list(set(images_paths.keys()) & set(habitat_paths.keys()))
        
        if not subjs:
            logging.error("No matching subjects found between original images and habitat maps")
            return features
            
        print(f"**************Starting habitat feature extraction for {len(subjs)} subjects using {self.n_processes} processes**************")
        
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            process_func = partial(self.process_subject, images_paths=images_paths, 
                                 habitat_paths=habitat_paths, mask_paths=mask_paths,
                                 feature_types=feature_types)
            
            total = len(subjs)
            progress_bar = CustomTqdm(total=total, desc="Extracting Features")
            for i, (subj, subject_features) in enumerate(pool.imap_unordered(process_func, subjs)):
                features[subj] = subject_features
                progress_bar.update(1)
                
        return features

    def run(self, feature_types: Optional[List[str]] = None, n_habitats: Optional[int] = None):
        """Run the complete analysis pipeline
        
        Args:
            feature_types: Types of features to extract
            n_habitats: Number of habitats to process (None for auto detection)
        """
        if not self.out_dir:
            raise ValueError("Output directory must be specified to run the analysis")
            
        # 提取特征并直接处理为CSV
        images_paths, habitat_paths, mask_paths = self.get_mask_and_raw_files()
        feature_data = self.extract_features(images_paths, habitat_paths, mask_paths, feature_types)
        
        # Log feature extraction results to screen and log file
        logging.info(f"Feature extraction completed for {len(feature_data)} subjects")
        logging.info(f"Feature data keys: {feature_data.keys()}")

        # 直接生成CSV (如果未指定特征类型则不生成)
        if feature_types:
            # 如果未指定n_habitats则自动检测
            if n_habitats is None:
                n_habitats = self._get_n_habitats_from_csv()
                
            logging.info(f"Using habitat count: {n_habitats}")
            self.n_habitats = n_habitats
            self.data = feature_data
            
            # 根据指定的特征类型生成CSV
            if 'traditional' in feature_types:
                self._extract_traditional_radiomics()
                logging.info(f"Traditional radiomics features saved to {os.path.join(self.out_dir, 'raw_image_radiomics.csv')}")
                
            if 'non_radiomics' in feature_types:
                self._extract_non_radiomics_features(n_habitats)
                logging.info(f"Basic habitat features saved to {os.path.join(self.out_dir, 'habitat_basic_features.csv')}")
            
            if 'whole_habitat' in feature_types:
                self._extract_radiomics_features_for_whole_habitat_map()
                logging.info(f"Whole habitat radiomics features saved to {os.path.join(self.out_dir, 'whole_habitat_radiomics.csv')}")
            
            if 'each_habitat' in feature_types:
                self._extract_radiomics_features_from_each_habitat(n_habitats)
                logging.info(f"Radiomics features from each habitat saved to {os.path.join(self.out_dir, 'habitat_radiomics.csv')}")
            
            if 'msi' in feature_types:
                self._extract_msi_features()
                logging.info(f"MSI features saved to {os.path.join(self.out_dir, 'msi_features.csv')}")
            
            if 'ith_score' in feature_types:
                self._extract_ith_features()
            
        return self

    def _extract_traditional_radiomics(self):
        """Extract traditional radiomics features from original images"""
        logging.info("Starting extraction of traditional radiomics features")
        subjs = list(self.data.keys())
        tranditional_radiomics = []
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing Traditional Radiomics")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)

            try:
                imgs = list(self.data.get(subj).get('tranditional_radiomics_features').keys())
                dfs = pd.DataFrame([self.data.get(subj).get('tranditional_radiomics_features').get(img) for img in imgs], index=imgs)
                dfs = dfs.loc[:, ~dfs.columns.str.contains('diagnostic')]
                new_columns = [f"{col}_of_{idx}" for idx in dfs.index for col in dfs.columns]
                dfs_reshaped = pd.DataFrame([dfs.values.flatten()], columns=new_columns)
                tranditional_radiomics.append(dfs_reshaped)
            except Exception as e:
                logging.error(f"Error processing traditional radiomics features for subject {subj}: {str(e)}")
                # Create an empty DataFrame with columns matching other subjects
                if len(tranditional_radiomics) > 0:
                    empty_df = FeatureUtils.create_empty_dataframe_like(
                        tranditional_radiomics[0], 
                        index=[0]
                    )
                    tranditional_radiomics.append(empty_df)
        
        if len(tranditional_radiomics) > 0:
            tranditional_radiomics = pd.concat(tranditional_radiomics)
            tranditional_radiomics.index = subjs
            
            out_file = os.path.join(self.out_dir, "raw_image_radiomics.csv")
            tranditional_radiomics.to_csv(out_file, index=True)
            logging.info(f"Traditional radiomics features saved to {out_file}")
            return tranditional_radiomics
        else:
            logging.error("Error processing traditional radiomics features, no valid results")
            return None

    def _extract_non_radiomics_features(self, n_habitats: int):
        """Extract basic habitat features (number of disconnected regions and volume ratio)"""
        logging.info("Starting extraction of basic habitat features")
        subjs = list(self.data.keys())
        
        # 定义列名：num_habitats + 每个生境的区域数量和体积比例
        n1 = [f"{i}_num_regions" for i in range(1, n_habitats+1)]
        n2 = [f"{i}_volume_ratio" for i in range(1, n_habitats+1)]
        columns = ['num_habitats'] + n1 + n2
        
        non_radiomics_features = pd.DataFrame(index=subjs, columns=columns)
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing Basic Habitat Features")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)
            
            try:
                # 获取基本特征数据
                features = self.data.get(subj).get('non_radiomics_features')
                
                # 处理num_habitats
                non_radiomics_features.loc[subj, 'num_habitats'] = features.get('num_habitats', 0)
                
                # 处理每个生境的特征
                for habitat_id in range(1, n_habitats+1):
                    habitat_data = features.get(habitat_id, {})
                    
                    # 区域数量
                    num_regions_col = f"{habitat_id}_num_regions"
                    if num_regions_col in columns:
                        non_radiomics_features.loc[subj, num_regions_col] = habitat_data.get('num_regions', 0)
                    
                    # 体积比例
                    volume_ratio_col = f"{habitat_id}_volume_ratio"
                    if volume_ratio_col in columns:
                        non_radiomics_features.loc[subj, volume_ratio_col] = habitat_data.get('volume_ratio', 0.0)
            except Exception as e:
                logging.error(f"Error processing basic habitat features for subject {subj}: {str(e)}")
                # 设置所有特征为NaN
                non_radiomics_features.loc[subj, :] = np.nan

        # 保存到CSV文件
        out_file = os.path.join(self.out_dir, "habitat_basic_features.csv")
        non_radiomics_features.to_csv(out_file, index=True)
        logging.info(f"Basic habitat features saved to {out_file}")
        return non_radiomics_features

    def _extract_radiomics_features_for_whole_habitat_map(self):
        """Extract radiomics features from the whole habitat map"""
        logging.info("Starting extraction of whole habitat radiomics features")
        subjs = list(self.data.keys())
        radiomics_of_whole_habitat = []
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing Whole Habitat Radiomics")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)

            try:
                features = self.data.get(subj).get('radiomics_features_of_whole_habitat_map')
                features_df = pd.DataFrame.from_dict(features, orient='index').T
                features_df = features_df.loc[:, ~features_df.columns.str.contains('diagnostic')]
                radiomics_of_whole_habitat.append(features_df)
            except Exception as e:
                logging.error(f"Error processing whole habitat radiomics features for subject {subj}: {str(e)}")
                # Create an empty DataFrame with columns matching other subjects
                if len(radiomics_of_whole_habitat) > 0:
                    empty_df = FeatureUtils.create_empty_dataframe_like(
                        radiomics_of_whole_habitat[0], 
                        index=[0]
                    )
                    radiomics_of_whole_habitat.append(empty_df)
        
        if len(radiomics_of_whole_habitat) > 0:
            radiomics_of_whole_habitat = pd.concat(radiomics_of_whole_habitat)
            radiomics_of_whole_habitat.index = subjs

            out_file = os.path.join(self.out_dir, "whole_habitat_radiomics.csv")
            radiomics_of_whole_habitat.to_csv(out_file, index=True)
            logging.info(f"Whole habitat radiomics features saved to {out_file}")
            return radiomics_of_whole_habitat
        else:
            logging.error("Error processing whole habitat radiomics features, no valid results")
            return None

    def _extract_radiomics_features_from_each_habitat(self, n_habitats: int):
        """Extract radiomics features from each habitat"""
        logging.info("Starting extraction of radiomics features from each habitat")
        subjs = list(self.data.keys())
        radiomics_of_each_habitat = {i+1: [] for i in range(n_habitats)}
        habitat_count = np.zeros((len(subjs), n_habitats))
        habitat_count = pd.DataFrame(habitat_count, index=subjs, columns=[np.arange(1, n_habitats+1)])

        total = len(radiomics_of_each_habitat)
        
        progress_bar = CustomTqdm(total=total, desc="Processing Habitats")
        for habitat_id in radiomics_of_each_habitat.keys():
            progress_bar.update(1)

            for i, subj in enumerate(subjs):
                try:
                    if i == 0: 
                        imgs = list(self.data.get(subj).get('radiomics_features_from_each_habitat').keys())
                    radiomics_of_habitat = [] 
                    for iimg, img in enumerate(imgs):
                        if (habitat_id == 1) & (iimg == 0): 
                            col = list(self.data.get(subj).get('radiomics_features_from_each_habitat').get(img).keys())
                            habitat_count.loc[subj, col] = 1

                        feature = self.data.get(subj).get('radiomics_features_from_each_habitat').get(img).get(habitat_id)
                        if feature is not None: 
                            df = pd.DataFrame.from_dict(feature, orient='index').T
                            radiomics_of_habitat.append(df)  

                    if len(radiomics_of_habitat) > 0:   
                        radiomics_of_habitat = pd.concat(radiomics_of_habitat)
                        radiomics_of_habitat.index = imgs
                        radiomics_of_habitat = radiomics_of_habitat.loc[:, ~radiomics_of_habitat.columns.str.contains('diagnostic')]
                        new_columns = [f"{col}_of_{idx}" for idx in radiomics_of_habitat.index for col in radiomics_of_habitat.columns]
                        radiomics_of_habitat = pd.DataFrame([radiomics_of_habitat.values.flatten()], columns=new_columns, index=[subj])
                        radiomics_of_each_habitat[habitat_id].append(radiomics_of_habitat)
                except Exception as e:
                    logging.error(f"Error processing radiomics features for subject {subj}, habitat {habitat_id}: {str(e)}")
                    # Create an empty DataFrame if there are successful subjects
                    if len(radiomics_of_each_habitat[habitat_id]) > 0:
                        first_df = radiomics_of_each_habitat[habitat_id][0]
                        empty_df = FeatureUtils.create_empty_dataframe_like(
                            first_df,
                            index=[subj]
                        )
                        radiomics_of_each_habitat[habitat_id].append(empty_df)
            
            if len(radiomics_of_each_habitat[habitat_id]) > 0:
                radiomics_of_each_habitat[habitat_id] = pd.concat(radiomics_of_each_habitat[habitat_id])
                out_file = os.path.join(self.out_dir, f"habitat_{habitat_id}_radiomics.csv")
                radiomics_of_each_habitat[habitat_id].to_csv(out_file, index=True)
                logging.info(f"Radiomics features for habitat {habitat_id} saved to {out_file}")
            else:
                logging.error(f"No valid radiomics features data for habitat {habitat_id}")
        
        habitat_count.columns = [f"has_habitat_{i}" for i in range(1, n_habitats+1)]
        habitat_count.to_csv(os.path.join(self.out_dir, "habitat_count.csv"), index=True)
        logging.info("Habitat count information saved")
        return radiomics_of_each_habitat

    def _extract_msi_features(self):
        """Extract MSI features and save as CSV file"""
        logging.info("Starting extraction of MSI features")
        subjs = list(self.data.keys())
        msi_features_list = []
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing MSI Features")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)
            
            try:
                if 'msi_features' in self.data.get(subj):
                    features = self.data.get(subj).get('msi_features')
                    if 'error' not in features:
                        features_df = pd.DataFrame.from_dict(features, orient='index').T
                        features_df.index = [subj]
                        msi_features_list.append(features_df)
                    else:
                        logging.error(f"Error extracting MSI features for subject {subj}: {features['error']}")
                        # Create empty DataFrame if there are successful subjects
                        if len(msi_features_list) > 0:
                            empty_df = FeatureUtils.create_empty_dataframe_like(
                                msi_features_list[0],
                                index=[subj]
                            )
                            msi_features_list.append(empty_df)
                else:
                    logging.error(f"No MSI features data for subject {subj}")
            except Exception as e:
                logging.error(f"Error processing MSI features for subject {subj}: {str(e)}")
                # Create empty DataFrame if there are successful subjects
                if len(msi_features_list) > 0:
                    empty_df = FeatureUtils.create_empty_dataframe_like(
                        msi_features_list[0],
                        index=[subj]
                    )
                    msi_features_list.append(empty_df)
        
        if len(msi_features_list) > 0:
            msi_features_df = pd.concat(msi_features_list)
            out_file = os.path.join(self.out_dir, "msi_features.csv")
            msi_features_df.to_csv(out_file, index=True)
            logging.info(f"MSI features saved to {out_file}")
            return msi_features_df
        else:
            logging.error("No valid MSI features data")
            return None

    def _extract_ith_features(self):
        """Extract ITH features and save as CSV file"""
        logging.info("Starting extraction of ITH features")
        subjs = list(self.data.keys())
        ith_features_list = {}
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing ITH Features")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)
            
            try:
                if 'ith_features' in self.data.get(subj):
                    # Process all images
                    value = self.data.get(subj).get('ith_features')
                    ith_features_list[subj] = value
            except Exception as e:
                logging.error(f"Error processing ITH features for subject {subj}: {str(e)}")
        
        if len(ith_features_list) > 0:
            ith_features_df = pd.DataFrame.from_dict(ith_features_list, orient='index')
            out_file = os.path.join(self.out_dir, "ith_scores.csv")
            ith_features_df.to_csv(out_file)
            logging.info(f"ITH features saved to {out_file}")
            
            
            return ith_features_df
        else:
            logging.error("No valid ITH features data")
            return None
