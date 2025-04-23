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

import time
import logging
import numpy as np
import os
from pathlib import Path
import warnings
import multiprocessing
from functools import partial
import argparse
import sys
import pandas as pd
from typing import Dict, List, Optional, Union

from ....utils.io_utils import get_image_and_mask_paths
from ..utils.progress_utils import CustomTqdm
from .basic_features import BasicFeatureExtractor
from .habitat_radiomics import HabitatRadiomicsExtractor
from .msi_features import MSIFeatureExtractor
from .ith_features import ITHFeatureExtractor
from .feature_utils import FeatureUtils

# Disable warnings
warnings.filterwarnings('ignore')

class HabitatFeatureExtractor:
    """Habitat Feature Extraction Class (Refactored)
    
    This class provides functionality for extracting various features from habitat maps:
    1. Radiomic features of raw images within different habitats
    2. Radiomic features of habitats within the entire ROI
    3. Number of disconnected regions and volume percentage for each habitat
    4. MSI (Mutual Spatial Integrity) features from habitat maps
    5. ITH (Intratumoral Heterogeneity) scores from habitat maps
    """
    
    def __init__(self, 
                params_file_of_non_habitat=None,
                params_file_of_habitat=None,
                raw_img_folder=None, 
                habitats_map_folder=None, 
                out_dir=None,
                n_processes=None,
                habitat_pattern=None,
                voxel_cutoff=10,
                extract_ith=False):
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
            extract_ith: Whether to extract ITH scores
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
        self.extract_ith = extract_ith
        
        # Process number settings
        if n_processes is None:
            self.n_processes = max(1, multiprocessing.cpu_count() // 2)
        else:
            self.n_processes = min(n_processes, multiprocessing.cpu_count()-2)

        # Initialize feature extractors
        self.basic_extractor = BasicFeatureExtractor()
        self.radiomics_extractor = HabitatRadiomicsExtractor()
        self.msi_extractor = MSIFeatureExtractor(voxel_cutoff=voxel_cutoff)
        if self.extract_ith:
            self.ith_extractor = ITHFeatureExtractor(
                params_file=self.params_file_of_non_habitat,
                window_size=3,
                margin_size=3,
                voxel_cutoff=voxel_cutoff
            )
        
        # Setup logging
        self._setup_logging()
        
        # Save settings
        self.save_every_n_files = 5

    def _setup_logging(self):
        """Configure logging settings"""
        # Get timestamp
        data = time.time()
        timeArray = time.localtime(data)
        timestr = time.strftime('%Y_%m_%d %H_%M_%S', timeArray)
        
        # Create output directory
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        log_file = os.path.join(self.out_dir, f'habitat_analysis_{timestr}.log')
        # Configure logging
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        
        logging.error(f"Log file will be saved to: {log_file}")

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

    def process_subject(self, subj, images_paths, habitat_paths, mask_paths=None):
        """Process a single subject for habitat feature extraction"""
        subject_features = {}
        imgs = list(images_paths[subj].keys())

        # Extract basic habitat features
        try:
            non_radiomics_features = self.basic_extractor.get_non_radiomics_features(habitat_paths[subj])
            subject_features['non_radiomics_features'] = non_radiomics_features
        except Exception as e:
            logging.error(f"Error processing basic features for subject {subj}: {str(e)}")
            subject_features['non_radiomics_features'] = {"error": str(e)}

        # Extract traditional radiomics features from original images
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
        try:
            n_habitats = self._get_n_habitats_from_csv()
            msi_features = self.msi_extractor.extract_MSI_features(habitat_paths[subj], n_habitats, subj)
            subject_features['msi_features'] = msi_features
        except Exception as e:
            logging.error(f"Error processing MSI features for subject {subj}: {str(e)}")
            subject_features['msi_features'] = {"error": str(e)}
            
        # Extract ITH features if enabled and masks are available
        if self.extract_ith and mask_paths and subj in mask_paths:
            subject_features['ith_features'] = {}
            for img in imgs:
                try:
                    # Create visualization directory if needed
                    viz_dir = None
                    if self.out_dir:
                        viz_dir = os.path.join(self.out_dir, 'ith_visualizations')
                        
                    # Extract ITH features
                    ith_features = self.ith_extractor.extract_ith_features(
                        images_paths[subj][img],
                        mask_paths[subj][img],
                        out_dir=viz_dir
                    )
                    subject_features['ith_features'][img] = ith_features
                except Exception as e:
                    logging.error(f"Error processing ITH features for subject {subj}, image {img}: {str(e)}")
                    subject_features['ith_features'][img] = {"error": str(e), "ith_score": 0.0}

        return subj, subject_features

    def extract_features(self, images_paths, habitat_paths, mask_paths=None):
        """Extract habitat features for all subjects"""
        features = {}
        subjs = list(set(images_paths.keys()) & set(habitat_paths.keys()))
        
        if not subjs:
            logging.error("No matching subjects found between original images and habitat maps")
            return self
            
        print(f"**************Starting habitat feature extraction for {len(subjs)} subjects using {self.n_processes} processes**************")
        
        temp_dir = os.path.join(self.out_dir, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            process_func = partial(self.process_subject, images_paths=images_paths, 
                                 habitat_paths=habitat_paths, mask_paths=mask_paths)
            
            total = len(subjs)
            progress_bar = CustomTqdm(total=total, desc="Extracting Features")
            for i, (subj, subject_features) in enumerate(pool.imap_unordered(process_func, subjs)):
                features[subj] = subject_features
                progress_bar.update(1)
                
                if (i + 1) % self.save_every_n_files == 0:
                    temp_file = os.path.join(temp_dir, f"features_temp_{i+1}.npy")
                    np.save(temp_file, features)
                    
        for temp_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, temp_file))
        os.rmdir(temp_dir)
            
        out_file = os.path.join(self.out_dir, "habitats_features.npy")
        np.save(out_file, features)
        print(f"Features saved to {out_file}")
        return self

    def parse_features(self, feature_types: List[str], n_habitats: Optional[int] = None):
        """Parse habitat features to CSV files"""
        if not self.out_dir:
            raise ValueError("Output directory must be specified to parse features")
            
        try:
            self.data = np.load(os.path.join(self.out_dir, "habitats_features.npy"), allow_pickle=True).item()
            logging.info(f"Successfully loaded data from habitats_features.npy")
        except Exception as e:
            logging.error(f"Error loading data from habitats_features.npy: {str(e)}")
            raise

        # If n_habitats is not specified, try to read from CSV file
        if n_habitats is None:
            n_habitats = self._get_n_habitats_from_csv()
        
        logging.info(f"Using habitat count: {n_habitats}")
        self.n_habitats = n_habitats
            
        results = {}
        
        if 'traditional' in feature_types:
            results['traditional'] = self._extract_traditional_radiomics()
            
        if 'non_radiomics' in feature_types:
            results['non_radiomics'] = self._extract_non_radiomics_features(n_habitats)
            
        if 'whole_habitat' in feature_types:
            results['whole_habitat'] = self._extract_radiomics_features_for_whole_habitat_map()
            
        if 'each_habitat' in feature_types:
            results['each_habitat'] = self._extract_radiomics_features_from_each_habitat(n_habitats)
            
        if 'msi' in feature_types:
            results['msi'] = self._extract_msi_features()
            
        if 'ith' in feature_types and self.extract_ith:
            results['ith'] = self._extract_ith_features()
            
        return results

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
        n1 = [f"{i}_num_regions" for i in range(1, n_habitats+1)]
        n2 = [f"{i}_volume_ratio" for i in range(1, n_habitats+1)]
        non_radiomics_features = pd.DataFrame(index=subjs, columns=['num_habitats']+n1+n2)
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing Basic Habitat Features")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)
            
            fd = FeatureUtils.flatten_dict(self.data.get(subj).get('non_radiomics_features'))
            for fn in non_radiomics_features.columns:
                try:
                    non_radiomics_features.loc[subj, fn] = fd.get(fn)
                except Exception as e:
                    logging.error(f"Error processing feature {fn} for subject {subj}: {str(e)}")
                    # Set feature to NaN
                    non_radiomics_features.loc[subj, fn] = np.nan

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
        ith_features_list = []
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing ITH Features")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)
            
            try:
                if 'ith_features' in self.data.get(subj):
                    # Process all images
                    imgs = list(self.data.get(subj).get('ith_features').keys())
                    
                    for img in imgs:
                        features = self.data.get(subj).get('ith_features').get(img)
                        if 'error' not in features:
                            features_df = pd.DataFrame.from_dict(features, orient='index').T
                            # Add subject and image info
                            features_df['subject_id'] = subj
                            features_df['image_id'] = img
                            features_df.set_index(['subject_id', 'image_id'], inplace=True)
                            ith_features_list.append(features_df)
                        else:
                            logging.error(f"Error extracting ITH features for subject {subj}, image {img}: {features['error']}")
                            # Create empty DataFrame if there are successful extractions
                            if len(ith_features_list) > 0:
                                empty_df = FeatureUtils.create_empty_dataframe_like(
                                    ith_features_list[0].reset_index().drop(['subject_id', 'image_id'], axis=1),
                                    index=[0]
                                )
                                empty_df['subject_id'] = subj
                                empty_df['image_id'] = img
                                empty_df.set_index(['subject_id', 'image_id'], inplace=True)
                                ith_features_list.append(empty_df)
                else:
                    logging.error(f"No ITH features data for subject {subj}")
            except Exception as e:
                logging.error(f"Error processing ITH features for subject {subj}: {str(e)}")
        
        if len(ith_features_list) > 0:
            ith_features_df = pd.concat(ith_features_list)
            out_file = os.path.join(self.out_dir, "ith_features.csv")
            ith_features_df.to_csv(out_file)
            logging.info(f"ITH features saved to {out_file}")
            
            # Also create a simplified version with just subject_id and ITH scores
            ith_scores_df = ith_features_df.reset_index()[['subject_id', 'image_id', 'ith_score']]
            ith_scores_df.to_csv(os.path.join(self.out_dir, "ith_scores.csv"), index=False)
            logging.info(f"ITH scores saved to {os.path.join(self.out_dir, 'ith_scores.csv')}")
            
            return ith_features_df
        else:
            logging.error("No valid ITH features data")
            return None

    def run(self, feature_types: Optional[List[str]] = None, n_habitats: Optional[int] = None, mode: str = 'both'):
        """Run the complete analysis pipeline"""
        # Feature extraction
        if self.out_dir and (mode == 'extract' or mode == 'both'):
            images_paths, habitat_paths, mask_paths = self.get_mask_and_raw_files()
            self.extract_features(images_paths, habitat_paths, mask_paths)
            
        # Feature parsing
        if feature_types and self.out_dir and (mode == 'parse' or mode == 'both'):
            self.parse_features(feature_types, n_habitats)
            
        return self

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Habitat Feature Extraction Tool')
    
    # Feature extraction parameters
    parser.add_argument('--params_file_of_non_habitat', type=str,
                        help='Parameter file for extracting radiomic features from raw images')
    parser.add_argument('--params_file_of_habitat', type=str,
                        help='Parameter file for extracting radiomic features from habitat images')
    parser.add_argument('--raw_img_folder', type=str,
                        help='Root directory of raw images')
    parser.add_argument('--habitats_map_folder', type=str,
                        help='Root directory of habitat maps')
    parser.add_argument('--out_dir', type=str,
                        help='Output directory')
    parser.add_argument('--n_processes', type=int,
                        help='Number of processes to use')
    parser.add_argument('--habitat_pattern', type=str,
                        choices=['*_habitats.nrrd', '*_habitats_remapped.nrrd'],
                        help='Pattern for matching habitat files')
    parser.add_argument('--voxel_cutoff', type=int, default=10,
                        help='Voxel threshold for filtering small regions in MSI feature calculation')
    parser.add_argument('--extract_ith', action='store_true',
                        help='Extract ITH scores')
    
    # Feature parsing parameters
    parser.add_argument('--feature_types', nargs='+', type=str,
                        choices=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi', 'ith'],
                        help='Types of features to parse')
    parser.add_argument('--n_habitats', type=int,
                        help='Number of habitats to process (if not specified, will be read from habitats.csv)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['extract', 'parse', 'both'],
                        help='Run mode: extract features only (extract), parse features only (parse), or both')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Use default values if no command line arguments are provided
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--params_file_of_non_habitat', 'F:\\work\\research\\radiomics_TLSs\\code_for_habitat_analysis\\parameter.yaml',
            '--params_file_of_habitat', 'F:\\work\\research\\radiomics_TLSs\\code_for_habitat_analysis\\parameter_habitat.yaml',
            '--raw_img_folder', 'H:\\lessThan50AndNoMacrovascularInvasion_structured',
            '--habitats_map_folder', 'F:\\work\\research\\radiomics_TLSs\\data\\results_kmeans_0402_4clusters_test',
            '--out_dir', 'F:\\work\\research\\radiomics_TLSs\\data\\results_kmeans_0402_4clusters_test\\parsed_features',
            '--n_processes', '4',
            '--habitat_pattern', '*_habitats.nrrd',
            '--voxel_cutoff', '10',
            '--extract_ith',
            '--feature_types', 'traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi', 'ith',
            '--mode', 'both'
        ])
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create feature extractor instance and run
    s = time.time()
    extractor = HabitatFeatureExtractor(
        params_file_of_non_habitat=args.params_file_of_non_habitat,
        params_file_of_habitat=args.params_file_of_habitat,
        raw_img_folder=args.raw_img_folder,
        habitats_map_folder=args.habitats_map_folder,
        out_dir=args.out_dir,
        n_processes=args.n_processes,
        habitat_pattern=args.habitat_pattern,
        voxel_cutoff=args.voxel_cutoff,
        extract_ith=args.extract_ith
    )
    extractor.run(feature_types=args.feature_types, n_habitats=args.n_habitats, mode=args.mode)
    e = time.time()
    print(f"Total time: {e - s:.2f} seconds") 

    # Command line form (using real sys parameters):
    # python new_extractor.py --params_file_of_non_habitat parameter.yaml --params_file_of_habitat parameter_habitat.yaml --raw_img_folder G:\lessThan50AndNoMacrovascularInvasion_structured --habitats_map_folder F:\work\research\radiomics_TLSs\data\results_kmeans_0401 --out_dir F:\work\research\radiomics_TLSs\data\results_kmeans_0401\parsed_features --n_processes 10 --habitat_pattern *_habitats.nrrd --feature_types traditional non_radiomics whole_habitat each_habitat msi ith --extract_ith --mode parse 