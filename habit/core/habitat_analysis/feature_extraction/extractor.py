#!/usr/bin/env python
"""
Habitat Feature Extraction Tool
This tool provides functionality for extracting features from habitat maps:
1. Radiomic features of raw images within different habitats
2. Radiomic features of habitats within the entire ROI
3. Number of disconnected regions and volume percentage for each habitat
4. MSI (Mutual Spatial Integrity) features from habitat maps

The data structure is as follows:
dataset/
├── images/
│   ├── subj001/
│   │   ├── img1
|   |   |   ├── img.nii.gz (OR img.nrrd)
│   │   ├── img2
|   |   |   ├── img.nii.gz (OR img.nrrd)
│   ├── subj002/
│   │   ├── img1
|   |   |   ├── img.nii.gz (OR img.nrrd)
│   │   ├── img2
|   |   |   ├── img.nii.gz (OR img.nrrd)
├── masks/
│   ├── subj001/
│   │   ├── img1
|   |   |   ├── mask.nii.gz (OR mask.nrrd)
│   │   ├── img2
|   |   |   ├── mask.nii.gz (OR mask.nrrd)
│   ├── subj002/
│   │   ├── img1
|   |   |   ├── mask.nii.gz (OR mask.nrrd)
│   │   ├── img2
|   |   |   ├── mask.nii.gz (OR mask.nrrd)

habitats_map/
├──subj001_habitats.nrrd
├──subj002_habitats.nrrd
"""

import time
import trimesh
import scipy
import logging
import numpy as np
import os
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import warnings
import multiprocessing
from functools import partial
import argparse
import sys
import pandas as pd
from typing import Dict, List, Optional, Union

from habit.utils.io_utils import get_image_and_mask_paths
from habit.utils.progress_utils import CustomTqdm

# 禁用警告
warnings.filterwarnings('ignore')

class HabitatFeatureExtractor:
    """Habitat Feature Extraction Class
    
    This class provides functionality for extracting various features from habitat maps:
    1. Radiomic features of raw images within different habitats
    2. Radiomic features of habitats within the entire ROI
    3. Number of disconnected regions and volume percentage for each habitat
    4. MSI (Mutual Spatial Integrity) features from habitat maps
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
        try:
            csv_path = os.path.join(self.habitats_map_folder, 'habitats.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'Habitats' in df.columns:
                    # Assume the Habitats column contains habitat labels, we count the number of unique values
                    unique_habitats = df['Habitats'].nunique()
                    logging.info(f"Read {unique_habitats} habitats from habitats.csv")
                    return unique_habitats
                else:
                    logging.error("Habitats column not found in habitats.csv")
            else:
                logging.error(f"habitats.csv file not found: {csv_path}")
        except Exception as e:
            logging.error(f"Error reading habitats.csv: {str(e)}")
        
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

    @staticmethod
    def get_non_radiomics_features(habitat_img):
        """计算每个habitat的不连通区域数及体积占比"""
        try:
            if isinstance(habitat_img, str):
                habitat_img = sitk.ReadImage(habitat_img)
            elif not isinstance(habitat_img, sitk.Image):
                raise ValueError("habitat_img must be a SimpleITK image or a file path.")

            results = {}
            
            # 计算整个habitat map的总体积
            stats_filter = sitk.StatisticsImageFilter()
            stats_filter.Execute(habitat_img != 0)
            total_voxels = int(stats_filter.GetSum())

            label_filter = sitk.LabelStatisticsImageFilter()
            label_filter.Execute(habitat_img, habitat_img)
            labels = label_filter.GetLabels()
            labels = [int(label) for label in labels if label != 0]
            
            for label in labels:
                try:
                    binary_img = sitk.BinaryThreshold(habitat_img, lowerThreshold=label, upperThreshold=label)
                    
                    stats_filter.Execute(binary_img)
                    habitat_voxels = int(stats_filter.GetSum())
                    volume_ratio = habitat_voxels / total_voxels if total_voxels > 0 else 0.0
                    
                    cc_filter = sitk.ConnectedComponentImageFilter()
                    cc_filter.SetFullyConnected(False)
                    labeled_img = cc_filter.Execute(binary_img)
                    num_regions = cc_filter.GetObjectCount()
                    
                    results[label] = {
                        'num_regions': num_regions,
                        'volume_ratio': volume_ratio
                    }
                except Exception as e:
                    logging.error(f"处理生境标签 {label} 的不连通区域数及体积占比时出错: {str(e)}")
                    results[label] = {
                        'num_regions': 0,
                        'volume_ratio': 0.0,
                        'error': str(e)
                    }
                    
            results['num_habitats'] = len(labels)
            
            return results
        except Exception as e:
            logging.error(f"计算habitat的不连通区域数及体积占比时出错: {str(e)}")
            return {"error": str(e), "num_habitats": 0}

    @staticmethod
    def extract_tranditional_radiomics(image_path, habitat_path, subject_id, params_file):
        """提取常规组学特征"""
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
            
            habitat_img = sitk.ReadImage(habitat_path)
            raw_img = sitk.ReadImage(image_path)

            if raw_img.GetDirection() != habitat_img.GetDirection():
                logging.info(f"raw and mask direction is different: {subject_id}")
                habitat_img.SetDirection(raw_img.GetDirection())

            if raw_img.GetOrigin() != habitat_img.GetOrigin():
                logging.info(f"raw and mask origin is different: {subject_id}")
                habitat_img.SetOrigin(raw_img.GetOrigin())

            if raw_img.GetSpacing() != habitat_img.GetSpacing():
                logging.info(f"raw and mask spacing is different: {subject_id}")
                habitat_img.SetSpacing(raw_img.GetSpacing())

            label_filter = sitk.LabelStatisticsImageFilter()
            label_filter.Execute(habitat_img, habitat_img)
            labels = label_filter.GetLabels()
            labels = [int(label) for label in labels if label != 0]

            mask = sitk.BinaryThreshold(
                habitat_img, 
                lowerThreshold=1, 
                upperThreshold=np.max(labels).astype(np.double), 
                insideValue=1, 
                outsideValue=0
            )

            return extractor.execute(
                imageFilepath=raw_img,
                maskFilepath=mask, 
                label=1
            )
        except Exception as e:
            logging.error(f"提取常规组学特征时出错: {str(e)}")
            return {"error": f"特征提取错误: {str(e)}"}

    @staticmethod
    def extract_radiomics_features_for_whole_habitat(habitat_img, params_file):
        """提取整个ROI内的habitat的组学特征"""
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
            
            if isinstance(habitat_img, str):
                habitat_img = sitk.ReadImage(habitat_img)
            elif not isinstance(habitat_img, sitk.Image):
                raise ValueError("habitat_img must be a SimpleITK image or a file path.")

            label_filter = sitk.LabelStatisticsImageFilter()
            label_filter.Execute(habitat_img, habitat_img)
            labels = label_filter.GetLabels()
            labels = [int(label) for label in labels if label != 0]

            # make a binary image
            habitat_img_binary = sitk.BinaryThreshold(
                habitat_img, 
                lowerThreshold=1, 
                upperThreshold=np.max(labels).astype(np.double), 
                insideValue=1, 
                outsideValue=0
            )

            return extractor.execute(
                imageFilepath=habitat_img,
                maskFilepath=habitat_img_binary, 
                label=1
            )
        except Exception as e:
            logging.error(f"提取整个ROI内的habitat组学特征时出错: {str(e)}")
            return {"error": f"特征提取错误: {str(e)}"}

    @staticmethod
    def extract_radiomics_features_from_each_habitat(habitat_path, image_path, subject_id, params_file):
        """提取每个habitat内原始影像的组学特征"""
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        
        try:
            habitat_img = sitk.ReadImage(habitat_path)
            raw_img = sitk.ReadImage(image_path)

            if raw_img.GetDirection() != habitat_img.GetDirection():
                logging.info(f"raw and mask direction is different: {subject_id}")
                habitat_img.SetDirection(raw_img.GetDirection())

            if raw_img.GetOrigin() != habitat_img.GetOrigin():
                logging.info(f"raw and mask origin is different: {subject_id}")
                habitat_img.SetOrigin(raw_img.GetOrigin())

            if raw_img.GetSpacing() != habitat_img.GetSpacing():
                logging.info(f"raw and mask spacing is different: {subject_id}")
                habitat_img.SetSpacing(raw_img.GetSpacing())

            label = sitk.LabelStatisticsImageFilter()
            label.Execute(habitat_img, habitat_img)
            labels = label.GetLabels()
            labels = [int(label) for label in labels if label != 0]
        except Exception as e:
            logging.error(f"准备生境图像数据时出错: {str(e)}")
            return {}

        featureVector = {}
        for label in labels:
            try:
                featureVector[label] = extractor.execute(
                    imageFilepath=raw_img, 
                    maskFilepath=habitat_img, 
                    label=label
                )
            except Exception as e:
                logging.error(f"提取生境 {label} 的组学特征时出错: {str(e)}")
                featureVector[label] = {}
               
        return featureVector

    def get_mask_and_raw_files(self):
        """获取所有原始影像和生境图文件路径"""
        # 使用从io_utils导入的get_image_and_mask_paths函数
        images_paths, _ = get_image_and_mask_paths(self.raw_img_folder)

        habitat_paths = {}
        for subj in Path(self.habitats_map_folder).glob(self._habitat_pattern):
            key = subj.name.replace(self._habitat_pattern.replace("*", ""), "")
            habitat_paths[key] = str(subj)

        return images_paths, habitat_paths

    def process_subject(self, subj, images_paths, habitat_paths):
        """处理单个受试者的生境特征提取"""
        subject_features = {}
        imgs = list(images_paths[subj].keys())

        # 提取生境基本特征
        try:
            non_radiomics_features = self.get_non_radiomics_features(habitat_paths[subj])
            subject_features['non_radiomics_features'] = non_radiomics_features
        except Exception as e:
            logging.error(f"处理受试者 {subj} 的生境基本特征时出错: {str(e)}")
            subject_features['non_radiomics_features'] = {"error": str(e)}

        # 提取原始影像的常规组学特征
        subject_features['tranditional_radiomics_features'] = {}
        for img in imgs:
            try:
                subject_features['tranditional_radiomics_features'][img] = \
                    self.extract_tranditional_radiomics(
                    images_paths[subj][img], 
                    habitat_paths[subj], 
                    subj,
                    self.params_file_of_non_habitat
                )
            except Exception as e:
                logging.error(f"处理受试者 {subj} 的常规组学特征 {img} 时出错: {str(e)}")
                subject_features['tranditional_radiomics_features'][img] = {"error": str(e)}

        # 提取整个生境图的组学特征
        try:
            radiomics_features_of_whole_habitat = self.extract_radiomics_features_for_whole_habitat(
                habitat_paths[subj], 
                self.params_file_of_habitat
            )
            subject_features['radiomics_features_of_whole_habitat_map'] = radiomics_features_of_whole_habitat
        except Exception as e:
            logging.error(f"处理受试者 {subj} 的整个生境图组学特征时出错: {str(e)}")
            subject_features['radiomics_features_of_whole_habitat_map'] = {"error": str(e)}
        
        # 提取每个生境的组学特征
        subject_features['radiomics_features_from_each_habitat'] = {}
        for img in imgs:
            try:
                habitat_features = self.extract_radiomics_features_from_each_habitat(
                    habitat_paths[subj], 
                    images_paths[subj][img], 
                    subj, 
                    self.params_file_of_non_habitat
                )
                subject_features['radiomics_features_from_each_habitat'][img] = habitat_features
            except Exception as e:
                logging.error(f"处理受试者 {subj} 的生境 {img} 组学特征时出错: {str(e)}")
                subject_features['radiomics_features_from_each_habitat'][img] = {"error": str(e)}
                
        # 提取MSI特征
        try:
            n_habitats = self._get_n_habitats_from_csv()
            msi_features = self.extract_MSI_features(habitat_paths[subj], n_habitats, subj)
            subject_features['msi_features'] = msi_features
        except Exception as e:
            logging.error(f"处理受试者 {subj} 的MSI特征时出错: {str(e)}")
            subject_features['msi_features'] = {"error": str(e)}

        return subj, subject_features

    def extract_features(self, images_paths, habitat_paths):
        """提取所有受试者的生境特征"""
        features = {}
        subjs = list(set(images_paths.keys()) & set(habitat_paths.keys()))
        
        if not subjs:
            logging.error("未在原始影像和生境图之间找到匹配的受试者")
            return self
            
        print(f"**************开始为 {len(subjs)} 个受试者提取生境特征,使用 {self.n_processes} 个进程**************")
        
        temp_dir = os.path.join(self.out_dir, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            process_func = partial(self.process_subject, images_paths=images_paths, habitat_paths=habitat_paths)
            
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
        print(f"特征已保存到 {out_file}")
        return self

    def parse_features(self, feature_types: List[str], n_habitats: Optional[int] = None):
        """解析生境特征到CSV文件"""
        if not self.out_dir:
            raise ValueError("必须指定输出目录以解析特征")
            
        try:
            self.data = np.load(os.path.join(self.out_dir, "habitats_features.npy"), allow_pickle=True).item()
            logging.info(f"成功从 habitats_features.npy 加载数据")
        except Exception as e:
            logging.error(f"从 habitats_features.npy 加载数据时出错: {str(e)}")
            raise

        # 如果没有指定n_habitats，尝试从CSV文件读取
        if n_habitats is None:
            n_habitats = self._get_n_habitats_from_csv()
        
        logging.info(f"使用生境数量: {n_habitats}")
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
            
        return results

    def _extract_traditional_radiomics(self):
        """提取原始影像的常规组学特征"""
        logging.info("开始提取原始影像的常规组学特征")
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
                logging.error(f"处理受试者 {subj} 的传统组学特征时出错: {str(e)}")
                # 创建一个空的DataFrame，列名与其他受试者一致
                if len(tranditional_radiomics) > 0:
                    # 直接创建与其他样本相同形状的DataFrame，填充NaN值
                    empty_df = pd.DataFrame(
                        data=np.nan, 
                        index=[0], 
                        columns=tranditional_radiomics[0].columns
                    )
                    tranditional_radiomics.append(empty_df)
        
        if len(tranditional_radiomics) > 0:
            tranditional_radiomics = pd.concat(tranditional_radiomics)
            tranditional_radiomics.index = subjs
            
            out_file = os.path.join(self.out_dir, "raw_image_radiomics.csv")
            tranditional_radiomics.to_csv(out_file, index=True)
            logging.info(f"已保存原始影像组学特征到 {out_file}")
            return tranditional_radiomics
        else:
            logging.error("处理传统组学特征时出现错误，无法生成有效结果")
            return None

    def _extract_non_radiomics_features(self, n_habitats: int):
        """提取生境的基本特征(不连通区域数和体积占比)"""
        logging.info("开始提取生境基本特征")
        subjs = list(self.data.keys())
        n1 = [f"{i}_num_regions" for i in range(1, n_habitats+1)]
        n2 = [f"{i}_volume_ratio" for i in range(1, n_habitats+1)]
        non_radiomics_features = pd.DataFrame(index=subjs, columns=['num_habitats']+n1+n2)
        total = len(subjs)
        
        progress_bar = CustomTqdm(total=total, desc="Processing Basic Habitat Features")
        for i, subj in enumerate(subjs):
            progress_bar.update(1)
            
            fd = self._flatten_dict(self.data.get(subj).get('non_radiomics_features'))
            for fn in non_radiomics_features.columns:
                try:
                    non_radiomics_features.loc[subj, fn] = fd.get(fn)
                except Exception as e:
                    logging.error(f"处理受试者 {subj} 的特征 {fn} 时出错: {str(e)}")
                    # 将该特征设为NaN
                    non_radiomics_features.loc[subj, fn] = np.nan

        out_file = os.path.join(self.out_dir, "habitat_basic_features.csv")
        non_radiomics_features.to_csv(out_file, index=True)
        logging.info(f"已保存生境基本特征到 {out_file}")
        return non_radiomics_features

    def _extract_radiomics_features_for_whole_habitat_map(self):
        """提取整个生境图的组学特征"""
        logging.info("开始提取整个生境图的组学特征")
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
                logging.error(f"处理受试者 {subj} 的整个生境图组学特征时出错: {str(e)}")
                # 创建一个空的DataFrame，列名与其他受试者一致
                if len(radiomics_of_whole_habitat) > 0:
                    empty_df = pd.DataFrame(
                        data=np.nan, 
                        index=[0], 
                        columns=radiomics_of_whole_habitat[0].columns
                    )
                    radiomics_of_whole_habitat.append(empty_df)
        
        if len(radiomics_of_whole_habitat) > 0:
            radiomics_of_whole_habitat = pd.concat(radiomics_of_whole_habitat)
            radiomics_of_whole_habitat.index = subjs

            out_file = os.path.join(self.out_dir, "whole_habitat_radiomics.csv")
            radiomics_of_whole_habitat.to_csv(out_file, index=True)
            logging.info(f"已保存整个生境图组学特征到 {out_file}")
            return radiomics_of_whole_habitat
        else:
            logging.error("处理整个生境图组学特征时出现错误，无法生成有效结果")
            return None

    def _extract_radiomics_features_from_each_habitat(self, n_habitats: int):
        """提取每个生境的组学特征"""
        logging.info("开始提取各个生境的组学特征")
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
                    logging.error(f"处理受试者 {subj} 的生境 {habitat_id} 组学特征时出错: {str(e)}")
                    # 如果已经有其他受试者成功处理，则创建一个空DataFrame
                    if len(radiomics_of_each_habitat[habitat_id]) > 0:
                        first_df = radiomics_of_each_habitat[habitat_id][0]
                        empty_df = pd.DataFrame(
                            data=np.nan, 
                            index=[subj], 
                            columns=first_df.columns
                        )
                        radiomics_of_each_habitat[habitat_id].append(empty_df)
            
            if len(radiomics_of_each_habitat[habitat_id]) > 0:
                radiomics_of_each_habitat[habitat_id] = pd.concat(radiomics_of_each_habitat[habitat_id])
                out_file = os.path.join(self.out_dir, f"habitat_{habitat_id}_radiomics.csv")
                radiomics_of_each_habitat[habitat_id].to_csv(out_file, index=True)
                logging.info(f"已保存生境 {habitat_id} 的组学特征到 {out_file}")
            else:
                logging.error(f"生境 {habitat_id} 没有有效的组学特征数据")
        
        habitat_count.columns = [f"has_habitat_{i}" for i in range(1, n_habitats+1)]
        habitat_count.to_csv(os.path.join(self.out_dir, "habitat_count.csv"), index=True)
        logging.info("已保存生境数量信息")
        return radiomics_of_each_habitat

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(HabitatFeatureExtractor._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def run(self, feature_types: Optional[List[str]] = None, n_habitats: Optional[int] = None, mode: str = 'both'):
        """运行完整的分析流程"""
        # 特征提取
        if self.out_dir and (mode == 'extract' or mode == 'both'):
            images_paths, habitat_paths = self.get_mask_and_raw_files()
            self.extract_features(images_paths, habitat_paths)
            
        # 特征解析
        if feature_types and self.out_dir and (mode == 'parse' or mode == 'both'):
            self.parse_features(feature_types, n_habitats)
            
        return self

    def calculate_MSI_matrix(self, habitat_array: np.ndarray, unique_class: int) -> np.ndarray:
        """
        Calculate the MSI matrix from habitat array
        
        Args:
            habitat_array: Array representing the habitat map
            unique_class: Number of habitat classes (including background)
            
        Returns:
            msi_matrix: Calculated MSI matrix
        """
        # Find the minimum bounding box of non-zero regions in habitat_array
        roi_z, roi_y, roi_x = np.where(habitat_array != 0)
        
        if len(roi_z) == 0:
            logging.warning("No non-zero elements found in habitat array")
            return np.zeros((unique_class, unique_class), dtype=np.int64)
            
        z_min, z_max = np.min(roi_z), np.max(roi_z)
        y_min, y_max = np.min(roi_y), np.max(roi_y)
        x_min, x_max = np.min(roi_x), np.max(roi_x)

        # Extract data within the bounding box
        box_of_VOI = habitat_array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        
        # Add a layer of zeros around the box to capture boundary information
        box_of_VOI = np.pad(box_of_VOI, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)

        # Define 3D neighborhood (face-connected only)
        neighborhood_3d_cube_only = [
            (-1, 0, 0), (1, 0, 0),  # Up and down neighbors
            (0, -1, 0), (0, 1, 0),  # Left and right neighbors
            (0, 0, -1), (0, 0, 1)   # Front and back neighbors
        ]

        # Initialize MSI matrix
        msi_matrix = np.zeros((unique_class, unique_class), dtype=np.int64)
        
        # Traverse the 3D image and count neighbor relationships
        for z in range(box_of_VOI.shape[0]):  
            for y in range(box_of_VOI.shape[1]):  
                for x in range(box_of_VOI.shape[2]): 
                    # Get current voxel value
                    current_voxel_value = box_of_VOI[z, y, x]

                    # Check all neighbors
                    for dz, dy, dx in neighborhood_3d_cube_only:
                        neighbor_z = z + dz
                        neighbor_y = y + dy
                        neighbor_x = x + dx
                        
                        # Check if neighbor is within image bounds
                        if 0 <= neighbor_z < box_of_VOI.shape[0] and \
                        0 <= neighbor_y < box_of_VOI.shape[1] and \
                        0 <= neighbor_x < box_of_VOI.shape[2]:
                            
                            neighbor_voxel_value = box_of_VOI[neighbor_z, neighbor_y, neighbor_x]
                            
                            # Update MSI matrix
                            msi_matrix[current_voxel_value, neighbor_voxel_value] += 1

        return msi_matrix

    def calculate_MSI_features(self, msi_matrix: np.ndarray, name: str) -> Dict:
        """
        Calculate MSI features from the MSI matrix
        
        Args:
            msi_matrix: MSI matrix
            name: Prefix for feature names
            
        Returns:
            Dict: Calculated MSI features
        """
        # Assert that msi_matrix is square and contains no negative values
        assert msi_matrix.shape[0] == msi_matrix.shape[1], f'msi_matrix of {name} is not a square matrix'
        assert np.all(msi_matrix >= 0), f'msi_matrix of {name} has negative value'
        
        # First-order features: Volume of each subregion (diagonal) and borders of two differing subregions (off-diagonal)
        firstorder_feature = {}
        for i in range(0, msi_matrix.shape[0]):
            for j in range(i+1, msi_matrix.shape[0]):
                firstorder_feature['firstorder_{}_and_{}'.format(i, j)] = msi_matrix[i, j]

        # Calculate diagonal elements, excluding background
        for i in range(1, msi_matrix.shape[0]):
            firstorder_feature['firstorder_{}_and_{}'.format(i, i)] = msi_matrix[i, i]

        # Normalized first-order features, denominator includes only the lower triangular part excluding the first element
        denominator_mat = np.tril(msi_matrix, k=0)
        denominator_mat[0] = 0
        denominator = np.sum(denominator_mat)
        
        if denominator == 0:
            logging.warning(f"MSI matrix denominator is 0 for {name}, cannot calculate normalized features")
            normal_msi_matrix = np.zeros_like(msi_matrix, dtype=float)
        else:
            normal_msi_matrix = msi_matrix / denominator
            
        firstorder_feature_normalized = {}
        for i in range(0, normal_msi_matrix.shape[0]):
            for j in range(i+1, normal_msi_matrix.shape[1]):
                firstorder_feature_normalized['firstorder_normalized_{}_and_{}'.format(i, j)] = normal_msi_matrix[i, j]

        for i in range(1, normal_msi_matrix.shape[0]):
            firstorder_feature_normalized['firstorder_normalized_{}_and_{}'.format(i, i)] = normal_msi_matrix[i, i]
        
        # Second-order features based on normalized MSI matrix
        p = normal_msi_matrix.copy()
        
        # Calculate contrast
        i_indices, j_indices = np.indices(p.shape)
        contrast = np.sum((i_indices - j_indices)**2 * p)
        
        # Calculate homogeneity
        homogeneity = np.sum(p / (1.0 + (i_indices - j_indices)**2))
        
        # Calculate correlation
        px = np.sum(p, axis=1)
        py = np.sum(p, axis=0)
        
        ux = np.sum(px * np.arange(len(px)))
        uy = np.sum(py * np.arange(len(py)))
        
        sigmax = np.sqrt(np.sum(px * (np.arange(len(px)) - ux)**2))
        sigmay = np.sqrt(np.sum(py * (np.arange(len(py)) - uy)**2))
        
        if sigmax > 0 and sigmay > 0:
            sum_p_ij = np.sum(p * i_indices * j_indices)
            correlation = (sum_p_ij - ux * uy) / (sigmax * sigmay)
        else:
            correlation = 1.0
        
        # Calculate energy
        energy = np.sum(p**2)
        
        secondorder_feature = { 
            'contrast': contrast,
            'homogeneity': homogeneity,
            'correlation': correlation,
            'energy': energy
        }

        # Combine all features
        msi_feature = {**firstorder_feature, **firstorder_feature_normalized, **secondorder_feature}
        return msi_feature

    def extract_MSI_features(self, habitat_path: str, n_habitats: int, subj: str) -> Dict:
        """
        Extract MSI features from a single habitat map
        
        Args:
            habitat_path: Path to the habitat map file
            n_habitats: Number of habitats
            subj: Subject ID
            
        Returns:
            Dict: Extracted MSI features
        """
        try:
            img = sitk.ReadImage(habitat_path)
            array = sitk.GetArrayFromImage(img)
            
            unique_class = n_habitats+1  # Number of habitats + 1 (including background)

            # Calculate MSI matrix
            msi_matrix = self.calculate_MSI_matrix(array, unique_class)

            # Calculate MSI features
            msi_feature = self.calculate_MSI_features(msi_matrix, subj)
            
            return msi_feature
        except Exception as e:
            logging.error(f"Error extracting MSI features for subject {subj}: {str(e)}")
            return {"error": str(e)}

    def _extract_msi_features(self):
        """提取MSI特征并保存为CSV文件"""
        logging.info("开始提取MSI特征")
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
                        logging.error(f"受试者 {subj} 的MSI特征提取出错: {features['error']}")
                        # 如果已经有其他受试者成功处理，创建一个空DataFrame
                        if len(msi_features_list) > 0:
                            empty_df = pd.DataFrame(
                                data=np.nan, 
                                index=[subj], 
                                columns=msi_features_list[0].columns
                            )
                            msi_features_list.append(empty_df)
                else:
                    logging.error(f"受试者 {subj} 没有MSI特征数据")
            except Exception as e:
                logging.error(f"处理受试者 {subj} 的MSI特征时出错: {str(e)}")
                # 如果已经有其他受试者成功处理，创建一个空DataFrame
                if len(msi_features_list) > 0:
                    empty_df = pd.DataFrame(
                        data=np.nan, 
                        index=[subj], 
                        columns=msi_features_list[0].columns
                    )
                    msi_features_list.append(empty_df)
        
        if len(msi_features_list) > 0:
            msi_features_df = pd.concat(msi_features_list)
            out_file = os.path.join(self.out_dir, "msi_features.csv")
            msi_features_df.to_csv(out_file, index=True)
            logging.info(f"已保存MSI特征到 {out_file}")
            return msi_features_df
        else:
            logging.error("没有有效的MSI特征数据")
            return None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生境特征提取工具')
    
    # 特征提取相关参数
    parser.add_argument('--params_file_of_non_habitat', type=str,
                        help='用于从原始影像提取组学特征的参数文件')
    parser.add_argument('--params_file_of_habitat', type=str,
                        help='用于从生境影像提取组学特征的参数文件')
    parser.add_argument('--raw_img_folder', type=str,
                        help='原始影像根目录')
    parser.add_argument('--habitats_map_folder', type=str,
                        help='生境图根目录')
    parser.add_argument('--out_dir', type=str,
                        help='输出目录')
    parser.add_argument('--n_processes', type=int,
                        help='使用的进程数')
    parser.add_argument('--habitat_pattern', type=str,
                        choices=['*_habitats.nrrd', '*_habitats_remapped.nrrd'],
                        help='生境文件匹配模式')
    parser.add_argument('--voxel_cutoff', type=int, default=10,
                        help='MSI特征计算时过滤小区域的体素阈值')
    
    # 特征解析相关参数
    parser.add_argument('--feature_types', nargs='+', type=str,
                        choices=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi'],
                        help='要解析的特征类型')
    parser.add_argument('--n_habitats', type=int,
                        help='要处理的生境数量（如不指定，将从habitats.csv中自动读取）')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['extract', 'parse', 'both'],
                        help='运行模式：仅提取特征(extract)、仅解析特征(parse)或两者都做(both)')
    
    return parser.parse_args()
