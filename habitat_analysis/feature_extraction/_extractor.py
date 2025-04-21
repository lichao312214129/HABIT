#!/usr/bin/env python
"""
Extract features from habitat maps:
1. Radiomic features of raw images within different habitats (mask=each habitat map, raw=each raw image)
2. Radiomic features of habitats within the entire ROI (mask=ROI, raw=the whole habitat map)
3. Number of disconnected regions and volume percentage for each habitat
Use case: Extract radiomic features of habitats after obtaining them using get_habitats

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
import trimesh  # 提取LBP3D需要，放在这里保证不会影响其他功能
import scipy # 提取LBP3D需要，放在这里保证不会影响其他功能
import logging
import numpy as np
import os
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import warnings
import multiprocessing
from functools import partial
import json
import logging
import argparse
import sys


class HabitatFeatureExtractor(object):
    
    def __init__(self, 
                params_file_of_non_habitat=None,
                params_file_of_habitat=None,
                raw_img_folder=None, 
                habitats_map_folder=None, 
                out_file=None,
                n_processes=None,
                habitat_pattern=None):

        self.params_file_of_non_habitat = params_file_of_non_habitat  # 非habitat的参数文件
        self.params_file_of_habitat = params_file_of_habitat  # habitat的参数文件
        self.raw_img_folder = raw_img_folder  # 原始影像根目录
        self.habitats_map_folder = habitats_map_folder  # habitats map根目录
        self.out_file = out_file  # 输出文件路径
        self._habitat_pattern = habitat_pattern
        
        # 如果未指定进程数，则使用CPU核心数的一半
        if n_processes is None:
            self.n_processes = max(1, multiprocessing.cpu_count() // 2)
        else:
            self.n_processes = min(n_processes, multiprocessing.cpu_count()-2)  # 至少保留两个核心用于其他任务

        # Save settings
        data = time.time()
        timeArray = time.localtime(data)
        timestr = time.strftime('%Y_%m_%d %H_%M_%S', timeArray)
        # self.out_file = os.path.join(self.out_folder, "features_" + timestr + ".csv")

        self.save_every_n_files = 5

        # Get the PyRadiomics logger default log-level = ERROR
        self.out_folder = os.path.dirname(self.out_file)
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
            
        log_file_path = os.path.join(self.out_folder, f"pyradiomics_log_{timestr}.log")
        logging.basicConfig(
            filename=log_file_path,
            level=logging.ERROR,
            format="%(asctime)s:%(levelname)s:%(name)s: %(message)s"
        )

        # 初始化特征提取器是在需要时创建，以避免在多进程中的序列化问题
        # self.extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

    @staticmethod
    def get_non_radiomics_features(habitat_img):
        """ 计算每个habitat的不连通区域数及体积占比 (纯SimpleITK实现)
        未来可以扩展其它非组学特征的计算
        
        Args:
            habitat_img (sitk.Image OR str): habitat map的路径或SimpleITK图像对象
            
        Returns:
            dict: 包含各habitat统计结果的字典，格式：
                {
                    1: {'num_regions': 3, 'volume_ratio': 0.25},
                    2: {'num_regions': 1, 'volume_ratio': 0.6},
                    ...
                }
        """
        if isinstance(habitat_img, str):
            habitat_img = sitk.ReadImage(habitat_img)
        elif not isinstance(habitat_img, sitk.Image):
            raise ValueError("habitat_img must be a SimpleITK image or a file path.")

        results = {}
        
        # 计算整个habitat map的总体积（非背景体素数）
        stats_filter = sitk.StatisticsImageFilter()
        stats_filter.Execute(habitat_img != 0)  # 生成全habitat二值图像
        total_voxels = int(stats_filter.GetSum())

        label_filter = sitk.LabelStatisticsImageFilter()
        label_filter.Execute(habitat_img, habitat_img)
        labels = label_filter.GetLabels()
        labels = [int(label) for label in labels if label != 0]
        
        # 遍历每个habitat标签
        for label in labels:
            # 生成当前habitat的二值图像
            binary_img = sitk.BinaryThreshold(habitat_img, lowerThreshold=label, upperThreshold=label)
            
            # 统计当前habitat体积
            stats_filter.Execute(binary_img)
            habitat_voxels = int(stats_filter.GetSum())
            volume_ratio = habitat_voxels / total_voxels if total_voxels > 0 else 0.0
            
            # 连通域分析
            cc_filter = sitk.ConnectedComponentImageFilter()
            cc_filter.SetFullyConnected(False)  # 仅面相邻的体素算作连通
            labeled_img = cc_filter.Execute(binary_img)
            num_regions = cc_filter.GetObjectCount()
            
            # 存储结果
            results[label] = {
                'num_regions': num_regions,
                'volume_ratio': volume_ratio
            }
        results['num_habitats'] = len(labels)  # habitat的数量
        
        return results

    @staticmethod
    def extract_radiomics_features_for_whole_habitat(habitat_img, params_file):
        """ 提取整个ROI内的habitat的组学特征（mask=ROI，raw=the whole habitat map）
        Arg:
            habitat_img (sitk.Image OR str): habitat map的路径或SimpleITK图像对象

        Returns:
            dict: 包含habitat map的组学特征的字典
        """
        # 创建特征提取器
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        
        if isinstance(habitat_img, str):
            habitat_img = sitk.ReadImage(habitat_img)
        elif not isinstance(habitat_img, sitk.Image):
            raise ValueError("habitat_img must be a SimpleITK image or a file path.")

        label_filter = sitk.LabelStatisticsImageFilter()
        label_filter.Execute(habitat_img, habitat_img)
        labels = label_filter.GetLabels()
        labels = [int(label) for label in labels if label != 0]

        habitat_img_ = sitk.BinaryThreshold(
            habitat_img, 
            lowerThreshold=1, 
            upperThreshold=np.max(labels).astype(np.double), 
            insideValue=1, 
            outsideValue=0
        )  # 将habitat_img不为0的都设置为1，即整个habitat map的mask 

        return extractor.execute(
            imageFilepath=habitat_img,
            maskFilepath=habitat_img_, 
            label=1
        )

    @staticmethod
    def extract_tranditional_radiomics(image_path, habitat_path, subject_id, params_file):
        """ 提取常规组学特征，即mask是ROI，原始影像是raw
        Arg:
            image_path (sitk.Image OR str): 原始影像的路径或SimpleITK图像对象
            habitat_img (sitk.Image OR str): habitat map的路径或SimpleITK图像对象
            subject_id (str): 受试者ID
            params_file (str): 特征提取参数文件的路径

        Returns:
            dict: 包含habitat map的组学特征的字典
        """
        # 创建特征提取器
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        
        # Get mask and img
        habitat_img = sitk.ReadImage(habitat_path)
        raw_img = sitk.ReadImage(image_path)

        # 检查raw和mask在direction，spacing，origin上是否一致
        is_different = False
        if raw_img.GetDirection() != habitat_img.GetDirection():
            logging.info(f"raw and mask direction is different: {subject_id}")
            habitat_img.SetDirection(raw_img.GetDirection())
            is_different = True

        if raw_img.GetOrigin() != habitat_img.GetOrigin():
            logging.info(f"raw and mask origin is different: {subject_id}")
            habitat_img.SetOrigin(raw_img.GetOrigin())
            is_different = True

        if raw_img.GetSpacing() != habitat_img.GetSpacing():
            logging.info(f"raw and mask spacing is different: {subject_id}")
            habitat_img.SetSpacing(raw_img.GetSpacing())
            is_different = True

        # mask中有几个label
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
        )  # 将habitat_img不为0的都设置为1，即整个habitat map的mask 

        return extractor.execute(
            imageFilepath=raw_img,
            maskFilepath=mask, 
            label=1
        )
    
    @staticmethod
    def extract_radiomics_features_from_each_habitat(
            habitat_path, 
            image_path, 
            subject_id, 
            params_file):
        """
        Extracting radiomics features from each habitat for raw image
        """
        # 创建特征提取器
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
        
        # Get mask and img
        habitat_img = sitk.ReadImage(habitat_path)
        raw_img = sitk.ReadImage(image_path)

        # 检查raw和mask在direction，spacing，origin上是否一致
        is_different = False
        if raw_img.GetDirection() != habitat_img.GetDirection():
            logging.info(f"raw and mask direction is different: {subject_id}")
            habitat_img.SetDirection(raw_img.GetDirection())
            is_different = True

        if raw_img.GetOrigin() != habitat_img.GetOrigin():
            logging.info(f"raw and mask origin is different: {subject_id}")
            habitat_img.SetOrigin(raw_img.GetOrigin())
            is_different = True

        if raw_img.GetSpacing() != habitat_img.GetSpacing():
            logging.info(f"raw and mask spacing is different: {subject_id}")
            habitat_img.SetSpacing(raw_img.GetSpacing())
            is_different = True

        # mask中有几个label
        label = sitk.LabelStatisticsImageFilter()
        label.Execute(habitat_img, habitat_img)
        labels = label.GetLabels()
        labels = [int(label) for label in labels if label != 0]

        # Extract features
        featureVector = {}
        for label in labels:
            featureVector[label] = extractor.execute(
                imageFilepath=raw_img, 
                maskFilepath=habitat_img, 
                label=label
            )
               
        return featureVector

    def get_mask_and_raw_files(self):
        """
        Get all mask and raw files
        """
        # Get all image files
        images_paths = {}
        images_root = os.path.join(self.raw_img_folder, "images")
        subjs = os.listdir(images_root)
        for subj in subjs:
            images_paths[subj] = {}
            subj_path = os.path.join(images_root, subj)
            img_subfolders = os.listdir(subj_path)
            for img_subfolder in img_subfolders:
                img_subfolder_path = os.path.join(subj_path, img_subfolder)
                if os.path.isdir(img_subfolder_path):
                    img_files = os.listdir(img_subfolder_path)
                    # 不应该有多个文件，如果有则取第一个并报警
                    if len(img_files) > 1:
                        logging.error(f"More than one image files in {subj}/{img_subfolder}")
                    img_file = img_files[0]
                    images_paths[subj][img_subfolder] = os.path.join(img_subfolder_path, img_file)

        # Get all habitat files using the specified pattern
        habitat_paths = {}
        for subj in Path(self.habitats_map_folder).glob(self._habitat_pattern):
            key = subj.name.replace(self._habitat_pattern.replace("*", ""), "")
            habitat_paths[key] = str(subj)

        return images_paths, habitat_paths

    def process_subject(self, subj, images_paths, habitat_paths):
        """
        处理单个受试者的特征提取，适用于多进程
        """
        try:
            imgs = list(images_paths[subj].keys())
            subject_features = {}

            # 提取非组学特征
            non_radiomics_features = self.get_non_radiomics_features(habitat_paths[subj])
            subject_features['non_radiomics_features'] = non_radiomics_features

            # 提取常规组学特征
            subject_features['tranditional_radiomics_features'] = {}
            for img in imgs:
                subject_features['tranditional_radiomics_features'][img] = \
                    self.extract_tranditional_radiomics(
                    images_paths[subj][img], 
                    habitat_paths[subj], 
                    subj,
                    self.params_file_of_non_habitat
                )

            # 提取整个habitat map的组学特征
            radiomics_features_of_whole_habitat = self.extract_radiomics_features_for_whole_habitat(
                habitat_paths[subj], 
                self.params_file_of_habitat
            )
            subject_features['radiomics_features_of_whole_habitat_map'] = radiomics_features_of_whole_habitat
            
            # 提取每个habitat内原始影像的组学特征
            subject_features['radiomics_features_from_each_habitat'] = {}
            for img in imgs:
                subject_features['radiomics_features_from_each_habitat'][img] = self.extract_radiomics_features_from_each_habitat(
                    habitat_paths[subj], 
                    images_paths[subj][img], 
                    subj, 
                    self.params_file_of_non_habitat
                )

            return subj, subject_features
        
        except Exception as e:
            logging.error(f"Error processing subject {subj}: {str(e)}")
            return subj, {"error": str(e)}

    def extract(self, images_paths, habitat_paths):
        """
        extract for all subjects using multiprocessing
        """
        features = {}
        subjs = list(set(images_paths.keys()) & set(habitat_paths.keys()))
        
        if not subjs:
            logging.error("No matching subjects found between images and habitat maps")
            return self
            
        print(f"**************Extracting features for {len(subjs)} subjects using {self.n_processes} processes**************")
        
        # 使用临时文件保存中间结果，避免进程间大量数据传输
        temp_dir = os.path.join(self.out_folder, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # 使用进程池并行处理
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            # 创建处理函数
            process_func = partial(self.process_subject, images_paths=images_paths, habitat_paths=habitat_paths)
            
            total = len(subjs)
            for i, (subj, subject_features) in enumerate(pool.imap_unordered(process_func, subjs)):
                features[subj] = subject_features
                
                # 创建简单的进度条
                progress = int((i + 1) / total * 50)  # 50是进度条的长度
                bar = "█" * progress + "-" * (50 - progress)
                percent = (i + 1) / total * 100
                print(f"\r[{bar}] {percent:.2f}% ({i+1}/{total})", end="")
                
                # 定期保存中间结果
                if (i + 1) % self.save_every_n_files == 0:
                    temp_file = os.path.join(temp_dir, f"features_temp_{i+1}.npy")
                    np.save(temp_file, features)
                    
        # 删除临时文件
        for temp_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, temp_file))
        os.rmdir(temp_dir)
            
        # 保存最终结果
        np.save(self.out_file, features)
        print(f"Features saved to {self.out_file}")
        return self
    
    def run(self):
        """
        运行提取器
        """
        # 获取mask和raw文件路径
        images_paths, habitat_paths = self.get_mask_and_raw_files()
        
        # 提取特征
        self.extract(images_paths, habitat_paths)
        print("Feature extraction completed.")
        return self
    

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Extract radiomics features from habitat maps')
    
    # 必需参数
    parser.add_argument('--params_file_of_non_habitat', type=str, required=True,
                        help='Parameter file for extracting radiomics features from raw image')
    
    parser.add_argument('--params_file_of_habitat', type=str, required=True,
                        help='Parameter file for extracting radiomics features from habitat image')
    
    parser.add_argument('--raw_img_folder', type=str, required=True,
                        help='Root folder containing raw images')
    
    parser.add_argument('--habitats_map_folder', type=str, required=True,
                        help='Folder containing habitat maps')
    
    parser.add_argument('--out_file', type=str, required=True,
                        help='Output file path')
    
    # 可选参数
    parser.add_argument('--n_processes', type=int, required=True,
                        help='Number of processes to use')
    
    parser.add_argument('--habitat_pattern', type=str, required=True,
                        choices=['*_habitats.nrrd', '*_habitats_remapped.nrrd'],
                        help='Pattern to match habitat files')
    
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    # If no command line arguments are provided, use default values
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--params_file_of_non_habitat', './parameter.yaml',
            '--params_file_of_habitat', './parameter_habitat.yaml',
            '--raw_img_folder', 'G:\\Registration_ICC_structured',
            '--habitats_map_folder', 'F:\\work\\research\\radiomics_TLSs\\data\\results_184_gmm',
            '--out_file', 'F:\\work\\research\\radiomics_TLSs\\data\\results_184_gmm\\habitats_radiomics_features.npy',
            '--n_processes', '6',
            '--habitat_pattern', '*_habitats.nrrd'
        ])
    # 解析命令行参数
    args = parse_arguments()
    
    s = time.time()
    # 创建提取器实例并运行
    extractor = HabitatFeatureExtractor(
        params_file_of_non_habitat=args.params_file_of_non_habitat,
        params_file_of_habitat=args.params_file_of_habitat,
        raw_img_folder=args.raw_img_folder, 
        habitats_map_folder=args.habitats_map_folder, 
        out_file=args.out_file,
        n_processes=args.n_processes,
        habitat_pattern=args.habitat_pattern
    )
    extractor.run()
    e = time.time()
    print(f"Total time: {e - s:.2f} seconds")
