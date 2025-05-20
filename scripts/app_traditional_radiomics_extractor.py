#!/usr/bin/env python
"""
传统组学特征提取工具
该工具提供从医学影像中提取传统放射组学特征的功能
使用多进程并行处理以提高效率

数据结构如下:
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
"""

import time
import logging
import numpy as np
import os
import SimpleITK as sitk
from radiomics import featureextractor
import warnings
import multiprocessing
from functools import partial
import argparse
import sys
import pandas as pd
from typing import Dict, List, Optional, Union

# 禁用警告
warnings.filterwarnings('ignore')

# 导入进度条工具
from habit.utils.progress_utils import CustomTqdm

class TraditionalRadiomicsExtractor:
    """传统组学特征提取类"""

    def __init__(self,
                params_file=None,
                images_folder=None,
                masks_folder=None,
                out_dir=None,
                n_processes=None):
        """
        初始化传统组学特征提取器

        Args:
            params_file: 用于提取组学特征的参数文件
            images_folder: 影像根目录
            masks_folder: 掩码根目录
            out_dir: 输出目录
            n_processes: 进程数
        """
        # 特征提取相关参数
        self.params_file = params_file
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.out_dir = out_dir

        # 进程数设置
        if n_processes is None:
            self.n_processes = max(1, multiprocessing.cpu_count() // 2)
        else:
            self.n_processes = min(n_processes, multiprocessing.cpu_count() - 2)

        # 设置日志
        self._setup_logging()

        # 保存设置
        self.save_every_n_files = 5

    def _setup_logging(self):
        """设置日志配置"""
        # 获取时间戳
        data = time.time()
        timeArray = time.localtime(data)
        timestr = time.strftime('%Y_%m_%d %H_%M_%S', timeArray)

        # 创建输出目录
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        log_file = os.path.join(self.out_dir, f'radiomics_extraction_{timestr}.log')

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

        logging.info(f"日志文件将保存到: {log_file}")

    @staticmethod
    def extract_radiomics_features(image_path, mask_path, subject_id, params_file):
        """提取组学特征"""
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

            mask_img = sitk.ReadImage(mask_path)
            raw_img = sitk.ReadImage(image_path)

            # 确保图像和掩码有相同的方向、原点和间距
            if raw_img.GetDirection() != mask_img.GetDirection():
                logging.info(f"图像和掩码方向不同: {subject_id}")
                mask_img.SetDirection(raw_img.GetDirection())

            if raw_img.GetOrigin() != mask_img.GetOrigin():
                logging.info(f"图像和掩码原点不同: {subject_id}")
                mask_img.SetOrigin(raw_img.GetOrigin())

            if raw_img.GetSpacing() != mask_img.GetSpacing():
                logging.info(f"图像和掩码间距不同: {subject_id}")
                mask_img.SetSpacing(raw_img.GetSpacing())

            # 使用label=1提取特征
            return extractor.execute(
                imageFilepath=raw_img,
                maskFilepath=mask_img,
                label=1
            )
        except Exception as e:
            logging.error(f"提取组学特征时出错: {str(e)}")
            return {"error": f"特征提取错误: {str(e)}"}

    def get_image_and_mask_files(self):
        """获取所有影像和掩码文件路径"""
        images_paths = {}
        images_root = os.path.join(self.images_folder, "images")
        subjs = os.listdir(images_root)
        for subj in subjs:
            images_paths[subj] = {}
            subj_path = os.path.join(images_root, subj)
            img_subfolders = os.listdir(subj_path)
            for img_subfolder in img_subfolders:
                img_subfolder_path = os.path.join(subj_path, img_subfolder)
                if os.path.isdir(img_subfolder_path):
                    img_files = os.listdir(img_subfolder_path)
                    if len(img_files) > 1:
                        logging.warning(f"文件夹 {subj}/{img_subfolder} 中包含多个影像文件")
                    img_file = img_files[0]
                    images_paths[subj][img_subfolder] = os.path.join(img_subfolder_path, img_file)

        masks_paths = {}
        masks_root = os.path.join(self.images_folder, "masks")
        subjs = os.listdir(masks_root)
        for subj in subjs:
            masks_paths[subj] = {}
            subj_path = os.path.join(masks_root, subj)
            mask_subfolders = os.listdir(subj_path)
            for mask_subfolder in mask_subfolders:
                mask_subfolder_path = os.path.join(subj_path, mask_subfolder)
                if os.path.isdir(mask_subfolder_path):
                    mask_files = os.listdir(mask_subfolder_path)
                    if len(mask_files) > 1:
                        logging.warning(f"文件夹 {subj}/{mask_subfolder} 中包含多个掩码文件")
                    mask_file = mask_files[0]
                    masks_paths[subj][mask_subfolder] = os.path.join(mask_subfolder_path, mask_file)

        return images_paths, masks_paths

    def process_subject(self, subj, images_paths, masks_paths):
        """处理单个受试者的特征提取"""
        subject_features = {}

        # 获取所有影像类型
        imgs = list(set(images_paths[subj].keys()) & set(masks_paths[subj].keys()))

        if not imgs:
            logging.warning(f"受试者 {subj} 没有找到匹配的影像和掩码")
            return subj, {}

        for img in imgs:
            try:
                features = self.extract_radiomics_features(
                    images_paths[subj][img],
                    masks_paths[subj][img],
                    f"{subj}/{img}",
                    self.params_file
                )
                subject_features[img] = features
            except Exception as e:
                logging.error(f"处理受试者 {subj} 的影像 {img} 时出错: {str(e)}")
                subject_features[img] = {"error": str(e)}

        return subj, subject_features

    def extract_features(self):
        """提取所有受试者的组学特征"""
        images_paths, masks_paths = self.get_image_and_mask_files()
        features = {}

        # 找到同时存在于影像和掩码中的受试者
        subjs = list(set(images_paths.keys()) & set(masks_paths.keys()))

        if not subjs:
            logging.error("未在影像和掩码文件夹中找到匹配的受试者")
            return self

        print(f"**************开始为 {len(subjs)} 个受试者提取组学特征，使用 {self.n_processes} 个进程**************")

        # 创建临时目录用于保存中间结果
        temp_dir = os.path.join(self.out_dir, "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            process_func = partial(self.process_subject, images_paths=images_paths, masks_paths=masks_paths)

            total = len(subjs)
            progress_bar = CustomTqdm(total=total, desc="Extracting Radiomics Features")

            for subj, subject_features in pool.imap_unordered(process_func, subjs):
                features[subj] = subject_features
                progress_bar.update(1)

                # 定期保存中间结果（仅在进程中保存）

        # 直接转换为CSV格式
        self.convert_to_csv(features)

        return self

    def convert_to_csv(self, features):
        """将特征转换为CSV格式"""
        try:
            # 创建一个字典，用于存储每种影像类型的特征
            image_types = set()
            for subj in features:
                image_types.update(features[subj].keys())

            for img_type in image_types:
                # 收集所有受试者该影像类型的特征
                subjs = []
                feature_dfs = []

                for subj in features:
                    if img_type in features[subj]:
                        subjs.append(subj)
                        # 将特征字典转换为DataFrame
                        feature_dict = features[subj][img_type]
                        # 排除诊断相关列
                        feature_dict = {k: v for k, v in feature_dict.items() if 'diagnostic' not in k}
                        feature_df = pd.DataFrame([feature_dict])
                        feature_dfs.append(feature_df)

                if feature_dfs:
                    # 合并所有受试者的特征
                    combined_df = pd.concat(feature_dfs, ignore_index=True)
                    combined_df.index = subjs

                    # 保存为CSV
                    out_file = os.path.join(self.out_dir, f"radiomics_features_{img_type}.csv")
                    combined_df.to_csv(out_file)
                    print(f"已将 {img_type} 的组学特征保存到 {out_file}")

            # 创建合并所有影像类型的特征文件
            all_features = {}
            for subj in features:
                all_features[subj] = {}
                for img_type in features[subj]:
                    feature_dict = features[subj][img_type]
                    # 排除诊断相关列
                    feature_dict = {k: v for k, v in feature_dict.items() if 'diagnostic' not in k}
                    # 添加影像类型前缀
                    prefixed_dict = {f"{img_type}_{k}": v for k, v in feature_dict.items()}
                    all_features[subj].update(prefixed_dict)

            # 转换为DataFrame
            all_df = pd.DataFrame.from_dict(all_features, orient='index')

            # 保存为CSV
            out_file = os.path.join(self.out_dir, "radiomics_features_all.csv")
            all_df.to_csv(out_file)
            print(f"已将所有组学特征保存到 {out_file}")

        except Exception as e:
            logging.error(f"转换特征为CSV时出错: {str(e)}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='传统组学特征提取工具')

    # 特征提取相关参数
    parser.add_argument('--params_file', type=str, required=True,
                        help='用于提取组学特征的参数文件')
    parser.add_argument('--images_folder', type=str, required=True,
                        help='影像根目录，应包含images和masks子目录')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--n_processes', type=int,
                        help='使用的进程数，默认为CPU核心数的一半')

    return parser.parse_args()

if __name__ == "__main__":
    # 如果未提供命令行参数，使用默认值
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--params_file', './config/parameter.yaml',
            '--images_folder', '../demo_data/datasets',
            '--out_dir', '../demo_data/results',
            '--n_processes', '4'
        ])

    # 解析命令行参数
    args = parse_arguments()

    # 创建特征提取器实例并运行
    start_time = time.time()
    extractor = TraditionalRadiomicsExtractor(
        params_file=args.params_file,
        images_folder=args.images_folder,
        out_dir=args.out_dir,
        n_processes=args.n_processes
    )
    extractor.extract_features()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    # 命令行形式为（用sys真实参数）：
    # python traditional_radiomics_extractor.py --params_file parameter.yaml --images_folder F:\work\workstation_b\dingHuYingXiang\the_third_training_202504\demo_data\datasets --out_dir F:\work\workstation_b\dingHuYingXiang\the_third_training_202504\demo_data\results --n_processes 4