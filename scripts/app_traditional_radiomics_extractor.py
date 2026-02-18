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
import yaml
import csv
from typing import Dict, List, Optional, Union, Any, Tuple

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
                n_processes=None,
                config_file=None):
        """
        初始化传统组学特征提取器

        Args:
            params_file: 用于提取组学特征的参数文件
            images_folder: 影像根目录
            masks_folder: 掩码根目录
            out_dir: 输出目录
            n_processes: 进程数
            config_file: 配置文件路径
        """
        # 如果提供了配置文件，从配置文件加载参数
        if config_file:
            self.load_config(config_file)
        
        # 特征提取相关参数，命令行参数优先级高于配置文件
        self.params_file = params_file or self.params_file
        self.images_folder = images_folder or self.images_folder
        self.masks_folder = masks_folder
        self.out_dir = out_dir or self.out_dir
        
        # 如果命令行没有指定进程数，使用配置文件的值或默认值
        if n_processes is not None:
            self.n_processes = n_processes
        elif not hasattr(self, 'n_processes') or self.n_processes == 0:
            self.n_processes = max(1, multiprocessing.cpu_count() // 2)
        
        # 保证进程数不超过CPU数量
        self.n_processes = min(self.n_processes, multiprocessing.cpu_count() - 2)
        
        # 设置日志
        self._setup_logging()
        
        # 保存设置
        if not hasattr(self, 'save_every_n_files'):
            self.save_every_n_files = 5
            
        # 处理图像类型设置
        if not hasattr(self, 'process_image_types'):
            self.process_image_types = None  # 默认处理所有图像类型
        
        # Target labels to extract from mask.
        # Default keeps backward compatibility with previous single-label behavior.
        if not hasattr(self, 'target_labels') or self.target_labels is None:
            self.target_labels = [1]
        self.target_labels = [int(label) for label in self.target_labels]
        if len(self.target_labels) == 0:
            raise ValueError("processing.target_labels must contain at least one label.")
            
        # 设置导出格式
        if not hasattr(self, 'export_format'):
            self.export_format = 'csv'
            
        # 设置时间戳
        if not hasattr(self, 'add_timestamp'):
            self.add_timestamp = True

    def load_config(self, config_file: str) -> None:
        """
        从YAML配置文件加载设置
        
        Args:
            config_file: 配置文件路径
        """
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # 路径设置
            if 'paths' in config:
                self.params_file = config['paths'].get('params_file')
                self.images_folder = config['paths'].get('images_folder')
                self.out_dir = config['paths'].get('out_dir')
                
            # 处理设置
            if 'processing' in config:
                self.n_processes = config['processing'].get('n_processes', 0)
                self.save_every_n_files = config['processing'].get('save_every_n_files', 5)
                self.process_image_types = config['processing'].get('process_image_types')
                # Labels to be merged as foreground before extraction.
                # Example: [1, 2] means voxels with value 1 or 2 become 1, others become 0.
                self.target_labels = config['processing'].get('target_labels', [1])
                
            # 导出设置
            if 'export' in config:
                self.export_by_image_type = config['export'].get('export_by_image_type', True)
                self.export_combined = config['export'].get('export_combined', True)
                self.export_format = config['export'].get('export_format', 'csv')
                self.add_timestamp = config['export'].get('add_timestamp', True)
                
            # 日志设置
            if 'logging' in config:
                self.log_level = config['logging'].get('level', 'INFO')
                self.console_output = config['logging'].get('console_output', True)
                self.file_output = config['logging'].get('file_output', True)
                
            logging.info(f"已从配置文件 {config_file} 加载设置")
        except Exception as e:
            logging.error(f"加载配置文件时出错: {str(e)}")
            raise

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

        # 获取日志级别
        log_level = logging.INFO
        if hasattr(self, 'log_level'):
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            log_level = level_map.get(self.log_level.upper(), logging.INFO)

        # 配置日志处理器
        handlers = []
        if not hasattr(self, 'console_output') or self.console_output:
            handlers.append(logging.StreamHandler())
        if not hasattr(self, 'file_output') or self.file_output:
            handlers.append(logging.FileHandler(log_file))

        # 配置日志
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

        logging.info(f"日志文件将保存到: {log_file}")

    @staticmethod
    def _merge_target_labels_to_binary_mask(
        mask_img: sitk.Image,
        target_labels: List[int],
        subject_id: str
    ) -> sitk.Image:
        """
        Convert selected labels in the original mask to a binary mask.

        Selected labels are mapped to 1 (foreground), all other voxels are mapped to 0.

        Args:
            mask_img (sitk.Image): Original multi-label mask image.
            target_labels (List[int]): Labels to keep as foreground.
            subject_id (str): Subject identifier for logging and error context.

        Returns:
            sitk.Image: Binary mask image with voxel values {0, 1}.
        """
        mask_array: np.ndarray = sitk.GetArrayFromImage(mask_img)
        binary_array: np.ndarray = np.isin(mask_array, target_labels).astype(np.uint8)

        if int(binary_array.sum()) == 0:
            raise ValueError(
                f"No voxels found for target labels {target_labels} in subject {subject_id}."
            )

        binary_mask_img: sitk.Image = sitk.GetImageFromArray(binary_array)
        # Preserve geometry so the binary mask remains aligned with the input image.
        binary_mask_img.CopyInformation(mask_img)
        return binary_mask_img

    @staticmethod
    def extract_radiomics_features(
        image_path: str,
        mask_path: str,
        subject_id: str,
        params_file: str,
        target_labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """提取组学特征"""
        try:
            labels_to_extract: List[int] = [int(v) for v in (target_labels or [1])]
            if len(labels_to_extract) == 0:
                raise ValueError("target_labels must contain at least one label.")

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

            # Merge selected labels into a binary mask for extraction.
            # Foreground is always encoded as value 1 for PyRadiomics.
            binary_mask_img: sitk.Image = TraditionalRadiomicsExtractor._merge_target_labels_to_binary_mask(
                mask_img=mask_img,
                target_labels=labels_to_extract,
                subject_id=subject_id
            )

            return extractor.execute(
                imageFilepath=raw_img,
                maskFilepath=binary_mask_img,
                label=1
            )
        except Exception as e:
            logging.error(f"提取组学特征时出错: {str(e)}")
            return {"error": f"特征提取错误: {str(e)}"}

    def get_image_and_mask_files(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
        """
        Get all image and mask file paths.

        Supported input styles for `self.images_folder`:
        1) Dataset root directory that contains `images/` and `masks/`.
        2) YAML file list that explicitly provides image/mask paths.

        Returns:
            Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
                - images_paths: subject -> image_type -> image_path
                - masks_paths: subject -> image_type -> mask_path
        """
        # Support YAML file list input for consistency with preprocessing behavior.
        if (
            isinstance(self.images_folder, str)
            and os.path.isfile(self.images_folder)
            and self.images_folder.lower().endswith((".yaml", ".yml"))
        ):
            logging.info(f"Detected YAML file list input: {self.images_folder}")
            yaml_path = os.path.abspath(self.images_folder)
            yaml_dir = os.path.dirname(yaml_path)
            with open(yaml_path, "r", encoding="utf-8") as file_obj:
                # BaseLoader keeps YAML keys as raw strings (e.g., "0002886419").
                raw_config = yaml.load(file_obj, Loader=yaml.BaseLoader) or {}

            auto_select_raw = str(raw_config.get("auto_select_first_file", "true")).strip().lower()
            auto_select_first_file = auto_select_raw in {"true", "1", "yes", "y"}

            def _resolve_and_normalize_paths(raw_paths: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
                normalized: Dict[str, Dict[str, str]] = {}
                for subject_id, modality_map in (raw_paths or {}).items():
                    subject_key = str(subject_id)
                    normalized[subject_key] = {}
                    for modality, raw_path in (modality_map or {}).items():
                        modality_key = str(modality)
                        path_value = str(raw_path)
                        if not os.path.isabs(path_value):
                            path_value = os.path.normpath(os.path.join(yaml_dir, path_value))
                        if auto_select_first_file and os.path.isdir(path_value):
                            files = [name for name in os.listdir(path_value) if not name.startswith('.')]
                            if files:
                                path_value = os.path.join(path_value, files[0])
                        normalized[subject_key][modality_key] = path_value
                return normalized

            images_paths = _resolve_and_normalize_paths(raw_config.get("images", {}))
            masks_paths = _resolve_and_normalize_paths(raw_config.get("masks", {}))

            # Respect process_image_types filter when YAML input is used.
            if self.process_image_types:
                images_paths = {
                    subj: {
                        img_type: img_path
                        for img_type, img_path in img_dict.items()
                        if img_type in self.process_image_types
                    }
                    for subj, img_dict in images_paths.items()
                }
                masks_paths = {
                    subj: {
                        mask_type: mask_path
                        for mask_type, mask_path in mask_dict.items()
                        if mask_type in self.process_image_types
                    }
                    for subj, mask_dict in masks_paths.items()
                }
            return images_paths, masks_paths

        images_paths = {}
        images_root = os.path.join(self.images_folder, "images")
        if not os.path.isdir(images_root):
            raise FileNotFoundError(
                f"Images directory not found: {images_root}. "
                "Expected either a dataset root with images/masks subfolders, "
                "or a YAML file list path."
            )
        subjs = os.listdir(images_root)
        for subj in subjs:
            images_paths[subj] = {}
            subj_path = os.path.join(images_root, subj)
            img_subfolders = os.listdir(subj_path)
            for img_subfolder in img_subfolders:
                # 如果有图像类型过滤，检查是否应该处理该类型
                if self.process_image_types and img_subfolder not in self.process_image_types:
                    logging.info(f"跳过处理未包含在列表中的图像类型: {img_subfolder}")
                    continue
                
                img_subfolder_path = os.path.join(subj_path, img_subfolder)
                if os.path.isdir(img_subfolder_path):
                    img_files = os.listdir(img_subfolder_path)
                    if len(img_files) > 1:
                        logging.warning(f"文件夹 {subj}/{img_subfolder} 中包含多个影像文件")
                    img_file = img_files[0]
                    images_paths[subj][img_subfolder] = os.path.join(img_subfolder_path, img_file)

        masks_paths = {}
        masks_root = self.masks_folder if self.masks_folder else os.path.join(self.images_folder, "masks")
        if not os.path.isdir(masks_root):
            raise FileNotFoundError(
                f"Masks directory not found: {masks_root}. "
                "Expected either a dataset root with images/masks subfolders, "
                "or a YAML file list path."
            )
        subjs = os.listdir(masks_root)
        for subj in subjs:
            masks_paths[subj] = {}
            subj_path = os.path.join(masks_root, subj)
            mask_subfolders = os.listdir(subj_path)
            for mask_subfolder in mask_subfolders:
                # 如果有图像类型过滤，检查是否应该处理该类型
                if self.process_image_types and mask_subfolder not in self.process_image_types:
                    continue
                    
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
                    self.params_file,
                    self.target_labels
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

        # 根据导出格式转换特征
        if self.export_format.lower() == 'csv':
            self.convert_to_csv(features)
        elif self.export_format.lower() == 'json':
            self.convert_to_json(features)
        elif self.export_format.lower() == 'pickle':
            self.convert_to_pickle(features)
        else:
            logging.warning(f"不支持的导出格式: {self.export_format}，将使用CSV格式")
            self.convert_to_csv(features)

        return self

    def convert_to_csv(self, features):
        """将特征转换为CSV格式"""
        try:
            # 创建一个字典，用于存储每种影像类型的特征
            image_types = set()
            for subj in features:
                image_types.update(features[subj].keys())
                
            # 获取时间戳字符串
            timestamp = ""
            if hasattr(self, 'add_timestamp') and self.add_timestamp:
                data = time.time()
                timeArray = time.localtime(data)
                timestamp = f"_{time.strftime('%Y%m%d_%H%M%S', timeArray)}"

            # 按影像类型导出
            if not hasattr(self, 'export_by_image_type') or self.export_by_image_type:
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
                        # Use an explicit ID column (not index) to preserve exact subject IDs.
                        combined_df.insert(0, "ID", [str(subj_id).strip() for subj_id in subjs])

                        # 保存为CSV
                        out_file = os.path.join(self.out_dir, f"radiomics_features_{img_type}{timestamp}.csv")
                        combined_df.to_csv(out_file, index=False, quoting=csv.QUOTE_MINIMAL)
                        print(f"已将 {img_type} 的组学特征保存到 {out_file}")

            # 创建合并所有影像类型的特征文件
            if not hasattr(self, 'export_combined') or self.export_combined:
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
                all_df.insert(0, "ID", all_df.index.map(lambda value: str(value).strip()))
                all_df = all_df.reset_index(drop=True)

                # 保存为CSV
                out_file = os.path.join(self.out_dir, f"radiomics_features_all{timestamp}.csv")
                all_df.to_csv(out_file, index=False, quoting=csv.QUOTE_MINIMAL)
                print(f"已将所有组学特征保存到 {out_file}")

        except Exception as e:
            logging.error(f"转换特征为CSV时出错: {str(e)}")
            
    def convert_to_json(self, features):
        """将特征转换为JSON格式"""
        try:
            import json
            
            # 获取时间戳字符串
            timestamp = ""
            if hasattr(self, 'add_timestamp') and self.add_timestamp:
                data = time.time()
                timeArray = time.localtime(data)
                timestamp = f"_{time.strftime('%Y%m%d_%H%M%S', timeArray)}"
                
            # 处理NumPy数据类型，使其可JSON序列化
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # 保存为JSON文件
            out_file = os.path.join(self.out_dir, f"radiomics_features_all{timestamp}.json")
            with open(out_file, 'w') as f:
                json.dump(features, f, default=convert_numpy_types, indent=2)
            
            print(f"已将所有组学特征保存到 {out_file}")
        except Exception as e:
            logging.error(f"转换特征为JSON时出错: {str(e)}")
            
    def convert_to_pickle(self, features):
        """将特征转换为Pickle格式"""
        try:
            import pickle
            
            # 获取时间戳字符串
            timestamp = ""
            if hasattr(self, 'add_timestamp') and self.add_timestamp:
                data = time.time()
                timeArray = time.localtime(data)
                timestamp = f"_{time.strftime('%Y%m%d_%H%M%S', timeArray)}"
                
            # 保存为Pickle文件
            out_file = os.path.join(self.out_dir, f"radiomics_features_all{timestamp}.pkl")
            with open(out_file, 'wb') as f:
                pickle.dump(features, f)
            
            print(f"已将所有组学特征保存到 {out_file}")
        except Exception as e:
            logging.error(f"转换特征为Pickle时出错: {str(e)}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='传统组学特征提取工具')

    # 配置文件参数
    parser.add_argument('--config_file', type=str,
                        help='配置文件路径')

    # 特征提取相关参数
    parser.add_argument('--params_file', type=str,
                        help='用于提取组学特征的参数文件')
    parser.add_argument('--images_folder', type=str,
                        help='影像根目录，应包含images和masks子目录')
    parser.add_argument('--out_dir', type=str,
                        help='输出目录')
    parser.add_argument('--n_processes', type=int,
                        help='使用的进程数，默认为CPU核心数的一半')

    return parser.parse_args()

if __name__ == "__main__":
    # 如果未提供命令行参数，使用默认值
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--config_file', './config/config_traditional_radiomics.yaml'
        ])

    # 解析命令行参数
    args = parse_arguments()

    # 创建特征提取器实例并运行
    start_time = time.time()
    
    extractor = TraditionalRadiomicsExtractor(
        config_file=args.config_file,
        params_file=args.params_file,
        images_folder=args.images_folder,
        out_dir=args.out_dir,
        n_processes=args.n_processes
    )
    extractor.extract_features()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

    # 命令行形式为（用sys真实参数）：
    # python traditional_radiomics_extractor.py --config_file ./config/config_traditional_radiomics.yaml
    # 或者：
    # python traditional_radiomics_extractor.py --params_file parameter.yaml --images_folder F:\work\workstation_b\dingHuYingXiang\the_third_training_202504\demo_data\datasets --out_dir F:\work\workstation_b\dingHuYingXiang\the_third_training_202504\demo_data\results --n_processes 4