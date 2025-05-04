#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像预处理示例脚本

此脚本展示了如何使用基于MONAI的图像预处理流水线进行多模态医学图像预处理，
包括加载图像、重采样、配准和强度归一化等步骤。
"""

import os
import yaml
import logging
import torch
import numpy as np
from pathlib import Path

from habit.core.preprocessing.image_processor_monai import ImagePreprocessorMonai

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 配置数据目录
data_root = "./data"
output_dir = "./processed_data"

def run_preprocessing_from_config():
    """使用配置文件运行预处理流水线"""
    
    # 加载配置文件
    config_path = "examples/preprocessing_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("加载的预处理配置:")
    print("-" * 50)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("-" * 50)
    
    # 创建预处理器
    preprocessor = ImagePreprocessorMonai(
        config=config,
        root_folder=data_root,
        out_folder=output_dir,
        n_processes=4,
        verbose=True
    )
    
    # 运行预处理
    print("\n开始运行预处理流水线...")
    dataset = preprocessor.run()
    
    # 如果有数据被处理
    if dataset and len(dataset) > 0:
        # 创建数据加载器
        dataloader = preprocessor.create_dataloader(
            dataset=dataset,
            batch_size=1,
            num_workers=4
        )
        
        # 示例：访问并显示第一个样本
        sample = dataset[0]
        print("\n预处理完成，处理后的第一个样本包含以下数据:")
        for key, value in sample.items():
            if torch.is_tensor(value):
                print(f"{key}: 形状 = {value.shape}, 范围 = [{value.min():.2f}, {value.max():.2f}]")
    else:
        print("\n没有找到可处理的数据。请确保数据目录结构正确。")

def run_custom_preprocessing():
    """自定义预处理流水线示例"""
    
    # 自定义预处理配置
    config = {
        # 基本设置
        "modalities": ["pre_contrast", "LAP", "PVP", "delay_3min"],
        "auto_load": True,  # 自动加载图像
        "save_output": True,  # 保存处理后的图像
        "cache_rate": 0.5,  # 缓存比例
        
        # 步骤1: 统一空间采样
        "resample": {
            "transform": "Spacingd",
            "keys": ["pre_contrast", "LAP", "PVP", "delay_3min"],
            "pixdim": [1.5, 1.5, 2.0],
            "mode": "bilinear",
            "align_corners": False
        },
        
        # 步骤2: 配准到pre_contrast阶段
        "register_lap": {
            "custom": "registration",
            "keys": ["LAP"],
            "fixed_key": "pre_contrast",
            "moving_key": "LAP",
            "transform_type": "rigid"
        },
        "register_pvp": {
            "custom": "registration",
            "keys": ["PVP"],
            "fixed_key": "pre_contrast",
            "moving_key": "PVP",
            "transform_type": "rigid"
        },
        "register_delay": {
            "custom": "registration",
            "keys": ["delay_3min"],
            "fixed_key": "pre_contrast",
            "moving_key": "delay_3min",
            "transform_type": "rigid"
        },
        
        # 步骤3: CT值归一化
        "normalize_images": {
            "transform": "ScaleIntensityRanged",
            "keys": ["pre_contrast", "LAP", "PVP", "delay_3min"],
            "a_min": -200,
            "a_max": 300,
            "b_min": 0.0,
            "b_max": 1.0,
            "clip": True
        }
    }
    
    # 创建预处理器
    preprocessor = ImagePreprocessorMonai(
        config=config,
        root_folder=data_root,
        out_folder=output_dir,
        n_processes=4,
        verbose=True
    )
    
    # 运行预处理
    print("\n开始运行自定义预处理流水线...")
    dataset = preprocessor.run()
    
    # 如果有数据被处理
    if dataset and len(dataset) > 0:
        print("\n自定义预处理完成!")
    else:
        print("\n没有找到可处理的数据。请确保数据目录结构正确。")

def process_specific_subjects():
    """处理特定的病例"""
    
    # 加载配置文件
    config_path = "examples/preprocessing_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 指定要处理的病例
    subjects_to_process = ["subject1", "subject3", "subject5"]
    
    # 创建预处理器
    preprocessor = ImagePreprocessorMonai(
        config=config,
        root_folder=data_root,
        out_folder=output_dir,
        n_processes=4,
        verbose=True
    )
    
    # 只处理指定的病例
    print(f"\n开始处理特定病例: {', '.join(subjects_to_process)}...")
    subset_dataset = preprocessor.run(subjects=subjects_to_process)
    
    # 如果有数据被处理
    if subset_dataset and len(subset_dataset) > 0:
        print(f"\n成功处理了 {len(subset_dataset)} 个病例!")
    else:
        print("\n没有找到指定的病例。请确保病例ID正确。")

if __name__ == "__main__":
    # 检查数据目录
    if not os.path.exists(data_root):
        print(f"警告: 数据目录 {data_root} 不存在。创建示例目录...")
        os.makedirs(data_root, exist_ok=True)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行预处理示例
    print("=" * 80)
    print("图像预处理示例")
    print("=" * 80)
    
    # 从配置文件运行预处理
    run_preprocessing_from_config()
    
    # 运行自定义预处理
    # run_custom_preprocessing()
    
    # 处理特定的病例
    # process_specific_subjects()
    
    print("\n预处理示例执行完成！")
