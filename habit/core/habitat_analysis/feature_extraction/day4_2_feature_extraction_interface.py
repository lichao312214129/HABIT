#!/usr/bin/env python
"""
特征提取接口
用于调用加密后的feature_extraction模块
"""


import yaml
import argparse
import sys

def extract_features(params_file_of_non_habitat=None,
                     params_file_of_habitat=None,
                     raw_img_folder=None, 
                     habitats_map_folder=None, 
                     out_dir=None,
                     n_processes=None,
                     habitat_pattern=None,
                     feature_types=None, 
                     n_habitats=None, 
                     mode='both'):
    """
    调用加密后的feature_extraction模块进行特征提取
    
    参数:
        params_file_of_non_habitat: 用于从原始影像提取组学特征的参数文件
        params_file_of_habitat: 用于从生境影像提取组学特征的参数文件
        raw_img_folder: 原始影像根目录
        habitats_map_folder: 生境图根目录
        out_dir: 输出目录
        n_processes: 进程数
        habitat_pattern: 生境文件匹配模式
        feature_types: 特征类型列表
        n_habitats: 生境数量
        mode: 提取模式 ('both', 'extract', 'parse')
    """
    # 导入加密后的模块
    from habitat_analysis.feature_extraction.extractor import HabitatFeatureExtractor
    
    # 创建特征提取器
    extractor = HabitatFeatureExtractor(
        params_file_of_non_habitat=params_file_of_non_habitat,
        params_file_of_habitat=params_file_of_habitat,
        raw_img_folder=raw_img_folder,
        habitats_map_folder=habitats_map_folder,
        out_dir=out_dir,
        n_processes=n_processes,
        habitat_pattern=habitat_pattern
    )
    
    # 运行特征提取
    extractor.run(feature_types=feature_types, n_habitats=n_habitats, mode=mode)
    
    print(f"特征提取完成，结果保存在 {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='特征提取接口')
    
    parser.add_argument('--params_non_habitat', type=str, help='用于从原始影像提取组学特征的参数文件')
    parser.add_argument('--params_habitat', type=str, help='用于从生境影像提取组学特征的参数文件')
    parser.add_argument('--raw_img_folder', type=str, required=True, help='原始影像根目录')
    parser.add_argument('--habitats_map_folder', type=str, required=True, help='生境图根目录')
    parser.add_argument('--out_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--n_processes', type=int, help='进程数')
    parser.add_argument('--habitat_pattern', type=str, help='生境文件匹配模式')
    parser.add_argument('--feature_types', type=str, help='特征类型列表，用逗号分隔，如"traditional,non_radiomics,whole_habitat,each_habitat"')
    parser.add_argument('--mode', type=str, default='both', choices=['both', 'extract', 'parse'], help='提取模式')
    parser.add_argument('--config', type=str, help='配置文件路径，可以用来替代命令行参数')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，则从配置文件加载参数
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从配置中更新参数
        for key, value in config.items():
            if key == 'feature_types' and isinstance(value, list):
                # 如果feature_types在配置文件中是列表
                setattr(args, key, value)
            elif value is not None:
                setattr(args, key, value)
    
    # 准备特征类型列表
    feature_types = None
    if hasattr(args, 'feature_types') and args.feature_types:
        if isinstance(args.feature_types, list):
            feature_types = args.feature_types
        elif isinstance(args.feature_types, str):
            feature_types = [ft.strip() for ft in args.feature_types.split(',')]
    
    # 调用特征提取函数
    extract_features(
        params_file_of_non_habitat=args.params_non_habitat,
        params_file_of_habitat=args.params_habitat,
        raw_img_folder=args.raw_img_folder,
        habitats_map_folder=args.habitats_map_folder,
        out_dir=args.out_dir,
        n_processes=args.n_processes,
        habitat_pattern=args.habitat_pattern,
        feature_types=feature_types,
        mode=args.mode
    )


if __name__ == "__main__":
    # 如果未提供命令行参数，使用默认值
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--params_non_habitat', 'parameter.yaml',
            '--params_habitat', 'parameter_habitat.yaml',
            '--raw_img_folder', 'G:\\lessThan50AndNoMacrovascularInvasion_structured',
            '--habitats_map_folder', 'F:\\work\\research\\radiomics_TLSs\\data\\results_kmeans_0402_4clusters_test',
            '--out_dir', 'F:\\work\\workstation_b\\dingHuYingXiang\\_the_third_training_202504\\demo_data\\results',
            '--n_processes', '4',
            '--habitat_pattern', '*_habitats.nrrd',
            '--feature_types', 'traditional, non_radiomics, whole_habitat, each_habitat, msi',
            '--mode', 'both'
        ])
    main() 

