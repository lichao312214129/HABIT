"""
Command-line interface for extracting habitat features from medical images
This module provides functionality for extracting various types of features from habitat maps,
including traditional radiomics, non-radiomics features, whole habitat features, 
individual habitat features, and MSI (Multi-Scale Image) features.
"""

import json
import os
import sys
import yaml
import argparse
import logging
import time
from pathlib import Path
from habit.core.habitat_analysis import HabitatFeatureExtractor
from habit.utils.io_utils import load_config, setup_logging


def parse_args():
    """
    Parse command line arguments for habitat feature extraction
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Habitat Feature Extraction Tool')
    
    # Configuration file argument
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Parameter files
    parser.add_argument('--params_file_of_non_habitat', type=str,
                        help='Parameter file for extracting radiomics features from original images')
    parser.add_argument('--params_file_of_habitat', type=str,
                        help='Parameter file for extracting radiomics features from habitat maps')
    
    # Data directories
    parser.add_argument('--raw_img_folder', type=str,
                        help='Root directory containing original images')
    parser.add_argument('--habitats_map_folder', type=str,
                        help='Root directory containing habitat maps')
    parser.add_argument('--out_dir', type=str,
                        help='Output directory for results')
    
    # Feature extraction parameters
    parser.add_argument('--n_processes', type=int, default=4,
                        help='Number of processes to use for parallel processing')
    parser.add_argument('--habitat_pattern', type=str, default='*_habitats.nrrd',
                        choices=['*_habitats.nrrd', '*_habitats_remapped.nrrd'],
                        help='Pattern for matching habitat files')
    
    # Feature type parameters
    parser.add_argument('--feature_types', nargs='+', type=str,
                        default=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi'],
                        choices=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi'],
                        help='Types of features to extract')
    parser.add_argument('--n_habitats', type=int, default=0,
                        help='Number of habitats to process (0 for auto detection)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['extract', 'parse', 'both'],
                        help='Operation mode: extract features only, parse features only, or both')
    
    # Debug parameter
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()


def validate_config(config):
    """
    Validate configuration parameters
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_fields = [
        'params_file_of_non_habitat',
        'params_file_of_habitat',
        'raw_img_folder',
        'habitats_map_folder',
        'out_dir',
        'feature_types'
    ]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in config or not config[field]]
    
    if missing_fields:
        logging.error(f"Missing required fields: {', '.join(missing_fields)}")
        return False
    
    # Check if files and directories exist
    file_fields = {
        'params_file_of_non_habitat': "original image radiomics parameters file",
        'params_file_of_habitat': "habitat map radiomics parameters file"
    }
    
    for field, desc in file_fields.items():
        if not os.path.exists(config[field]):
            logging.error(f"{desc} does not exist: {config[field]}")
            return False
    
    directory_fields = {
        'raw_img_folder': "original image directory",
        'habitats_map_folder': "habitat map directory"
    }
    
    for field, desc in directory_fields.items():
        if not os.path.exists(config[field]):
            logging.error(f"{desc} does not exist: {config[field]}")
            return False
    
    # Check if out_dir is writable
    try:
        if not os.path.exists(config['out_dir']):
            os.makedirs(config['out_dir'])
        test_file = os.path.join(config['out_dir'], '.test_write_access')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        logging.error(f"Output directory is not writable: {config['out_dir']}, error: {str(e)}")
        return False
    
    return True


def print_config_summary(config):
    """
    Print a summary of the configuration
    
    Args:
        config (dict): Configuration dictionary
    """
    logging.info("==== 生境特征提取参数 ====")
    logging.info(f"原始图像Radiomics参数文件: {config.get('params_file_of_non_habitat')}")
    logging.info(f"生境图Radiomics参数文件: {config.get('params_file_of_habitat')}")
    logging.info(f"原始图像目录: {config.get('raw_img_folder')}")
    logging.info(f"生境图目录: {config.get('habitats_map_folder')}")
    logging.info(f"输出目录: {config.get('out_dir')}")
    logging.info(f"生境文件匹配模式: {config.get('habitat_pattern')}")
    logging.info(f"需要提取的特征类型: {', '.join(config.get('feature_types', []))}")
    logging.info(f"操作模式: {config.get('mode', 'both')}")
    logging.info(f"进程数量: {config.get('n_processes', 4)}")
    
    if 'n_habitats' in config and config['n_habitats'] > 0:
        logging.info(f"生境数量: {config['n_habitats']}")
    else:
        logging.info("生境数量: 自动检测")
        
    logging.info(f"调试模式: {config.get('debug', False)}")
    logging.info("=========================")


def run_extractor(config):
    """
    Run the habitat feature extractor with the provided configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        # Create feature extractor instance
        logging.info("正在创建特征提取器实例...")
        print("正在创建特征提取器实例...")
        
        extractor = HabitatFeatureExtractor(
            params_file_of_non_habitat=config['params_file_of_non_habitat'],
            params_file_of_habitat=config['params_file_of_habitat'],
            raw_img_folder=config['raw_img_folder'],
            habitats_map_folder=config['habitats_map_folder'],
            out_dir=config['out_dir'],
            n_processes=config.get('n_processes', 4),
            habitat_pattern=config.get('habitat_pattern', '*_habitats.nrrd')
        )
        
        # Run feature extractor
        logging.info("开始执行特征提取...")
        print("开始执行特征提取...")
        
        start_time = time.time()
        extractor.run(
            feature_types=config['feature_types'],
            n_habitats=config.get('n_habitats'),
            mode=config.get('mode', 'both')
        )
        end_time = time.time()
        
        # Report execution time
        execution_time = end_time - start_time
        logging.info(f"特征提取完成，总耗时: {execution_time:.2f} 秒")
        print(f"特征提取完成，总耗时: {execution_time:.2f} 秒")
            
        return True
        
    except Exception as e:
        error_msg = f"特征提取过程中发生错误: {str(e)}"
        logging.error(error_msg)
        print(error_msg)
        
        if config.get('debug', False):
            import traceback
            traceback.print_exc()
        return False


def main() -> None:
    """Main function to run habitat feature extraction"""
    # Parse command line arguments
    args = parse_args()
    
    # Determine configuration source
    config = None
    
    # Try to build configuration directly from command line arguments
    if (args.params_file_of_non_habitat and args.params_file_of_habitat and
        args.raw_img_folder and args.habitats_map_folder and args.out_dir):
        
        config = {
            'params_file_of_non_habitat': args.params_file_of_non_habitat,
            'params_file_of_habitat': args.params_file_of_habitat,
            'raw_img_folder': args.raw_img_folder,
            'habitats_map_folder': args.habitats_map_folder,
            'out_dir': args.out_dir,
            'n_processes': args.n_processes,
            'habitat_pattern': args.habitat_pattern,
            'feature_types': args.feature_types,
            'mode': args.mode,
            'debug': args.debug
        }
        
        if args.n_habitats:
            config['n_habitats'] = args.n_habitats
            
    # If a configuration file is provided, load from configuration file
    elif args.config:
        try:
            config = load_config(args.config)
            # Command line arguments will override configuration file
            for key, value in vars(args).items():
                if value is not None and key != 'config':
                    config[key] = value
        except Exception as e:
            print(f"配置文件加载错误: {str(e)}")
            return
    
    # If not enough configuration information
    if not config:
        print("错误: 请提供必要的参数或配置文件")
        print("使用方法示例:")
        print("python app_extracting_habitat_features.py --params_file_of_non_habitat parameter.yaml --params_file_of_habitat parameter_habitat.yaml --raw_img_folder raw_images --habitats_map_folder habitat_maps --out_dir results")
        print("或者:")
        print("python app_extracting_habitat_features.py --config config.yaml")
        return
    
    # Set up logging
    setup_logging(config['out_dir'], debug=config.get('debug', False))
    
    # Validate configuration
    if not validate_config(config):
        logging.error("配置验证失败，程序退出")
        return
    
    # Print configuration summary
    print_config_summary(config)
    
    # Run feature extractor
    success = run_extractor(config)
    
    if success:
        logging.info("生境特征提取已完成")
        print("生境特征提取已完成")
    else:
        logging.error("生境特征提取失败")
        print("生境特征提取失败")


if __name__ == "__main__":
    # # 开发模式：当没有命令行参数时，使用默认值
    if len(sys.argv) == 1:
        print("调试模式：使用默认参数")
        sys.argv.extend([
            '--params_file_of_non_habitat', 'parameter.yaml',
            '--params_file_of_habitat', 'parameter_habitat.yaml',
            '--raw_img_folder', 'G:\\lessThan50AndNoMacrovascularInvasion_structured',
            '--habitats_map_folder', 'F:\\work\\research\\radiomics_TLSs\\data\\results_416',
            '--out_dir', 'F:\\work\\research\\radiomics_TLSs\\data\\results_416\\parsed_features',
            '--n_processes', '4',
            '--habitat_pattern', '*_habitats.nrrd',
            '--feature_types', 'traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi',
            '--mode', 'both',
            '--debug'
        ])
    main()
