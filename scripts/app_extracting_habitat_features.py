"""
Command-line interface for extracting habitat features from medical images
This module provides functionality for extracting various types of features from habitat maps,
including traditional radiomics, non-radiomics features, whole habitat features, 
individual habitat features, and MSI (Multi-Scale Image) features.
"""

import os
import sys
import argparse
import logging
import time
from habit.core.habitat_analysis.habitat_feature_extraction.new_extractor import HabitatFeatureExtractor
from habit.utils.io_utils import load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for habitat feature extraction
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing only config file path
    """
    parser = argparse.ArgumentParser(description='Habitat Feature Extraction Tool')
    
    # Configuration file argument - now the only required argument
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to YAML configuration file with extraction parameters')
    
    # Debug parameter
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    return parser.parse_args()

def validate_config(config: dict) -> bool:
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

def print_config_summary(config: dict) -> None:
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
    logging.info(f"生境文件匹配模式: {config.get('habitat_pattern', '*_habitats.nrrd')}")
    logging.info(f"需要提取的特征类型: {', '.join(config.get('feature_types', []))}")
    logging.info(f"进程数量: {config.get('n_processes', 4)}")
    
    if 'n_habitats' in config and config['n_habitats'] is not None:
        logging.info(f"生境数量: {config['n_habitats']}")
    else:
        logging.info("生境数量: 自动检测")
        
    logging.info(f"调试模式: {config.get('debug', False)}")
    logging.info("=========================")

def run_extractor(config: dict) -> bool:
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
        
        start_time = time.time()
        extractor.run(
            feature_types=config['feature_types'],
            n_habitats=config.get('n_habitats', 0)
        )
        end_time = time.time()
        
        # Report execution time
        execution_time = end_time - start_time
        logging.info(f"特征提取完成，总耗时: {execution_time:.2f} 秒")
            
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
    
    try:
        # Load configuration from YAML file
        config = load_config(args.config)
        
        # Add debug flag from command line
        if args.debug:
            config['debug'] = True
            
    except Exception as e:
        logging.error(f"配置文件加载错误: {str(e)}")
        logging.error("请提供有效的YAML配置文件, 例如:")
        logging.error("python app_extracting_habitat_features.py --config config/extract_features_config.yaml")
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
    else:
        logging.error("生境特征提取失败")


if __name__ == "__main__":
    # 开发模式：当没有命令行参数时，使用默认配置文件
    if len(sys.argv) == 1:
        print("调试模式：使用默认配置文件")
        sys.argv.extend([
            '--config', './config/config_extract_features.yaml'
        ])
    main()

    # python scripts/app_extracting_habitat_features.py --config ./config/config_extract_features.yaml
