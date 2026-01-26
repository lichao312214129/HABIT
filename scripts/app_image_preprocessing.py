import sys
import os
import platform
import traceback
import logging
import multiprocessing
import argparse
from habit.core.preprocessing.image_processor_pipeline import BatchProcessor
from habit.utils.log_utils import get_module_logger

# Get module logger
logger = get_module_logger('scripts.app_preprocessing')

def parse_args():
    """
    Parse command line arguments using argparse
    
    Returns:
        str: Path to the configuration file
    """
    parser = argparse.ArgumentParser(description='图像预处理程序')
    parser.add_argument('-c', '--config', 
                        type=str, 
                        default="./config/config_image_preprocessing.yaml",
                        help='配置文件路径')
    return parser.parse_args()

def main(config_path):
    try:
        # Log system information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check config file
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return
        
        logger.info(f"Using configuration file: {config_path}")
        
        # Initialize processor
        try:
            # Force spawn method on Windows for multiprocessing
            if platform.system() == 'Windows':
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn' on Windows")
            
            processor = BatchProcessor(config_path=config_path)
            logger.info("Successfully initialized BatchProcessor")
        except Exception as e:
            logger.error(f"Failed to initialize BatchProcessor: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Process data
        try:
            logger.info("Starting batch processing")
            processor.process_batch()
            logger.info("Batch processing completed")
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Uncaught error during execution: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) == 1:
        config_path = "./config/config_image_preprocessing.yaml"
    else:
        # 检查是否使用了argparse格式的参数
        if sys.argv[1].startswith('-'):
            args = parse_args()
            config_path = args.config
        else:
            config_path = sys.argv[1]
    main(config_path)

    # python scripts/app_image_preprocessing.py --config ./config/config_image_preprocessing.yaml