import sys
import os
import platform
import traceback
import logging
import multiprocessing
import argparse
import yaml
from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

# 创建全局logger对象
logger = logging.getLogger("app_preprocessing")

def setup_logging(config_path):
    """
    Setup logging configuration
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        logger: 配置好的logger对象
    """
    global logger
    
    try:
        # Read config file to get output directory, 指定UTF-8编码
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        out_dir = config.get('out_dir', os.getcwd())
        
        # Create output directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)
        
        # Setup logging
        log_file = os.path.join(out_dir, "preprocessing_debug.log")
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger("app_preprocessing")
        logger.info(f"Log file saved to: {log_file}")
        return logger
    except Exception as e:
        # 如果在设置日志时出错，创建一个基本的日志配置
        print(f"设置日志出错: {str(e)}")
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger("app_preprocessing")
        logger.error(f"设置日志时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return logger

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
    global logger
    
    try:
        # Setup logging
        logger = setup_logging(config_path)
        
        # 记录系统信息
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # 检查配置文件
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return
        
        logger.info(f"使用配置文件: {config_path}")
        
        # 初始化处理器
        try:
            # 在Windows上强制使用spawn方法创建进程
            if platform.system() == 'Windows':
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("在Windows上设置多进程启动方法为'spawn'")
            
            processor = BatchProcessor(config_path=config_path)
            logger.info("成功初始化BatchProcessor")
        except Exception as e:
            logger.error(f"初始化BatchProcessor失败: {e}")
            logger.error(traceback.format_exc())
            return
        
        # 处理数据
        try:
            logger.info("开始处理批次数据")
            processor.process_batch()
            logger.info("批次处理完成")
        except Exception as e:
            logger.error(f"批次处理过程中发生错误: {e}")
            logger.error(traceback.format_exc())
    except Exception as e:
        # 这里使用全局logger，避免UnboundLocalError
        if logger is not None:
            logger.error(f"程序执行过程中发生未捕获的错误: {e}")
            logger.error(traceback.format_exc())
        else:
            # 如果logger还没有初始化
            print(f"程序执行过程中发生未捕获的错误: {e}")
            print(traceback.format_exc())

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