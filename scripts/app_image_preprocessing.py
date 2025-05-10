import sys
import os
import platform
import traceback
import logging
import multiprocessing
from pathlib import Path
from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

# 设置日志记录
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing_debug.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app_preprocessing")

def main():
    try:
        # 记录系统信息
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # 检查配置文件
        config_path = "./config/config_kmeans.yaml"
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
        logger.error(f"程序执行过程中发生未捕获的错误: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()