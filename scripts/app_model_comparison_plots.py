import sys
import argparse
from habit.core.machine_learning.model_comparison import ModelComparison

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Model comparison and evaluation tool")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    return parser.parse_args()

if __name__ == "__main__":
    # 调试模式：如果没有提供命令行参数，使用默认配置文件
    if len(sys.argv) == 1:
        print("调试模式：使用默认配置文件 config_model_comparison.yaml")
        sys.argv = [sys.argv[0], "--config", "./config/config_model_comparison.yaml"]
    args = parse_args()
    mc = ModelComparison(args.config)
    mc.run() 

    # python scripts/app_model_comparison_plots.py --config ./config/config_model_comparison.yaml