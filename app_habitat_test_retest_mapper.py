"""
Command-line interface for mapping habitat labels between test-retest data
This module provides functionality for aligning habitat labels between 
test and retest scans, ensuring consistent interpretation of habitat analysis.
"""

import argparse
import sys
import os
from machine_learning.feature_selectors.icc.habitat_test_retest_mapper import (
    find_habitat_mapping, batch_process_files, setup_logger
)

def main() -> None:
    """
    Main function to run the habitat test-retest mapper
    
    This function:
    1. Parses command line arguments
    2. Sets up logging
    3. Finds habitat mapping between test and retest data
    4. Processes and remaps habitat files
    """
    parser = argparse.ArgumentParser(description='Map habitat labels between test-retest data')
    parser.add_argument('--test-habitat-table', type=str, required=True,
                      help='测试组的habitat特征表格文件路径 (CSV或Excel格式)')
    parser.add_argument('--retest-habitat-table', type=str, required=True,
                      help='重测组的habitat特征表格文件路径 (CSV或Excel格式)')
    parser.add_argument('--features', type=str, nargs='+',
                      help='用于计算相似性的特征名称列表，如果不指定则使用全部特征')
    parser.add_argument('--similarity-method', type=str, default='pearson',
                      choices=['pearson', 'spearman', 'kendall', 'euclidean', 
                               'cosine', 'manhattan', 'chebyshev'],
                      help='相似度计算方法')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='包含重测组NRRD文件的目录')
    parser.add_argument('--out-dir', type=str, required=True,
                      help='处理后文件的输出目录')
    parser.add_argument('--processes', type=int,
                      default=4,
                      help='使用的进程数 (默认: 4)')
    parser.add_argument('--debug', action='store_true',
                      help='启用调试日志')
    
    args = parser.parse_args()
    setup_logger(args.debug)
    
    try:
        # Create output directory
        os.makedirs(args.out_dir, exist_ok=True)
        
        # Find habitat mapping
        print("计算测试和重测数据之间的habitat映射...")
        habitat_mapping = find_habitat_mapping(
            args.test_habitat_table, args.retest_habitat_table, 
            args.features, args.similarity_method
        )
        
        # Print mapping
        print("Habitat映射:")
        for retest_label, test_label in habitat_mapping.items():
            print(f"  重测 Habitat {retest_label} -> 测试 Habitat {test_label}")
        
        # Process files
        print(f"使用{args.processes}个进程开始处理文件...")
        batch_process_files(
            args.input_dir, 
            habitat_mapping, 
            args.out_dir, 
            args.processes
        )
        
        print("处理完成")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 