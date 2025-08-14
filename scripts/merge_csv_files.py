#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Files Merger Script
This script merges multiple CSV files horizontally based on their first column (index).
"""

import os
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Optional
import sys

# Add the habit package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from habit.utils.log_utils import setup_logger


def merge_csv_files(
    input_folder: str,
    csv_names: List[str],
    output_file: str,
    index_col: str = None,
    separator: str = ',',
    encoding: str = 'utf-8'
) -> None:
    """
    Merge multiple CSV files horizontally based on their first column (index).
    
    Parameters:
    -----------
    input_folder : str
        Path to the folder containing CSV files
    csv_names : List[str]
        List of CSV file names to merge (without .csv extension)
    output_file : str
        Path for the output merged CSV file
    index_col : str, optional
        Name of the index column. If None, uses the first column
    separator : str, default=','
        CSV separator character
    encoding : str, default='utf-8'
        File encoding
    """
    
    logger = setup_logger('csv_merger')
    logger.info(f"Starting CSV merge process...")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"CSV files to merge: {csv_names}")
    logger.info(f"Output file: {output_file}")
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Initialize merged dataframe
    merged_df = None
    processed_files = []
    
    print(f"Starting to merge {len(csv_names)} CSV files...")
    
    try:
        for i, csv_name in enumerate(csv_names):
            print(f"Processing file {i+1}/{len(csv_names)}: {csv_name}")
            
            # 支持文件名带或不带扩展名
            if csv_name.lower().endswith('.csv') or csv_name.lower().endswith('.xlsx'):
                file_base = os.path.splitext(csv_name)[0]
            else:
                file_base = csv_name
            
            # 优先查找csv，其次xlsx
            csv_path = os.path.join(input_folder, f"{file_base}.csv")
            xlsx_path = os.path.join(input_folder, f"{file_base}.xlsx")
            
            if os.path.exists(csv_path):
                file_path = csv_path
                file_type = 'csv'
            elif os.path.exists(xlsx_path):
                file_path = xlsx_path
                file_type = 'xlsx'
            else:
                logger.warning(f"File not found: {csv_path} or {xlsx_path}")
                continue
            
            logger.info(f"Reading file: {file_path}")
            # 根据文件类型选择读取方法
            if file_type == 'csv':
                df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            else:
                df = pd.read_excel(file_path)
            
            if df.empty:
                logger.warning(f"Empty file: {file_path}")
                continue
            
            # 设置索引列
            if index_col is None:
                df.set_index(df.columns[0], inplace=True)
                logger.info(f"Using first column '{df.index.name}' as index")
            else:
                if index_col in df.columns:
                    df.set_index(index_col, inplace=True)
                    logger.info(f"Using column '{index_col}' as index")
                else:
                    logger.warning(f"Index column '{index_col}' not found in {file_path}, using first column")
                    df.set_index(df.columns[0], inplace=True)
            
            # 列名前缀避免冲突（已去除，保持原始header不变）
            # if len(csv_names) > 1:
            #     df.columns = [f"{file_base}_{col}" for col in df.columns]
            
            # 合并
            if merged_df is None:
                merged_df = df
            else:
                # 由outer join改为inner join，只保留所有文件共有的index（交集）
                merged_df = merged_df.join(df, how='inner')
            
            processed_files.append(file_base)
            print(f"Successfully processed: {file_base}")
        
        print("All files processed successfully!")
        
        if merged_df is None:
            logger.error("No valid CSV files were processed")
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Save merged dataframe
        merged_df.to_csv(output_file, sep=separator, encoding=encoding)
        
        logger.info(f"Successfully merged {len(processed_files)} CSV files")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"Final dataframe shape: {merged_df.shape}")
        logger.info(f"Processed files: {processed_files}")
        
        # Display summary
        print(f"\nMerge Summary:")
        print(f"  Input folder: {input_folder}")
        print(f"  Files processed: {len(processed_files)}/{len(csv_names)}")
        print(f"  Output file: {output_file}")
        print(f"  Final shape: {merged_df.shape}")
        print(f"  Index column: {merged_df.index.name}")
        
    except Exception as e:
        logger.error(f"Error during merge process: {str(e)}")
        raise


def main():
    """Main function to handle command line arguments and execute the merge."""
    
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV files horizontally based on their first column (index)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge specific CSV files
  python ./habit/scipts/merge_csv_files.py -i "H:/results/features" -n "habitat_basic_features,msi_features,ith_scores" -o "merged_output.csv"
  
  # Merge with custom index column
  python merge_csv_files.py -i "H:/results/features" -n "file1,file2" -o "merged.csv" --index-col "subject_id"
  
  # Merge with different separator
  python merge_csv_files.py -i "H:/results/features" -n "file1,file2" -o "merged.csv" --separator ";"
        """
    )
    
    parser.add_argument(
        '-i', '--input-folder',
        type=str,
        required=True,
        help='Path to the folder containing CSV files'
    )
    
    parser.add_argument(
        '-n', '--csv-names',
        type=str,
        required=True,
        help='Comma-separated list of CSV file names (without .csv extension)'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        type=str,
        required=True,
        help='Path for the output merged CSV file'
    )
    
    parser.add_argument(
        '--index-col',
        type=str,
        default=None,
        help='Name of the column to use as index. If not specified, uses the first column'
    )
    
    parser.add_argument(
        '--separator',
        type=str,
        default=',',
        help='CSV separator character (default: comma)'
    )
    
    parser.add_argument(
        '--encoding',
        type=str,
        default='utf-8',
        help='File encoding (default: utf-8)'
    )
    
    args = parser.parse_args()
    
    # Parse CSV names
    csv_names = [name.strip() for name in args.csv_names.split(',')]
    
    # Execute merge
    try:
        merge_csv_files(
            input_folder=args.input_folder,
            csv_names=csv_names,
            output_file=args.output_file,
            index_col=args.index_col,
            separator=args.separator,
            encoding=args.encoding
        )
        print("\nMerge completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 