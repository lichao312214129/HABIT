#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Files Merger Script
This script merges multiple CSV/Excel files horizontally based on their index column.

Note: The recommended way to use this functionality is through the habit CLI:
    habit merge-csv file1.csv file2.csv -o merged.csv
"""

import os
import pandas as pd
import click
from pathlib import Path
from typing import List, Optional, Tuple
import sys

# Add the habit package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from habit.utils.log_utils import setup_logger


def merge_csv_files(
    input_files: Tuple[str, ...],
    output_file: str,
    index_cols: Optional[List[str]] = None,
    separator: str = ',',
    encoding: str = 'utf-8',
    join_type: str = 'inner'
) -> None:
    """
    Merge multiple CSV/Excel files horizontally based on index column.
    
    Parameters:
    -----------
    input_files : Tuple[str, ...]
        Tuple of file paths to merge (CSV or Excel)
    output_file : str
        Path for the output merged CSV file
    index_cols : List[str], optional
        List of index column names. Can be:
        - None: use first column for all files
        - Single element: use same column for all files
        - Multiple elements: one per file (must match file count)
    separator : str, default=','
        CSV separator character
    encoding : str, default='utf-8'
        File encoding
    join_type : str, default='inner'
        Join type: 'inner' (only common rows) or 'outer' (all rows)
    """
    
    logger = setup_logger('csv_merger')
    logger.info(f"Starting CSV merge process...")
    logger.info(f"Input files: {input_files}")
    logger.info(f"Output file: {output_file}")
    
    if len(input_files) < 2:
        raise ValueError("At least 2 input files are required for merging")
    
    # Validate and expand index_cols
    if index_cols is not None and len(index_cols) > 1:
        if len(index_cols) != len(input_files):
            raise ValueError(f"Number of index columns ({len(index_cols)}) must match "
                           f"number of input files ({len(input_files)}) or be 1")
    
    # Initialize merged dataframe
    merged_df = None
    processed_files = []
    
    print(f"Starting to merge {len(input_files)} files...")
    
    try:
        for i, file_path in enumerate(input_files):
            print(f"Processing file {i+1}/{len(input_files)}: {file_path}")
            
            # Determine which index column to use for this file
            if index_cols is None:
                current_index_col = None
            elif len(index_cols) == 1:
                current_index_col = index_cols[0]
            else:
                current_index_col = index_cols[i]
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                print(f"  Warning: File not found: {file_path}")
                continue
            
            logger.info(f"Reading file: {file_path}")
            
            # Determine file type and read
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.xlsx' or file_ext == '.xls':
                df = pd.read_excel(file_path)
            else:
                # Default to CSV
                df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            
            if df.empty:
                logger.warning(f"Empty file: {file_path}")
                print(f"  Warning: Empty file: {file_path}")
                continue
            
            # Set index column
            if current_index_col is None:
                df.set_index(df.columns[0], inplace=True)
                logger.info(f"Using first column '{df.index.name}' as index")
            else:
                if current_index_col in df.columns:
                    df.set_index(current_index_col, inplace=True)
                    logger.info(f"Using column '{current_index_col}' as index")
                else:
                    logger.warning(f"Index column '{current_index_col}' not found in {file_path}, using first column")
                    df.set_index(df.columns[0], inplace=True)
            
            # Merge dataframes
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how=join_type)
            
            processed_files.append(os.path.basename(file_path))
            print(f"  Successfully processed: {os.path.basename(file_path)}")
        
        print("All files processed!")
        
        if merged_df is None:
            logger.error("No valid files were processed")
            raise ValueError("No valid files were processed")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Save merged dataframe
        merged_df.to_csv(output_file, sep=separator, encoding=encoding)
        
        logger.info(f"Successfully merged {len(processed_files)} files")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"Final dataframe shape: {merged_df.shape}")
        
        # Display summary
        print(f"\nMerge Summary:")
        print(f"  Files processed: {len(processed_files)}/{len(input_files)}")
        print(f"  Output file: {output_file}")
        print(f"  Final shape: {merged_df.shape}")
        print(f"  Index column: {merged_df.index.name}")
        print(f"  Join type: {join_type}")
        
    except Exception as e:
        logger.error(f"Error during merge process: {str(e)}")
        raise


@click.command()
@click.argument('input_files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-o', '--output',
              type=click.Path(),
              required=True,
              help='Path for the output merged CSV file')
@click.option('--index-col', '-c',
              type=str,
              default=None,
              help='Index column name(s). Single name for all files, or comma-separated (one per file)')
@click.option('--separator',
              type=str,
              default=',',
              show_default=True,
              help='CSV separator character')
@click.option('--encoding',
              type=str,
              default='utf-8',
              show_default=True,
              help='File encoding')
@click.option('--join', 'join_type',
              type=click.Choice(['inner', 'outer'], case_sensitive=False),
              default='inner',
              show_default=True,
              help='Join type: inner (only common rows) or outer (all rows)')
def main(input_files: Tuple[str, ...], output: str, index_col: Optional[str], 
         separator: str, encoding: str, join_type: str):
    """
    Merge multiple CSV/Excel files horizontally based on index column.
    
    \b
    INPUT_FILES: Two or more CSV/Excel files to merge
    
    \b
    Examples:
      # Same index column for all files
      python merge_csv_files.py file1.csv file2.csv -o merged.csv --index-col subject_id
      
      # Different index column for each file
      python merge_csv_files.py a.csv b.csv -o merged.csv --index-col "PatientID,subject_id"
    
    \b
    Note: The recommended way is to use the habit CLI:
      habit merge-csv file1.csv file2.csv -o merged.csv
    """
    # Parse index columns if provided
    index_cols = None
    if index_col:
        index_cols = [col.strip() for col in index_col.split(',')]
    
    try:
        merge_csv_files(
            input_files=input_files,
            output_file=output,
            index_cols=index_cols,
            separator=separator,
            encoding=encoding,
            join_type=join_type
        )
        click.secho("\nMerge completed successfully!", fg='green')
        
    except Exception as e:
        click.secho(f"\nError: {str(e)}", fg='red', err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
