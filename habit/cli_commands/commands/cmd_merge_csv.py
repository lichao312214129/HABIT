"""
CSV merge command implementation
Handles merging multiple CSV files horizontally based on index column
"""

import os
import sys
import click
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from habit.utils.log_utils import setup_logger, get_module_logger

logger = get_module_logger('cli.merge_csv')


def run_merge_csv(
    input_files: Tuple[str, ...],
    output_file: str,
    index_cols: Optional[List[str]] = None,
    separator: str = ',',
    encoding: str = 'utf-8',
    join_type: str = 'inner'
) -> None:
    """
    Merge multiple CSV/Excel files horizontally based on index column.
    
    Args:
        input_files: Tuple of file paths to merge (CSV or Excel)
        output_file: Path for the output merged CSV file
        index_cols: List of index column names. Can be:
                   - None: use first column for all files
                   - Single element: use same column for all files
                   - Multiple elements: one per file (must match file count)
        separator: CSV separator character
        encoding: File encoding
        join_type: Join type for merging ('inner' or 'outer')
    """
    file_logger = setup_logger('csv_merger')
    file_logger.info(f"Starting CSV merge process...")
    file_logger.info(f"Input files: {input_files}")
    file_logger.info(f"Output file: {output_file}")
    
    if len(input_files) < 2:
        click.secho("Error: At least 2 input files are required for merging", fg='red', err=True)
        sys.exit(1)
    
    # Validate and expand index_cols
    if index_cols is not None and len(index_cols) > 1:
        if len(index_cols) != len(input_files):
            click.secho(f"Error: Number of index columns ({len(index_cols)}) must match "
                       f"number of input files ({len(input_files)}) or be 1", fg='red', err=True)
            sys.exit(1)
    
    # Initialize merged dataframe
    merged_df = None
    processed_files = []
    
    click.echo(f"Starting to merge {len(input_files)} files...")
    
    try:
        for i, file_path in enumerate(input_files):
            click.echo(f"Processing file {i+1}/{len(input_files)}: {file_path}")
            
            # Determine which index column to use for this file
            if index_cols is None:
                current_index_col = None
            elif len(index_cols) == 1:
                current_index_col = index_cols[0]
            else:
                current_index_col = index_cols[i]
            
            # Check if file exists
            if not os.path.exists(file_path):
                file_logger.warning(f"File not found: {file_path}")
                click.echo(f"  Warning: File not found: {file_path}", err=True)
                continue
            
            file_logger.info(f"Reading file: {file_path}")
            
            # Determine file type and read
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.xlsx' or file_ext == '.xls':
                df = pd.read_excel(file_path)
            else:
                # Default to CSV
                df = pd.read_csv(file_path, sep=separator, encoding=encoding)
            
            if df.empty:
                file_logger.warning(f"Empty file: {file_path}")
                click.echo(f"  Warning: Empty file: {file_path}", err=True)
                continue
            
            # Set index column
            if current_index_col is None:
                df.set_index(df.columns[0], inplace=True)
                file_logger.info(f"Using first column '{df.index.name}' as index")
            else:
                if current_index_col in df.columns:
                    df.set_index(current_index_col, inplace=True)
                    file_logger.info(f"Using column '{current_index_col}' as index")
                else:
                    file_logger.warning(f"Index column '{current_index_col}' not found in {file_path}, using first column")
                    df.set_index(df.columns[0], inplace=True)
            
            # Merge dataframes
            if merged_df is None:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how=join_type)
            
            processed_files.append(os.path.basename(file_path))
            click.echo(f"  Successfully processed: {os.path.basename(file_path)}")
        
        click.echo("All files processed!")
        
        if merged_df is None:
            click.secho("Error: No valid files were processed", fg='red', err=True)
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            file_logger.info(f"Created output directory: {output_dir}")
        
        # Save merged dataframe
        merged_df.to_csv(output_file, sep=separator, encoding=encoding)
        
        file_logger.info(f"Successfully merged {len(processed_files)} files")
        file_logger.info(f"Output saved to: {output_file}")
        file_logger.info(f"Final dataframe shape: {merged_df.shape}")
        
        # Display summary
        click.echo(f"\n{'='*50}")
        click.echo("Merge Summary:")
        click.echo(f"  Files processed: {len(processed_files)}/{len(input_files)}")
        click.echo(f"  Output file: {output_file}")
        click.echo(f"  Final shape: {merged_df.shape}")
        click.echo(f"  Index column: {merged_df.index.name}")
        click.echo(f"  Join type: {join_type}")
        click.secho(f"\n✓ Merge completed successfully!", fg='green')
        
    except Exception as e:
        file_logger.error(f"Error during merge process: {str(e)}")
        click.secho(f"Error: {str(e)}", fg='red', err=True)
        sys.exit(1)
