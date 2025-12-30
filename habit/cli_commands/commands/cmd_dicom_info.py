"""
DICOM information extraction command implementation
Handles reading and saving DICOM file information
"""

import sys
import os
import click
import logging
from pathlib import Path
from typing import Optional, List, Set
from habit.utils.dicom_utils import batch_read_dicom_info, list_available_tags, PYDICOM_AVAILABLE
from habit.utils.log_utils import get_module_logger

logger = get_module_logger('cli.dicom_info')


def run_dicom_info(input_path: str,
                   tags: Optional[List[str]] = None,
                   recursive: bool = True,
                   output: Optional[str] = None,
                   output_format: str = 'csv',
                   list_tags: bool = False,
                   num_samples: int = 1,
                   group_by_series: bool = True,
                   one_file_per_folder: bool = False,
                   dicom_extensions: Optional[Set[str]] = None,
                   include_no_extension: bool = False,
                   num_workers: Optional[int] = None,
                   max_depth: Optional[int] = None) -> None:
    """
    Run DICOM information extraction
    
    Args:
        input_path: Path to DICOM directory, file, or YAML config file
        tags: List of DICOM tags to extract (comma-separated or list)
        recursive: Whether to search recursively in subdirectories
        output: Optional path to save results
        output_format: Format to save results ('csv', 'excel', 'json')
        list_tags: Whether to list available tags instead of extracting
        num_samples: Number of files to sample when listing tags
        group_by_series: If True, group files by SeriesInstanceUID and only read one file per series.
                        If False, read all files. Default is True.
        one_file_per_folder: If True, only take one DICOM file per folder to speed up scanning.
        dicom_extensions: Set of valid DICOM file extensions (e.g., {'.dcm', '.dicom', '.ima'}).
                         Only used when one_file_per_folder=True. Default: {'.dcm', '.dicom'}
        include_no_extension: If True, also check files without extensions by reading DICOM magic bytes.
                             Only used when one_file_per_folder=True.
        num_workers: Number of worker threads for parallel processing.
                    Default: min(32, cpu_count + 4). Set to 1 to disable parallelism.
        max_depth: Maximum recursion depth for directory traversal.
                  Only used when one_file_per_folder=True.
    """
    try:
        # Check if pydicom is available
        if not PYDICOM_AVAILABLE:
            click.echo("Error: pydicom is not installed. Install it with: pip install pydicom", err=True)
            sys.exit(1)
        
        # Validate input path
        if not os.path.exists(input_path):
            logger.error(f"Input path does not exist: {input_path}")
            click.echo(f"Error: Input path does not exist: {input_path}", err=True)
            sys.exit(1)
        
        # If list_tags is True, just list available tags
        if list_tags:
            click.echo(f"Listing available tags from {input_path}...")
            available_tags = list_available_tags(input_path, num_samples=num_samples)
            
            if not available_tags:
                click.echo("No DICOM tags found or no DICOM files found.")
                return
            
            click.echo(f"\nFound {len(available_tags)} available tag(s):\n")
            for tag in available_tags:
                click.echo(f"  - {tag}")
            
            # Save to file if output is specified
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write("Available DICOM Tags\n")
                    f.write("=" * 50 + "\n\n")
                    for tag in available_tags:
                        f.write(f"{tag}\n")
                click.echo(f"\nTag list saved to {output_path}")
            
            return
        
        # Parse tags if provided as string
        tag_list = None
        if tags:
            if isinstance(tags, str):
                # Split by comma and strip whitespace
                tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
            elif isinstance(tags, list):
                tag_list = tags
        
        # Extract DICOM information
        click.echo(f"Reading DICOM information from {input_path}...")
        if tag_list:
            click.echo(f"Extracting tags: {', '.join(tag_list)}")
        
        df = batch_read_dicom_info(
            input_path=input_path,
            tags=tag_list,
            recursive=recursive,
            output_file=output,
            output_format=output_format,
            group_by_series=group_by_series,
            one_file_per_folder=one_file_per_folder,
            dicom_extensions=dicom_extensions,
            include_no_extension=include_no_extension,
            num_workers=num_workers,
            max_depth=max_depth
        )
        
        if df.empty:
            click.echo("No DICOM information could be extracted.", err=True)
            sys.exit(1)
        
        # Display summary
        click.echo(f"\n✓ Successfully extracted information from {len(df)} DICOM file(s)")
        click.echo(f"  Columns: {', '.join(df.columns.tolist())}")
        
        # Display preview
        if len(df) <= 10:
            click.echo("\nPreview of extracted data:")
            click.echo(df.to_string(index=False))
        else:
            click.echo("\nPreview of first 5 rows:")
            click.echo(df.head(5).to_string(index=False))
            click.echo(f"\n... and {len(df) - 5} more row(s)")
        
        if output:
            click.secho(f"\n✓ Results saved to {output}", fg='green')
        else:
            click.echo("\nNote: Use --output to save results to a file")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during DICOM information extraction: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

