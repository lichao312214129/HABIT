"""
HABIT Command Line Interface
Main entry point for all HABIT (Habitat Analysis: Biomedical Imaging Toolkit) commands
"""

import click
from pathlib import Path


@click.group()
@click.version_option(version='0.1.0', prog_name='HABIT')
def cli():
    """
    HABIT - Habitat Analysis: Biomedical Imaging Toolkit
    
    A comprehensive toolkit for medical image analysis including:
    - Image preprocessing
    - Habitat analysis and clustering  
    - Feature extraction
    - Machine learning modeling
    - Statistical analysis
    """
    pass

@cli.command('preprocess')
@click.option('--config', '-c', 
              type=click.Path(exists=True),
              default=None,
              help='Path to configuration YAML file')
def preprocess(config):
    """Preprocess medical images (resampling, registration, normalization)"""
    from habit.cli_commands.commands.cmd_preprocess import run_preprocess
    run_preprocess(config)


@cli.command('get-habitat')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              help='Path to configuration YAML file')
@click.option('--debug', is_flag=True,
              help='Enable debug mode')
def get_habitat(config, debug):
    """Generate habitat maps from medical images"""
    from habit.cli_commands.commands.cmd_habitat import run_habitat
    run_habitat(config, debug)


@cli.command('extract')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              help='Path to configuration YAML file')
def extract(config):
    """Extract habitat features from clustered images"""
    from habit.cli_commands.commands.cmd_extract_features import run_extract_features
    run_extract_features(config)


@cli.command('model')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration YAML file')
@click.option('--mode', '-m',
              type=click.Choice(['train', 'predict']),
              default='train',
              help='Operation mode: train or predict')
@click.option('--model',
              type=click.Path(exists=True),
              help='Path to model package file (.pkl) for prediction')
@click.option('--data',
              type=click.Path(exists=True),
              help='Path to data file (.csv) for prediction')
@click.option('--output', '-o',
              type=click.Path(),
              help='Path to save prediction results')
@click.option('--model-name',
              help='Name of specific model to use for prediction')
@click.option('--evaluate/--no-evaluate',
              default=True,
              help='Whether to evaluate model performance')
def model(config, mode, model, data, output, model_name, evaluate):
    """Train or predict using machine learning models"""
    from habit.cli_commands.commands.cmd_ml import run_ml
    run_ml(config, mode, model, data, output, model_name, evaluate)


@cli.command('cv')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration YAML file')
def cv(config):
    """Run K-fold cross-validation for model evaluation"""
    from habit.cli_commands.commands.cmd_kfold import run_kfold
    run_kfold(config)


@cli.command('compare')
# 该装饰器为命令行工具添加一个名为 --config 或 -c 的参数选项
# 参数类型为存在的文件路径，且为必填项
# 用于指定配置文件（YAML 格式）的路径
@click.option('--config', '-c',
              type=click.Path(exists=True),     # 参数类型为存在的文件路径
              required=True,                    # 必须提供该参数
              help='Path to configuration YAML file')  # 参数描述，会显示在帮助信息中
def compare(config):
    """Generate model comparison plots and statistics"""
    from habit.cli_commands.commands.cmd_compare import run_compare
    run_compare(config)


@cli.command('icc')
@click.option('--config', '-c',
              type=click.Path(exists=True),  # 参数类型为存在的文件路径
              required=True,                 # 必须提供该参数
              help='Path to configuration YAML file')
def icc(config):
    """Perform ICC (Intraclass Correlation Coefficient) analysis"""
    from habit.cli_commands.commands.cmd_icc import run_icc
    run_icc(config)


@cli.command('radiomics')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              help='Path to configuration YAML file')
def radiomics(config):
    """Extract traditional radiomics features"""
    from habit.cli_commands.commands.cmd_radiomics import run_radiomics
    run_radiomics(config)


@cli.command('retest')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              help='Path to configuration YAML file')
def retest(config):
    """Perform test-retest reproducibility analysis"""
    from habit.cli_commands.commands.cmd_test_retest import run_test_retest
    run_test_retest(config)


@cli.command('dice')
@click.option('--input1', required=True, type=click.Path(exists=True), help='Path to first batch (root directory or config file)')
@click.option('--input2', required=True, type=click.Path(exists=True), help='Path to second batch (root directory or config file)')
@click.option('--output', default='dice_results.csv', show_default=True, help='Path to save results CSV')
@click.option('--mask-keyword', default='masks', show_default=True, help='Keyword for mask folder (used if input is a directory)')
@click.option('--label-id', default=1, show_default=True, help='Label ID to calculate Dice for')
def dice(input1, input2, output, mask_keyword, label_id):
    """Calculate Dice coefficient between two batches of images"""
    from habit.utils.dice_calculator import run_dice_calculation
    run_dice_calculation(input1, input2, output, mask_keyword, label_id)


@cli.command('dicom-info')
@click.option('--input', '-i', 
              required=True,
              type=click.Path(exists=True),
              help='Path to DICOM directory, file, or YAML config file')
@click.option('--tags', '-t',
              help='Comma-separated list of DICOM tags to extract (e.g., "PatientName,StudyDate,Modality")')
@click.option('--output', '-o',
              type=click.Path(),
              help='Path to save results (CSV, Excel, or JSON)')
@click.option('--format', '-f',
              type=click.Choice(['csv', 'excel', 'json'], case_sensitive=False),
              default='csv',
              show_default=True,
              help='Output format for saved results')
@click.option('--recursive/--no-recursive',
              default=True,
              show_default=True,
              help='Search recursively in subdirectories')
@click.option('--list-tags', is_flag=True,
              help='List available tags instead of extracting information')
@click.option('--num-samples',
              type=int,
              default=1,
              show_default=True,
              help='Number of files to sample when listing tags')
@click.option('--group-by-series/--no-group-by-series',
              default=True,
              show_default=True,
              help='Group files by SeriesInstanceUID and only read one file per series. '
                   'If disabled, read all files individually.')
@click.option('--one-file-per-folder', is_flag=True,
              help='Only read one DICOM file per folder to speed up scanning. '
                   'Useful when each folder contains exactly one series.')
@click.option('--dicom-extensions',
              type=str,
              default=None,
              help='Comma-separated list of DICOM file extensions to recognize '
                   '(e.g., ".dcm,.dicom,.ima"). Only used with --one-file-per-folder. '
                   'Default: .dcm,.dicom')
@click.option('--include-no-extension', is_flag=True,
              help='Also check files without extensions by reading DICOM magic bytes. '
                   'Only used with --one-file-per-folder. Useful for some medical devices '
                   'that produce DICOM files without file extensions.')
@click.option('--num-workers', '-j',
              type=int,
              default=None,
              help='Number of worker threads for parallel processing. '
                   'Default: min(32, cpu_count + 4). Set to 1 to disable parallelism.')
@click.option('--max-depth', '-d',
              type=int,
              default=None,
              help='Maximum recursion depth for directory traversal. '
                   'Only used with --one-file-per-folder. '
                   '0=root only, 1=root+1 level, etc. Default: unlimited. '
                   'Example: For DICOM structure patient/study/series/, use -d 3')
def dicom_info(input, tags, output, format, recursive, list_tags, num_samples, 
               group_by_series, one_file_per_folder, dicom_extensions, include_no_extension,
               num_workers, max_depth):
    """Extract and view DICOM file information"""
    from habit.cli_commands.commands.cmd_dicom_info import run_dicom_info
    
    # Parse tags if provided
    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
    
    # Parse DICOM extensions if provided
    ext_set = None
    if dicom_extensions:
        ext_set = {ext.strip() for ext in dicom_extensions.split(',') if ext.strip()}
    
    # Warn about parameter conflicts
    if not one_file_per_folder:
        if max_depth is not None:
            click.echo("Warning: --max-depth is only effective with --one-file-per-folder", err=True)
        if dicom_extensions is not None:
            click.echo("Warning: --dicom-extensions is only effective with --one-file-per-folder", err=True)
        if include_no_extension:
            click.echo("Warning: --include-no-extension is only effective with --one-file-per-folder", err=True)
    
    run_dicom_info(
        input_path=input,
        tags=tag_list,
        recursive=recursive,
        output=output,
        output_format=format.lower(),
        list_tags=list_tags,
        num_samples=num_samples,
        group_by_series=group_by_series,
        one_file_per_folder=one_file_per_folder,
        dicom_extensions=ext_set,
        include_no_extension=include_no_extension,
        num_workers=num_workers,
        max_depth=max_depth
    )


@cli.command('merge-csv')
@click.argument('input_files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('-o', '--output',
              type=click.Path(),
              required=True,
              help='Path for the output merged CSV file')
@click.option('--index-col', '-c',
              type=str,
              default=None,
              help='Index column name(s). Can be: single name for all files, '
                   'or comma-separated names (one per file). Example: "id" or "PatientID,subject_id"')
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
def merge_csv(input_files, output, index_col, separator, encoding, join_type):
    """
    Merge multiple CSV/Excel files horizontally based on index column.
    
    \b
    INPUT_FILES: Two or more CSV/Excel files to merge
    
    \b
    Examples:
      # Same index column for all files
      habit merge-csv file1.csv file2.csv -o merged.csv --index-col subject_id
      
      # Different index column for each file
      habit merge-csv a.csv b.csv -o merged.csv --index-col "PatientID,subject_id"
      
      # Use first column as index (default)
      habit merge-csv file1.csv file2.csv -o merged.csv
    """
    from habit.cli_commands.commands.cmd_merge_csv import run_merge_csv
    
    # Parse index columns if provided
    index_cols = None
    if index_col:
        index_cols = [col.strip() for col in index_col.split(',')]
    
    run_merge_csv(
        input_files=input_files,
        output_file=output,
        index_cols=index_cols,
        separator=separator,
        encoding=encoding,
        join_type=join_type
    )


if __name__ == '__main__':
    cli()
