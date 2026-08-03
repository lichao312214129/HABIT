# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
HABIT Command Line Interface
Main entry point for all HABIT (Habitat Analysis: Biomedical Imaging Toolkit) commands
"""

from functools import lru_cache

import click
from pathlib import Path
from typing import Optional

from habit.commands.common import ensure_cli_stdio
from habit.utils.project_urls import docs_page


@lru_cache(maxsize=1)
def _package_version() -> str:
    """Return installed package version with a safe fallback."""
    try:
        from importlib.metadata import version
        return version("HABIT")
    except Exception:  # noqa: BLE001
        from habit import __version__
        return __version__


def config_option(**overrides: object):
    """
    Standard ``--config / -c`` option for pipeline commands.

    Args:
        **overrides: Extra keyword arguments forwarded to ``click.option``.
    """
    params = {
        "type": click.Path(exists=True),
        "required": True,
        "help": "Path to configuration YAML file",
    }
    params.update(overrides)
    return click.option("--config", "-c", **params)


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option(version=_package_version(), prog_name='HABIT')
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
    ensure_cli_stdio()

@cli.command('check-config')
@config_option(help='Path to configuration YAML to check (does not run the pipeline)')
@click.option(
    '--workflow', '-w',
    type=click.Choice([
        'preprocess', 'habitat', 'extract', 'radiomics',
        'model', 'cv', 'compare', 'icc', 'retest', 'sort-dicom',
    ], case_sensitive=False),
    default=None,
    help='Workflow schema to validate against (optional; guessed from path when omitted)',
)
@click.option(
    '--syntax-only',
    is_flag=True,
    help='Check YAML syntax only (for manifests and PyRadiomics parameter presets)',
)
def check_config(
    config: str,
    workflow: Optional[str],
    syntax_only: bool,
) -> None:
    """Check YAML syntax and optional schema without running a pipeline"""
    from habit.commands.cmd_check_config import run_check_config
    run_check_config(config, workflow, syntax_only)


@cli.command('migrate-config')
@config_option(help='Path to the v0 configuration YAML to upgrade')
@click.option(
    '--output', '-o',
    type=click.Path(),
    default=None,
    help='Destination v1 YAML (default: <name>.v1.yaml next to the source)',
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Print the diff without writing a file',
)
@click.option(
    '--workflow', '-w',
    type=click.Choice([
        'preprocess', 'habitat', 'extract', 'radiomics',
        'model', 'cv', 'compare', 'icc', 'retest', 'sort-dicom',
    ], case_sensitive=False),
    default=None,
    help='Workflow alias (optional; guessed from path when omitted)',
)
def migrate_config(
    config: str,
    output: Optional[str],
    dry_run: bool,
    workflow: Optional[str],
) -> None:
    """Upgrade a v0 YAML config to the v1 spec/data/policy/output layout"""
    from habit.commands.cmd_migrate_config import run_migrate_config
    run_migrate_config(config, output, dry_run, workflow)


@cli.command('preprocess')
@config_option()
def preprocess(config):
    """Preprocess medical images (resampling, registration, normalization)"""
    from habit.commands.cmd_preprocess import run_preprocess
    run_preprocess(config)


@cli.command('sort-dicom')
@config_option(help='Path to DicomSortConfig YAML (standalone dcm2niix sort)')
def sort_dicom(config):
    """Sort/rename DICOM files with dcm2niix (separate from batch preprocessing)"""
    from habit.commands.cmd_sort_dicom import run_sort_dicom
    run_sort_dicom(config)


@cli.command('get-habitat')
@config_option()
@click.option('--mode', '-m',
              type=click.Choice(['train', 'predict']),
              default=None,
              help='Override run mode in the YAML (train or predict)')
@click.option('--pipeline',
              default=None,
              type=click.Path(exists=True),
              help='Override pipeline path for predict mode')
@click.option('--debug', is_flag=True,
              help='Enable debug mode')
@click.option('--resume', is_flag=True,
              help='Resume train run from individual-level checkpoint')
def get_habitat(
    config: str,
    mode: Optional[str],
    pipeline: Optional[str],
    debug: bool,
    resume: bool,
) -> None:
    """Generate habitat maps from medical images"""
    from habit.commands.cmd_habitat import run_habitat
    run_habitat(config, debug, mode, pipeline, resume)


@cli.command('extract')
@config_option()
def extract(config):
    """Extract habitat features from clustered images"""
    from habit.commands.cmd_extract_features import run_extract_features
    run_extract_features(config)


@cli.command('model')
@config_option()
@click.option('--mode', '-m',
              type=click.Choice(['train', 'predict']),
              default=None,
              help='Override run mode in the YAML (train or predict)')
def model(config, mode):
    """
    Train or predict using machine learning models.
    
    All parameters (including model path, data path, output directory) 
    must be specified in the configuration file. When --mode is omitted,
    YAML run_mode is used (default train if the key is absent).
    """
    from habit.commands.cmd_ml import run_ml
    run_ml(config, mode)


@cli.command('cv')
@config_option()
def cv(config):
    """Run K-fold cross-validation for model evaluation"""
    from habit.commands.cmd_ml import run_kfold
    run_kfold(config)


@cli.command('compare')
@config_option()
def compare(config):
    f"""
    Model comparison — generate plots and statistics across multiple models.

    Compare performance of two or more machine learning models side by side.
    Produces publication-quality evaluation plots and statistical summaries,
    including ROC, calibration, decision-curve (DCA), and precision-recall curves.

    Input:
    - Prediction CSV files per model (true labels and predicted probabilities)
    - Per model: file path, model name, and column mappings

    Output (under output_dir):
    - ROC curves (roc_curves.pdf)
    - Decision curves (decision_curves.pdf)
    - Calibration curves (calibration_curves.pdf)
    - Precision-recall curves (precision_recall_curves.pdf)
    - DeLong test results (delong_results.json)
    - Merged predictions (combined_predictions.csv)

  Config example:
    \b
    output_dir: ./results/comparison/
    files_config:
      - path: ./results/model1_predictions.csv
        model_name: model1
        subject_id_col: PatientID
        label_col: label
        prob_col: probability
      - path: ./results/model2_predictions.csv
        model_name: model2
        subject_id_col: PatientID
        label_col: label
        prob_col: probability

  Example:
    \b
    habit compare --config config_model_comparison.yaml

  Documentation: {docs_page('how_to/compare_models.html')}
    """
    from habit.commands.cmd_compare import run_compare
    run_compare(config)


@cli.command('icc')
@config_option()
def icc(config):
    """Perform ICC (Intraclass Correlation Coefficient) analysis"""
    from habit.commands.cmd_icc import run_icc
    run_icc(config)


@cli.command('radiomics')
@config_option()
def radiomics(config):
    """Extract traditional radiomics features"""
    from habit.commands.cmd_radiomics import run_radiomics
    run_radiomics(config)


@cli.command('retest')
@config_option()
def retest(config):
    """Perform test-retest reproducibility analysis"""
    from habit.commands.cmd_test_retest import run_test_retest
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
    from habit.commands.cmd_dicom_info import run_dicom_info
    
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
    from habit.commands.cmd_merge_csv import run_merge_csv
    
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


@cli.command('gui')
@click.option('--host', default='127.0.0.1', show_default=True, help='Host to bind the Web GUI server to')
@click.option('--port', '-p', default=8501, show_default=True, help='Local port for the Web GUI')
@click.option('--no-browser', is_flag=True, help='Do not open a browser tab automatically')
def gui(host, port, no_browser):
    """Launch the HABIT Web GUI (under development; CLI/YAML recommended for now)"""
    from habit.commands.cmd_gui import run_next_gui_server
    run_next_gui_server(host=host, port=port, open_browser=not no_browser)


if __name__ == '__main__':
    cli()
