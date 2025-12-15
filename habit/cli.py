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
              default='./config/config_image_preprocessing.yaml',
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
@click.option('--config', '-c',
              type=click.Path(exists=True),
              required=True,
              help='Path to configuration YAML file')
def compare(config):
    """Generate model comparison plots and statistics"""
    from habit.cli_commands.commands.cmd_compare import run_compare
    run_compare(config)


@cli.command('icc')
@click.option('--config', '-c',
              type=click.Path(exists=True),
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


if __name__ == '__main__':
    cli()

