"""
K-fold cross-validation command implementation
Performs k-fold cross-validation for model evaluation
"""

import sys
import click


def run_kfold(config_file: str) -> None:
    """
    Run K-fold cross-validation pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    click.echo(f"Starting K-fold cross-validation with config: {config_file}")
    
    try:
        # Import and run the k-fold script
        import sys
        old_argv = sys.argv
        sys.argv = ['app_kfold_cv.py', '--config', config_file]
        
        from scripts import app_kfold_cv
        app_kfold_cv.main()
        
        sys.argv = old_argv
        click.secho("✓ K-fold cross-validation completed successfully!", fg='green')
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

