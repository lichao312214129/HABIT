"""
Model comparison command implementation
Generates comparison plots and statistics for different models
"""

import sys
import click


def run_compare(config_file: str) -> None:
    """
    Run model comparison analysis
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    click.echo(f"Starting model comparison with config: {config_file}")
    
    try:
        # Import and run the comparison script
        import sys
        old_argv = sys.argv
        sys.argv = ['app_model_comparison_plots.py', '--config', config_file]
        
        from scripts import app_model_comparison_plots
        app_model_comparison_plots.main()
        
        sys.argv = old_argv
        click.secho("✓ Model comparison completed successfully!", fg='green')
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

