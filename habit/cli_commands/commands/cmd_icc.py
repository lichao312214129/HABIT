"""
ICC analysis command implementation
Performs Intraclass Correlation Coefficient analysis
"""

import sys
import click


def run_icc(config_file: str) -> None:
    """
    Run ICC (Intraclass Correlation Coefficient) analysis
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    click.echo(f"Starting ICC analysis with config: {config_file}")
    
    try:
        # Import and run the ICC analysis script
        import sys
        old_argv = sys.argv
        sys.argv = ['app_icc_analysis.py', '--config', config_file]
        
        from scripts import app_icc_analysis
        app_icc_analysis.main()
        
        sys.argv = old_argv
        click.secho("✓ ICC analysis completed successfully!", fg='green')
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

