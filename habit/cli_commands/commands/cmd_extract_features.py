"""
Feature extraction command implementation
Extracts habitat features from clustered images
"""

import sys
import click


def run_extract_features(config_file: str) -> None:
    """
    Run habitat feature extraction pipeline
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    # This will import the corresponding script functionality
    # TODO: Implement feature extraction command
    # For now, we'll import and run the existing script
    
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    click.echo(f"Starting habitat feature extraction with config: {config_file}")
    
    try:
        # Import and run the feature extraction script
        import sys
        old_argv = sys.argv
        sys.argv = ['app_extracting_habitat_features.py', '--config', config_file]
        
        from scripts import app_extracting_habitat_features
        app_extracting_habitat_features.main()
        
        sys.argv = old_argv
        click.secho("✓ Feature extraction completed successfully!", fg='green')
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

