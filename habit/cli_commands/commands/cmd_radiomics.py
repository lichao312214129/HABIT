"""
Traditional radiomics extraction command implementation
Extracts traditional radiomics features from medical images
"""

import sys
import click


def run_radiomics(config_file: str) -> None:
    """
    Run traditional radiomics feature extraction
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    click.echo(f"Starting traditional radiomics extraction with config: {config_file}")
    
    try:
        # Import and run the radiomics extraction script
        import sys
        old_argv = sys.argv
        sys.argv = ['app_traditional_radiomics_extractor.py', '--config', config_file]
        
        from scripts import app_traditional_radiomics_extractor
        app_traditional_radiomics_extractor.main()
        
        sys.argv = old_argv
        click.secho("✓ Radiomics extraction completed successfully!", fg='green')
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

