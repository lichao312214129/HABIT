"""
Test-retest reproducibility analysis command implementation
Analyzes test-retest reproducibility of habitat features
"""

import sys
import click


def run_test_retest(config_file: str) -> None:
    """
    Run test-retest reproducibility analysis
    
    Args:
        config_file (str): Path to configuration YAML file
    """
    if not config_file:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    click.echo(f"Starting test-retest analysis with config: {config_file}")
    
    try:
        # Import and run the test-retest script
        import sys
        old_argv = sys.argv
        sys.argv = ['app_habitat_test_retest_mapper.py', '--config', config_file]
        
        from scripts import app_habitat_test_retest_mapper
        app_habitat_test_retest_mapper.main()
        
        sys.argv = old_argv
        click.secho("✓ Test-retest analysis completed successfully!", fg='green')
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

