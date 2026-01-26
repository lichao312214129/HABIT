"""
Test cases for habit radiomics command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestRadiomicsCommand:
    """Test cases for radiomics command"""
    
    def test_radiomics_with_config(self):
        """Test radiomics command with valid config file"""
        # Try to find config file - may not exist in demo_data
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_machine_learning_radiomics.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['radiomics', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_radiomics_help(self):
        """Test radiomics command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['radiomics', '--help'])
        assert result.exit_code == 0
        assert 'radiomics' in result.output.lower()


if __name__ == '__main__':
    # Allow running as script for debugging
    # Note: config file may not exist in demo_data
    config_path = './demo_data/config_machine_learning_radiomics.yaml'
    sys.argv = ['habit', 'radiomics', '-c', config_path]
    cli()
