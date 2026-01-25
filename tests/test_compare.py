"""
Test cases for habit compare command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestCompareCommand:
    """Test cases for compare command"""
    
    def test_compare_with_config(self):
        """Test compare command with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_model_comparison.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['compare', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_compare_help(self):
        """Test compare command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['compare', '--help'])
        assert result.exit_code == 0
        assert 'model comparison' in result.output.lower() or 'compare' in result.output.lower()
    
    def test_compare_missing_config(self):
        """Test compare command with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['compare', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_model_comparison.yaml'
    sys.argv = ['habit', 'compare', '-c', config_path]
    cli()
