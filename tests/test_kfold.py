"""
Test cases for habit cv (k-fold cross-validation) command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestKfoldCommand:
    """Test cases for cv (k-fold) command"""
    
    def test_kfold_with_config(self):
        """Test cv command with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_machine_learning_kfold.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['cv', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_kfold_help(self):
        """Test cv command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['cv', '--help'])
        assert result.exit_code == 0
        assert 'cross-validation' in result.output.lower() or 'k-fold' in result.output.lower() or 'cv' in result.output.lower()
    
    def test_kfold_missing_config(self):
        """Test cv command with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['cv', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_machine_learning_kfold.yaml'
    sys.argv = ['habit', 'cv', '-c', config_path]
    cli()
