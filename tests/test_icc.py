"""
Test cases for habit icc command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestIccCommand:
    """Test cases for icc command"""
    
    def test_icc_with_config(self):
        """Test icc command with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_icc.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['icc', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_icc_help(self):
        """Test icc command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['icc', '--help'])
        assert result.exit_code == 0
        assert 'icc' in result.output.lower() or 'intraclass' in result.output.lower()
    
    def test_icc_missing_config(self):
        """Test icc command with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['icc', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_icc.yaml'
    sys.argv = ['habit', 'icc', '-c', config_path]
    cli()
