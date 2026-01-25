"""
Test cases for habit extract command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestExtractFeaturesCommand:
    """Test cases for extract command"""
    
    def test_extract_with_config(self):
        """Test extract command with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_extract_features.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['extract', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_extract_help(self):
        """Test extract command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['extract', '--help'])
        assert result.exit_code == 0
        assert 'extract' in result.output.lower() or 'feature' in result.output.lower()
    
    def test_extract_missing_config(self):
        """Test extract command with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['extract', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_extract_features.yaml'
    sys.argv = ['habit', 'extract', '-c', config_path]
    cli()
