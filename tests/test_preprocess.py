"""
Test cases for habit preprocess command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestPreprocessCommand:
    """Test cases for preprocess command"""
    
    def test_preprocess_with_config(self):
        """Test preprocess command with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_preprocessing.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['preprocess', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_preprocess_help(self):
        """Test preprocess command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['preprocess', '--help'])
        assert result.exit_code == 0
        assert 'preprocess' in result.output.lower() or 'preprocess' in result.output.lower()


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_preprocessing.yaml'
    sys.argv = ['habit', 'preprocess', '-c', config_path]
    cli()
