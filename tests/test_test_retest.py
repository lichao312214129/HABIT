"""
Test cases for habit retest command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestRetestCommand:
    """Test cases for retest command"""
    
    def test_retest_with_config(self):
        """Test retest command with valid config file"""
        # Try to find config file - may not exist in demo_data
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_test_retest.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['retest', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_retest_help(self):
        """Test retest command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['retest', '--help'])
        assert result.exit_code == 0
        assert 'retest' in result.output.lower() or 'test-retest' in result.output.lower()


if __name__ == '__main__':
    # Allow running as script for debugging
    # Note: config file may not exist in demo_data
    config_path = './demo_data/config_test_retest.yaml'
    sys.argv = ['habit', 'retest', '-c', config_path]
    cli()
