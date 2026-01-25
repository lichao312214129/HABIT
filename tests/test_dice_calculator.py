"""
Test cases for habit dice command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestDiceCommand:
    """Test cases for dice command"""
    
    def test_dice_help(self):
        """Test dice command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['dice', '--help'])
        assert result.exit_code == 0
        assert 'dice' in result.output.lower() or 'coefficient' in result.output.lower()
    
    def test_dice_missing_inputs(self):
        """Test dice command with missing required inputs"""
        runner = CliRunner()
        
        # Missing input1
        result = runner.invoke(cli, ['dice', '--input2', 'test', '--output', 'out.csv'])
        assert result.exit_code != 0
        
        # Missing input2
        result = runner.invoke(cli, ['dice', '--input1', 'test', '--output', 'out.csv'])
        assert result.exit_code != 0
    
    def test_dice_with_config_file(self):
        """Test dice command with config file (if supported)"""
        # This test assumes dice can accept config files
        # If not, this test can be skipped or modified
        runner = CliRunner()
        
        # Try with a non-existent config to test error handling
        result = runner.invoke(cli, [
            'dice',
            '--input1', 'nonexistent1.yaml',
            '--input2', 'nonexistent2.yaml',
            '--output', 'dice_results.csv'
        ])
        
        # Should fail gracefully
        assert result.exit_code != 0


if __name__ == '__main__':
    # Allow running as script for debugging
    # Note: Update paths as needed
    sys.argv = [
        'habit', 'dice',
        '--input1', 'input1.yaml',
        '--input2', 'input2.yaml',
        '--output', 'dice_results.csv'
    ]
    cli()
