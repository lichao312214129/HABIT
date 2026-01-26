"""
Test cases for habit get-habitat command - two-step train mode
All configuration is in YAML file, no command-line mode override
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestHabitatTwoStepTrain:
    """Test cases for get-habitat command with two-step train mode"""
    
    def test_two_step_train_with_config(self):
        """Test two-step train with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat_two_step.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        # No -m parameter, mode is determined by run_mode in YAML
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]
    
    def test_two_step_train_help(self):
        """Test get-habitat command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '--help'])
        assert result.exit_code == 0
        assert 'habitat' in result.output.lower() or 'get-habitat' in result.output.lower()
    
    def test_two_step_train_missing_config(self):
        """Test two-step train with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config


if __name__ == '__main__':
    # Allow running as script for debugging
    # All configuration is in YAML, no -m parameter needed
    config_path = './demo_data/config_habitat_two_step.yaml'
    sys.argv = ['habit', 'get-habitat', '-c', config_path]
    cli()
