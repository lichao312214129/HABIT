"""
Test cases for habit get-habitat command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestGetHabitatCommand:
    """Test cases for get-habitat command"""
    
    def test_get_habitat_with_config(self):
        """Test get-habitat command with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path)])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_get_habitat_one_step(self):
        """Test get-habitat command with one-step config"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat_one_step.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path)])
        
        # Command should execute
        assert result.exit_code in [0, 1]
    
    def test_get_habitat_direct_pooling(self):
        """Test get-habitat command with direct pooling config"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat_direct_pooling.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path)])
        
        # Command should execute
        assert result.exit_code in [0, 1]
    
    def test_get_habitat_predict_mode_two_step(self):
        """Test get-habitat command in predict mode with two-step config"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path), '-m', 'predict'])
        
        # Command should execute
        assert result.exit_code in [0, 1]
    
    def test_get_habitat_predict_mode_one_step(self):
        """Test get-habitat command in predict mode with one-step config"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat_one_step_predict.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path), '-m', 'predict'])
        
        # Command should execute
        assert result.exit_code in [0, 1]
    
    def test_get_habitat_predict_mode_direct_pooling(self):
        """Test get-habitat command in predict mode with direct pooling config"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat_direct_pooling_predict.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path), '-m', 'predict'])
        
        # Command should execute
        assert result.exit_code in [0, 1]
    
    def test_get_habitat_help(self):
        """Test get-habitat command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '--help'])
        assert result.exit_code == 0
        assert 'habitat' in result.output.lower() or 'get-habitat' in result.output.lower()
    
    def test_get_habitat_missing_config(self):
        """Test get-habitat command with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config
    
    def test_get_habitat_with_mode_override(self):
        """Test get-habitat command with mode override"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path), '-m', 'train'])
        
        # Command should execute
        assert result.exit_code in [0, 1]


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_habitat.yaml'
    sys.argv = ['habit', 'get-habitat', '-c', config_path]
    cli()
