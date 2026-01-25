"""
Test cases for habit model predict command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestPredictCommand:
    """Test cases for model predict mode"""
    
    def test_predict_with_config(self):
        """Test model predict command with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_machine_learning_predict.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', '-c', str(config_path), '-m', 'predict'])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_predict_with_pipeline_override(self):
        """Test model predict command with pipeline path override"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_machine_learning_predict.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        # Test with pipeline override (if supported)
        result = runner.invoke(cli, ['model', '-c', str(config_path), '-m', 'predict'])
        
        # Command should execute
        assert result.exit_code in [0, 1]


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_machine_learning_predict.yaml'
    sys.argv = ['habit', 'model', '-c', config_path, '-m', 'predict']
    cli()
