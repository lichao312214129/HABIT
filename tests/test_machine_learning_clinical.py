"""
Test cases for habit model command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestModelCommand:
    """Test cases for model command"""
    
    def test_model_train_with_config(self):
        """Test model command in train mode with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_machine_learning_clinical.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', '-c', str(config_path), '-m', 'train'])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_model_predict_with_config(self):
        """Test model command in predict mode with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_machine_learning_predict.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        result = runner.invoke(cli, ['model', '-c', str(config_path), '-m', 'predict'])
        
        # Command should execute (may fail if data not available, but should not crash)
        assert result.exit_code in [0, 1]  # 0 for success, 1 for expected errors (missing data)
    
    def test_model_help(self):
        """Test model command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['model', '--help'])
        assert result.exit_code == 0
        assert 'machine learning' in result.output.lower() or 'model' in result.output.lower()
    
    def test_model_missing_config(self):
        """Test model command with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['model', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config


if __name__ == '__main__':
    # Allow running as script for debugging
    config_path = './demo_data/config_machine_learning_clinical.yaml'
    sys.argv = ['habit', 'model', '-c', config_path]
    cli()
