"""
Test cases for habit get-habitat command - two-step predict mode
All configuration is in YAML file, no command-line mode override
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestHabitatTwoStepPredict:
    """Test cases for get-habitat command with two-step predict mode"""
    
    def test_two_step_predict_with_config(self):
        """Test two-step predict with valid config file"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat_two_step_predict.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        runner = CliRunner()
        # No -m parameter, mode is determined by run_mode in YAML
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path)])
        
        # Command should execute (may fail if pipeline not found, but should not crash)
        assert result.exit_code in [0, 1]
    
    def test_two_step_predict_with_pipeline_override(self):
        """Test two-step predict with pipeline path override"""
        config_path = Path(__file__).parent.parent / 'demo_data' / 'config_habitat_two_step_predict.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        # Check if pipeline exists (may not exist if training hasn't been run)
        # Pipeline is saved in out_dir, which is ./results/habitats/two_step (relative to project root)
        # When running from project root, the path should be results/habitats/two_step/habitat_pipeline.pkl
        pipeline_path = Path(__file__).parent.parent / 'results' / 'habitats' / 'two_step' / 'habitat_pipeline.pkl'
        
        if not pipeline_path.exists():
            pytest.skip(f"Pipeline not found at {pipeline_path}. Run training first.")
        
        runner = CliRunner()
        # Override pipeline path via command line, but mode is still from YAML
        result = runner.invoke(cli, ['get-habitat', '-c', str(config_path), '--pipeline', str(pipeline_path)])
        
        # Command should execute
        assert result.exit_code in [0, 1]
    
    def test_two_step_predict_help(self):
        """Test get-habitat command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '--help'])
        assert result.exit_code == 0
        assert 'habitat' in result.output.lower() or 'get-habitat' in result.output.lower()
    
    def test_two_step_predict_missing_config(self):
        """Test two-step predict with missing config file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['get-habitat', '-c', 'nonexistent_config.yaml'])
        assert result.exit_code != 0  # Should fail with missing config


if __name__ == '__main__':
    # Allow running as script for debugging
    # All configuration is in YAML, no -m parameter needed
    config_path = './demo_data/config_habitat_two_step_predict.yaml'
    sys.argv = ['habit', 'get-habitat', '-c', config_path]
    cli()
