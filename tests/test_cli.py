# test_cli.py
"""Unit tests for CLI commands"""
import sys
import pytest
from pathlib import Path
from click.testing import CliRunner

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from habit.cli import cli


class TestCLICommands:
    """Test CLI command interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command"""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'HABIT' in result.output or 'Usage' in result.output
    
    def test_preprocess_command_help(self):
        """Test preprocess command help"""
        result = self.runner.invoke(cli, ['preprocess', '--help'])
        assert result.exit_code == 0
    
    def test_habitat_command_help(self):
        """Test habitat command help"""
        result = self.runner.invoke(cli, ['habitat', '--help'])
        assert result.exit_code == 0
    
    def test_extract_features_command_help(self):
        """Test extract-features command help"""
        result = self.runner.invoke(cli, ['extract-features', '--help'])
        assert result.exit_code == 0
    
    def test_radiomics_command_help(self):
        """Test radiomics command help"""
        result = self.runner.invoke(cli, ['radiomics', '--help'])
        assert result.exit_code == 0
    
    def test_ml_command_help(self):
        """Test ml command help"""
        result = self.runner.invoke(cli, ['ml', '--help'])
        assert result.exit_code == 0
    
    def test_kfold_command_help(self):
        """Test kfold command help"""
        result = self.runner.invoke(cli, ['kfold', '--help'])
        assert result.exit_code == 0
    
    def test_icc_command_help(self):
        """Test icc command help"""
        result = self.runner.invoke(cli, ['icc', '--help'])
        assert result.exit_code == 0
    
    def test_test_retest_command_help(self):
        """Test test-retest command help"""
        result = self.runner.invoke(cli, ['test-retest', '--help'])
        assert result.exit_code == 0
    
    def test_compare_command_help(self):
        """Test compare command help"""
        result = self.runner.invoke(cli, ['compare', '--help'])
        assert result.exit_code == 0


class TestCLIConfigValidation:
    """Test CLI configuration file validation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file"""
        result = self.runner.invoke(cli, ['preprocess', '-c', 'nonexistent.yaml'])
        # Should fail with non-zero exit code
        assert result.exit_code != 0


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])

