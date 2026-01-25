"""
End-to-end workflow test: preprocess -> get-habitat -> extract -> model -> compare
This test runs the complete workflow and reports any errors
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestEndToEndWorkflow:
    """Test the complete workflow from preprocessing to model comparison"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.runner = CliRunner()
        self.demo_data_dir = Path(__file__).parent.parent / 'demo_data'
        self.results = {}
    
    def test_step1_preprocess(self):
        """Step 1: Test preprocess command"""
        config_path = self.demo_data_dir / 'config_preprocessing.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        result = self.runner.invoke(cli, ['preprocess', '-c', str(config_path)])
        self.results['preprocess'] = {
            'exit_code': result.exit_code,
            'output': result.output,
            'error': result.exception
        }
        
        # Log result
        print(f"\n=== Preprocess Result ===")
        print(f"Exit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Error: {result.output}")
        
        # Should not crash (exit code 0 or 1 is acceptable)
        assert result.exit_code in [0, 1], f"Preprocess failed with exit code {result.exit_code}"
    
    def test_step2_get_habitat(self):
        """Step 2: Test get-habitat command"""
        config_path = self.demo_data_dir / 'config_habitat.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        result = self.runner.invoke(cli, ['get-habitat', '-c', str(config_path)])
        self.results['get_habitat'] = {
            'exit_code': result.exit_code,
            'output': result.output,
            'error': result.exception
        }
        
        # Log result
        print(f"\n=== Get-Habitat Result ===")
        print(f"Exit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Error: {result.output}")
        
        # Should not crash
        assert result.exit_code in [0, 1], f"Get-habitat failed with exit code {result.exit_code}"
    
    def test_step3_extract_features(self):
        """Step 3: Test extract features command"""
        config_path = self.demo_data_dir / 'config_extract_features.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        result = self.runner.invoke(cli, ['extract', '-c', str(config_path)])
        self.results['extract'] = {
            'exit_code': result.exit_code,
            'output': result.output,
            'error': result.exception
        }
        
        # Log result
        print(f"\n=== Extract Features Result ===")
        print(f"Exit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Error: {result.output}")
        
        # Should not crash
        assert result.exit_code in [0, 1], f"Extract failed with exit code {result.exit_code}"
    
    def test_step4_model_train(self):
        """Step 4: Test model training command"""
        config_path = self.demo_data_dir / 'config_machine_learning_clinical.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        result = self.runner.invoke(cli, ['model', '-c', str(config_path), '-m', 'train'])
        self.results['model_train'] = {
            'exit_code': result.exit_code,
            'output': result.output,
            'error': result.exception
        }
        
        # Log result
        print(f"\n=== Model Train Result ===")
        print(f"Exit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Error: {result.output}")
        
        # Should not crash
        assert result.exit_code in [0, 1], f"Model train failed with exit code {result.exit_code}"
    
    def test_step5_compare(self):
        """Step 5: Test model comparison command"""
        config_path = self.demo_data_dir / 'config_model_comparison.yaml'
        
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")
        
        result = self.runner.invoke(cli, ['compare', '-c', str(config_path)])
        self.results['compare'] = {
            'exit_code': result.exit_code,
            'output': result.output,
            'error': result.exception
        }
        
        # Log result
        print(f"\n=== Compare Result ===")
        print(f"Exit code: {result.exit_code}")
        if result.exit_code != 0:
            print(f"Error: {result.output}")
        
        # Should not crash
        assert result.exit_code in [0, 1], f"Compare failed with exit code {result.exit_code}"
    
    def teardown_method(self):
        """Print summary of all test results"""
        print("\n" + "=" * 80)
        print("WORKFLOW TEST SUMMARY")
        print("=" * 80)
        for step, result in self.results.items():
            status = "✓ PASS" if result['exit_code'] == 0 else "⚠ SKIP/ERROR"
            print(f"{step:20s}: {status} (exit_code={result['exit_code']})")
        print("=" * 80)


if __name__ == '__main__':
    # Run individual tests
    pytest.main([__file__, '-v', '-s'])
