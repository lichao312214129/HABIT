"""
Test workflow steps individually and report errors
This script tests each step: preprocess -> get-habitat -> extract -> model -> compare
"""
import sys
from pathlib import Path
from click.testing import CliRunner
from habit.cli import cli


def test_step(step_name, command, config_path, extra_args=None):
    """Test a single workflow step"""
    print(f"\n{'='*80}")
    print(f"Testing Step: {step_name}")
    print(f"Command: {' '.join(command)}")
    print(f"Config: {config_path}")
    print(f"{'='*80}")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    runner = CliRunner()
    cmd = command + ['-c', str(config_path)]
    if extra_args:
        cmd.extend(extra_args)
    
    result = runner.invoke(cli, cmd)
    
    print(f"Exit Code: {result.exit_code}")
    if result.exit_code == 0:
        print(f"✅ {step_name} completed successfully")
        return True
    else:
        print(f"❌ {step_name} failed")
        print(f"\nError Output:")
        print(result.output)
        if result.exception:
            print(f"\nException:")
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        return False


def main():
    """Run all workflow steps"""
    # Update path: tests/ is now the parent, so go up one more level
    demo_data_dir = Path(__file__).parent.parent / 'demo_data'
    results = {}
    
    # Step 1: Preprocess
    config_path = demo_data_dir / 'config_preprocessing.yaml'
    results['preprocess'] = test_step(
        'Preprocess',
        ['preprocess'],
        config_path
    )
    
    # Step 2: Get Habitat
    config_path = demo_data_dir / 'config_habitat.yaml'
    results['get_habitat'] = test_step(
        'Get Habitat',
        ['get-habitat'],
        config_path
    )
    
    # Step 3: Extract Features
    config_path = demo_data_dir / 'config_extract_features.yaml'
    results['extract'] = test_step(
        'Extract Features',
        ['extract'],
        config_path
    )
    
    # Step 4: Model Train
    config_path = demo_data_dir / 'config_machine_learning_clinical.yaml'
    results['model_train'] = test_step(
        'Model Train',
        ['model'],
        config_path,
        extra_args=['-m', 'train']
    )
    
    # Step 5: Compare
    config_path = demo_data_dir / 'config_model_comparison.yaml'
    results['compare'] = test_step(
        'Compare',
        ['compare'],
        config_path
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("WORKFLOW TEST SUMMARY")
    print(f"{'='*80}")
    for step, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{step:20s}: {status}")
    print(f"{'='*80}")
    
    # Return exit code
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
