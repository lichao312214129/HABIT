# run_all_tests.py
"""Run all unit tests and generate coverage report"""
import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with pytest"""
    
    print("=" * 80)
    print("Running HABIT Test Suite")
    print("=" * 80)
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Run pytest with coverage
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir),
        "-v",
        "--tb=short",
        "-s",  # Show print statements
    ]
    
    # Try to run with coverage if available
    try:
        import pytest_cov
        cmd.extend([
            "--cov=habit",
            "--cov-report=html",
            "--cov-report=term-missing",
        ])
        print("Running with coverage analysis...")
    except ImportError:
        print("pytest-cov not installed. Running without coverage.")
        print("Install with: pip install pytest-cov")
    
    print("\nCommand:", " ".join(cmd))
    print("=" * 80)
    
    # Run the command
    result = subprocess.run(cmd)
    
    return result.returncode


def run_debug_tests():
    """Run all debug scripts"""
    
    print("\n" + "=" * 80)
    print("Available Debug Scripts:")
    print("=" * 80)
    
    tests_dir = Path(__file__).parent
    debug_scripts = sorted(tests_dir.glob("debug_*.py"))
    
    for i, script in enumerate(debug_scripts, 1):
        print(f"{i}. {script.name}")
    
    print("\nTo run a debug script, execute:")
    print("python tests/debug_<module>.py")
    print("=" * 80)


if __name__ == '__main__':
    # Run unit tests
    exit_code = run_tests()
    
    # Show debug scripts
    run_debug_tests()
    
    sys.exit(exit_code)

