# test_icc.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for ICC analysis
    sys.argv = ['habit', 'icc', '-c', './demo_data/config_icc.yaml']
    cli()