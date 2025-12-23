# debug_compare.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for model comparison
    sys.argv = ['habit', 'compare', '-c', 'config/config_model_comparison.yaml']
    cli()

