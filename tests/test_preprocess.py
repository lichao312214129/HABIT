# debug_preprocess.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for preprocessing
    sys.argv = ['habit', 'preprocess', '-c', './demo_data/config_preprocessing.yaml']
    cli()

