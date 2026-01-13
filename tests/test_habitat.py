# debug_habitat.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for habitat analysis
    sys.argv = ['habit', 'get-habitat', '-c', './demo_data/config_habitat.yaml']
    cli()

