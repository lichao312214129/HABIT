# debug_extract_features.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for feature extraction
    sys.argv = ['habit', 'extract-features', '-c', 'F:\work\workstation_b\huangTeng\config_extract_features.yaml']
    cli()

