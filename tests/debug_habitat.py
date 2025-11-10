# debug_habitat.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for habitat analysis
    sys.argv = ['habit', 'habitat', '-c', 'F:\work\workstation_b\huangTeng\config.yaml']
    cli()

