# debug_ml.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for machine learning
    sys.argv = ['habit', 'model', '-c', './demo_data/config_predict.yaml', '-m', 'predict']
    cli()

