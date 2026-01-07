# debug_ml.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for machine learning
    sys.argv = ['habit', 'ml', '-c', 'F:/work/research/radiomics_TLSs/habit_project/demo_image_data/config_ml.yaml']
    cli()

