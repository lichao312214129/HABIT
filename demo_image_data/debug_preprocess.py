# debug_preprocess.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments
    sys.argv = ['habit', 'preprocess', '-c', 'F:/work/research/radiomics_TLSs/habit_project/demo_image_data/config_image_preprocessing.yaml']
    cli()

