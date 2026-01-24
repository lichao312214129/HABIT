# debug_kfold.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for k-fold cross validation
    sys.argv = ['habit', 'kfold', '-c', 'F:/work/research/radiomics_TLSs/habit_project/demo_image_data/config_kfold.yaml']
    cli()

