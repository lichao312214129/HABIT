# debug_compare.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for model comparison
    sys.argv = ['habit', 'compare', '-c', 'F:/work/research/radiomics_TLSs/habit_project/demo_image_data/config_compare.yaml']
    cli()

