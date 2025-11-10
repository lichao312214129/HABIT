# debug_icc.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for ICC analysis
    sys.argv = ['habit', 'icc', '-c', 'F:/work/research/radiomics_TLSs/habit_project/demo_image_data/config_icc.yaml']
    cli()

