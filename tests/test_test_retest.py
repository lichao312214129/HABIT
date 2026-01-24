# debug_test_retest.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for test-retest analysis
    sys.argv = ['habit', 'test-retest', '-c', 'F:/work/research/radiomics_TLSs/habit_project/demo_image_data/config_test_retest.yaml']
    cli()

