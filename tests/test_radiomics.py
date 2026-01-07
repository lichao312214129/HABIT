# debug_radiomics.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for radiomics extraction
    sys.argv = ['habit', 'radiomics', '-c', 'F:/work/research/radiomics_TLSs/habit_project/demo_image_data/config_radiomics.yaml']
    cli()

