# debug_habitat.py
import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for habitat analysis
    sys.argv = ['habit', 'habitat', '-c', 'F:/work/research/radiomics_TLSs/test/config.yaml']
    cli()

