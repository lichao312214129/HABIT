"""Test dice_calculator.py
"""


import sys
from habit.cli import cli

if __name__ == '__main__':
    # Simulate command line arguments for model comparison
    sys.argv = ['habit', 'dice', '--input1', 'F:/work/research/radiomics_TLSs/manuscript/JMRI/revision1/roi1.yaml', '--input2', 'F:/work/research/radiomics_TLSs/manuscript/JMRI/revision1/roi2.yaml', '--output', 'F:/work/research/radiomics_TLSs/manuscript/JMRI/revision1/dice.csv']
    cli()