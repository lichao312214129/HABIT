#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for merge-csv command using sys.argv
Usage: python tests/test_merge_csv_files.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

if __name__ == "__main__":
    # Set sys.argv to simulate command line: habit merge-csv file1.csv file2.csv -o output.csv
    sys.argv = [
        'habit',
        'merge-csv',
        r'G:\DMS\intratumoral_heterogeneity\HCC-DCM\dicom_info.csv',
        r'H:\results\features\clinicalDataClearedAddedLesionDescribe_HCV.csv',
        '-o', r'G:\DMS\intratumoral_heterogeneity\HCC-DCM\merged_features.csv',
    ]
    
    # Import and run the CLI
    from habit.cli import cli
    cli()
