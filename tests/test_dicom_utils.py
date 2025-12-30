import sys
from habit.cli import cli

if __name__ == '__main__':
    sys.argv = ['habit', 'dicom-info', 
                '--input', 'G:\DMS\intratumoral_heterogeneity\HCC-DCM\ZSSY\Dicom', 
                '--output', 'G:\DMS\intratumoral_heterogeneity\HCC-DCM\dicom_info.csv', 
                '--one-file-per-folder', 
                '-j', '10', 
                '--no-group-by-series',
                '--max-depth', '2'
    ]
    cli()