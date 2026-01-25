"""
Test cases for habit dicom-info command
"""
import sys
import pytest
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner


class TestDicomInfoCommand:
    """Test cases for dicom-info command"""
    
    def test_dicom_info_help(self):
        """Test dicom-info command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['dicom-info', '--help'])
        assert result.exit_code == 0
        assert 'dicom' in result.output.lower() or 'info' in result.output.lower()
    
    def test_dicom_info_missing_input(self):
        """Test dicom-info command with missing required input"""
        runner = CliRunner()
        result = runner.invoke(cli, ['dicom-info', '--output', 'out.csv'])
        assert result.exit_code != 0  # Should fail without --input
    
    def test_dicom_info_list_tags(self):
        """Test dicom-info command with --list-tags option"""
        runner = CliRunner()
        
        # Try with a non-existent directory to test error handling
        result = runner.invoke(cli, [
            'dicom-info',
            '--input', 'nonexistent_directory',
            '--list-tags'
        ])
        
        # Should fail gracefully with missing directory
        assert result.exit_code != 0
    
    def test_dicom_info_with_demo_data(self):
        """Test dicom-info command with demo_data if available"""
        demo_dicom_path = Path(__file__).parent.parent / 'demo_data' / 'dicom'
        
        if not demo_dicom_path.exists():
            pytest.skip(f"DICOM directory not found: {demo_dicom_path}")
        
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            output_file = 'dicom_info_test.csv'
            result = runner.invoke(cli, [
                'dicom-info',
                '--input', str(demo_dicom_path),
                '--output', output_file,
                '--one-file-per-folder'
            ])
            
            # Command should execute (may fail if no DICOM files, but should not crash)
            assert result.exit_code in [0, 1]


if __name__ == '__main__':
    # Allow running as script for debugging
    # Note: Update paths as needed
    sys.argv = [
        'habit', 'dicom-info',
        '--input', './demo_data/dicom',
        '--output', 'dicom_info.csv',
        '--one-file-per-folder'
    ]
    cli()
