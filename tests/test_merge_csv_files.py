"""
Test cases for habit merge-csv command
"""
import sys
import pytest
import tempfile
from pathlib import Path
from habit.cli import cli
from click.testing import CliRunner
import pandas as pd


class TestMergeCsvCommand:
    """Test cases for merge-csv command"""
    
    def test_merge_csv_help(self):
        """Test merge-csv command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ['merge-csv', '--help'])
        assert result.exit_code == 0
        assert 'merge' in result.output.lower() or 'csv' in result.output.lower()
    
    def test_merge_csv_with_test_files(self):
        """Test merge-csv command with temporary test files"""
        runner = CliRunner()
        
        # Create temporary CSV files for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first CSV file
            df1 = pd.DataFrame({
                'id': ['A', 'B', 'C'],
                'value1': [1, 2, 3]
            })
            file1 = Path(tmpdir) / 'file1.csv'
            df1.to_csv(file1, index=False)
            
            # Create second CSV file
            df2 = pd.DataFrame({
                'id': ['A', 'B', 'C'],
                'value2': [10, 20, 30]
            })
            file2 = Path(tmpdir) / 'file2.csv'
            df2.to_csv(file2, index=False)
            
            # Output file
            output_file = Path(tmpdir) / 'merged.csv'
            
            # Run merge-csv command
            result = runner.invoke(cli, [
                'merge-csv',
                str(file1),
                str(file2),
                '-o', str(output_file),
                '--index-col', 'id'
            ])
            
            # Command should succeed
            assert result.exit_code == 0
            assert output_file.exists()
            
            # Verify merged content
            merged = pd.read_csv(output_file)
            assert 'id' in merged.columns
            assert 'value1' in merged.columns
            assert 'value2' in merged.columns
            assert len(merged) == 3
    
    def test_merge_csv_missing_files(self):
        """Test merge-csv command with missing input files"""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'merged.csv'
            
            result = runner.invoke(cli, [
                'merge-csv',
                'nonexistent1.csv',
                'nonexistent2.csv',
                '-o', str(output_file)
            ])
            
            # Should fail with missing files
            assert result.exit_code != 0


if __name__ == '__main__':
    # Allow running as script for debugging
    # Note: Update paths as needed
    sys.argv = [
        'habit',
        'merge-csv',
        'file1.csv',
        'file2.csv',
        '-o', 'merged.csv',
        '--index-col', 'id'
    ]
    cli()
