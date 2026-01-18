"""
Script to remove old ICC calculation files
"""
import os
from pathlib import Path

# Files to remove
files_to_remove = [
    'habit/core/machine_learning/feature_selectors/icc/icc.py',
    'habit/core/machine_learning/feature_selectors/icc_selector.py',
    'scripts/app_icc_analysis.py',
    'scripts/app_icc_analysis_simple.py',
    'tests/test_simple_icc.py'
]

# Remove files
removed_count = 0
for file_path in files_to_remove:
    path = Path(file_path)
    if path.exists():
        try:
            path.unlink()
            print(f"Removed: {file_path}")
            removed_count += 1
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    else:
        print(f"Not found: {file_path}")

print(f"\nCleanup completed! Removed {removed_count} files.")