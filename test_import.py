#!/usr/bin/env python
"""
Test script to verify the import of HabitatFeatureExtractor
"""

try:
    from habit.utils.io_utils import load_config, setup_logging
    print("Successfully imported from habit.utils.io_utils")
except ImportError as e:
    print(f"Failed to import from habit.utils.io_utils: {e}")

try:
    from habit.core.habitat_analysis import HabitatFeatureExtractor
    print("Successfully imported HabitatFeatureExtractor from habit.core.habitat_analysis")
except ImportError as e:
    print(f"Failed to import HabitatFeatureExtractor: {e}")

from habit.core.habitat_analysis import HabitatAnalysis

print("Successfully imported HabitatAnalysis")

print("Import test completed.") 