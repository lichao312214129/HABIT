#!/usr/bin/env python
"""
Simple test script to verify dcm2niix module import
"""

try:
    from habit.core.preprocessing import Dcm2niixConverter, batch_convert_dicom_directories
    print("✓ Successfully imported Dcm2niixConverter and batch_convert_dicom_directories")
    
    # Test basic instantiation
    converter = Dcm2niixConverter(
        keys=["test_key"],
        output_dir="./test_output"
    )
    print("✓ Successfully created Dcm2niixConverter instance")
    
    print("✓ All imports and basic functionality test passed!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Other error: {e}") 