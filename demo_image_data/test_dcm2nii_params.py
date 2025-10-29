"""
Test different dcm2niix parameter combinations

This script helps you test different parameter combinations to find
which one gives you the desired 3D output instead of 4D.

Usage:
    python test_dcm2nii_params.py
"""

import subprocess
import SimpleITK as sitk
from pathlib import Path
import shutil
import tempfile


def run_dcm2niix_with_params(
    dcm2niix_exe: str,
    input_dir: str,
    output_dir: str,
    merge_param: str = None,
    single_file_param: str = None
) -> tuple:
    """
    Run dcm2niix with specific parameters.
    
    Args:
        dcm2niix_exe: Path to dcm2niix executable
        input_dir: Input DICOM directory
        output_dir: Output directory
        merge_param: Value for -m parameter (None, "y", "n", "2")
        single_file_param: Value for -s parameter (None, "y", "n")
        
    Returns:
        tuple: (command, output_files, success)
    """
    # Build command
    cmd = [dcm2niix_exe]
    cmd.extend(["-b", "n"])  # No JSON
    cmd.extend(["-z", "y"])  # Compress
    cmd.extend(["-v", "y"])  # Verbose
    
    if merge_param is not None:
        cmd.extend(["-m", merge_param])
    
    if single_file_param is not None:
        cmd.extend(["-s", single_file_param])
    
    cmd.extend(["-o", output_dir])
    cmd.append(input_dir)
    
    # Convert to string
    cmd_str = ' '.join(f'"{part}"' if ' ' in part else part for part in cmd)
    
    print(f"\nCommand: {cmd_str}")
    
    # Run command
    try:
        result = subprocess.run(
            cmd_str,
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Find output files
        output_path = Path(output_dir)
        nifti_files = list(output_path.glob("*.nii*"))
        
        return cmd_str, nifti_files, result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return cmd_str, [], False


def check_nifti_dimensions(nifti_files: list) -> None:
    """
    Check and display dimensions of NIfTI files.
    
    Args:
        nifti_files: List of NIfTI file paths
    """
    for nifti_file in nifti_files:
        try:
            img = sitk.ReadImage(str(nifti_file))
            size = img.GetSize()
            dims = len(size)
            
            print(f"  File: {nifti_file.name}")
            print(f"  Size: {size}")
            print(f"  Dimensions: {dims}D {'✓ 3D' if dims == 3 else '✗ 4D' if dims == 4 else ''}")
        except Exception as e:
            print(f"  Error reading {nifti_file.name}: {e}")


def main():
    """Main function to test different parameter combinations."""
    
    # Configuration
    dcm2niix_exe = r"F:\work\research\radiomics_TLSs\habit_project\demo_image_data\dcm2niix.exe"
    input_dir = r"F:\work\research\radiomics_TLSs\habit_project\demo_image_data\dicom\sub001\WATER_BHAxLAVA-Flex-2min_Series0012"
    
    # Check if paths exist
    if not Path(dcm2niix_exe).exists():
        print(f"Error: dcm2niix executable not found: {dcm2niix_exe}")
        print("Please update the dcm2niix_exe path in this script.")
        return
    
    if not Path(input_dir).exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please update the input_dir path in this script.")
        return
    
    # Test configurations
    test_configs = [
        {
            "name": "Test 1: No -m or -s parameters (dcm2niix defaults)",
            "merge": None,
            "single": None
        },
        {
            "name": "Test 2: -m 2, no -s (Recommended)",
            "merge": "2",
            "single": None
        },
        {
            "name": "Test 3: -m y, no -s",
            "merge": "y",
            "single": None
        },
        {
            "name": "Test 4: -m n, no -s (No merging)",
            "merge": "n",
            "single": None
        },
        {
            "name": "Test 5: -m 2, -s y",
            "merge": "2",
            "single": "y"
        },
        {
            "name": "Test 6: -m y, -s y",
            "merge": "y",
            "single": "y"
        },
    ]
    
    print("=" * 80)
    print("Testing different dcm2niix parameter combinations")
    print("=" * 80)
    print(f"dcm2niix: {dcm2niix_exe}")
    print(f"Input: {input_dir}")
    print("=" * 80)
    
    results = []
    
    for config in test_configs:
        print(f"\n{'=' * 80}")
        print(config["name"])
        print("=" * 80)
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp(prefix="dcm2niix_test_")
        
        try:
            cmd, files, success = run_dcm2niix_with_params(
                dcm2niix_exe,
                input_dir,
                temp_dir,
                config["merge"],
                config["single"]
            )
            
            if success and files:
                print(f"\nSuccess! Generated {len(files)} file(s):")
                check_nifti_dimensions(files)
                
                # Store result
                results.append({
                    "config": config["name"],
                    "files": files,
                    "success": True
                })
            else:
                print("\nFailed or no files generated")
                results.append({
                    "config": config["name"],
                    "files": [],
                    "success": False
                })
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    
    for result in results:
        status = "✓" if result["success"] else "✗"
        print(f"\n{status} {result['config']}")
        if result["files"]:
            for f in result["files"]:
                try:
                    img = sitk.ReadImage(str(f))
                    size = img.GetSize()
                    dims = len(size)
                    print(f"    {f.name}: {size} ({dims}D)")
                except:
                    pass
    
    print("\n" + "=" * 80)
    print("Recommendation:")
    print("- If you want 3D output (e.g., 512x512x80), use the config that gives 3D")
    print("- If you get 4D output (e.g., 512x512x40x2), try different -m parameters")
    print("- Update your YAML config with the parameters from successful test")
    print("=" * 80)


if __name__ == "__main__":
    main()

