"""
Diagnose DICOM to NIfTI conversion issue

This script helps diagnose why dcm2niix produces 4D (40x2) instead of 3D (80) output.
"""

import subprocess
import SimpleITK as sitk
from pathlib import Path
import pydicom
import shutil
import tempfile


def check_dicom_tags(dicom_dir: str, num_samples: int = 5) -> None:
    """
    Check DICOM tags to understand the structure.
    
    Args:
        dicom_dir: Directory containing DICOM files
        num_samples: Number of files to sample
    """
    print("\n" + "="*80)
    print("DICOM TAG ANALYSIS")
    print("="*80)
    
    dicom_files = list(Path(dicom_dir).glob("*.dcm"))
    print(f"Total DICOM files: {len(dicom_files)}")
    
    if not dicom_files:
        print("No DICOM files found!")
        return
    
    # Sample first few files
    samples = dicom_files[:min(num_samples, len(dicom_files))]
    
    print(f"\nChecking {len(samples)} sample files:")
    print("-" * 80)
    
    for i, dcm_file in enumerate(samples, 1):
        try:
            ds = pydicom.dcmread(str(dcm_file), force=True)
            
            print(f"\nFile {i}: {dcm_file.name}")
            
            # Key tags that might indicate multiple volumes
            tags_to_check = {
                'SeriesNumber': 'Series Number',
                'SeriesDescription': 'Series Description',
                'InstanceNumber': 'Instance Number',
                'ImageOrientationPatient': 'Image Orientation',
                'SliceLocation': 'Slice Location',
                'EchoNumbers': 'Echo Number',
                'EchoTime': 'Echo Time (TE)',
                'TemporalPositionIdentifier': 'Temporal Position',
                'AcquisitionNumber': 'Acquisition Number',
                'ImageType': 'Image Type',
            }
            
            for tag, desc in tags_to_check.items():
                if hasattr(ds, tag):
                    value = getattr(ds, tag)
                    print(f"  {desc:25s}: {value}")
        
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    print("\n" + "="*80)


def test_dcm2niix_params(dcm2niix_exe: str, input_dir: str, test_configs: list) -> None:
    """
    Test different dcm2niix parameter combinations.
    
    Args:
        dcm2niix_exe: Path to dcm2niix executable
        input_dir: Input DICOM directory
        test_configs: List of configuration dictionaries
    """
    print("\n" + "="*80)
    print("TESTING DIFFERENT dcm2niix PARAMETERS")
    print("="*80)
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {config['name']}")
        print("="*80)
        
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp(prefix=f"dcm2niix_test_{i}_")
        
        try:
            # Build command
            cmd_parts = [dcm2niix_exe]
            cmd_parts.extend(["-b", "n"])  # No JSON
            cmd_parts.extend(["-z", "y"])  # Compress
            cmd_parts.extend(["-v", "y"])  # Verbose
            
            # Add test-specific parameters
            for param, value in config.get('params', {}).items():
                if value is not None:
                    cmd_parts.extend([param, value])
            
            cmd_parts.extend(["-o", temp_dir])
            cmd_parts.append(input_dir)
            
            # Convert to string
            cmd_str = ' '.join(f'"{part}"' if ' ' in str(part) else str(part) for part in cmd_parts)
            
            print(f"\nCommand: {cmd_str}")
            print("-" * 80)
            
            # Run command
            result = subprocess.run(
                cmd_str,
                shell=True,
                capture_output=True,
                text=True
            )
            
            # Check output
            output_path = Path(temp_dir)
            nifti_files = list(output_path.glob("*.nii*"))
            
            print(f"\nGenerated {len(nifti_files)} file(s):")
            
            for nifti_file in nifti_files:
                try:
                    img = sitk.ReadImage(str(nifti_file))
                    size = img.GetSize()
                    dims = len(size)
                    
                    print(f"  📄 {nifti_file.name}")
                    print(f"     Size: {size}")
                    print(f"     Dimensions: {dims}D", end="")
                    
                    if dims == 3 and size[2] == 80:
                        print(" ✅ PERFECT! This is what we want!")
                        results.append({
                            'config': config['name'],
                            'size': size,
                            'dims': dims,
                            'success': True
                        })
                    elif dims == 3:
                        print(f" ✓ 3D but only {size[2]} slices")
                        results.append({
                            'config': config['name'],
                            'size': size,
                            'dims': dims,
                            'success': False
                        })
                    else:
                        print(f" ✗ 4D - not desired")
                        results.append({
                            'config': config['name'],
                            'size': size,
                            'dims': dims,
                            'success': False
                        })
                except Exception as e:
                    print(f"  Error reading {nifti_file.name}: {e}")
            
            # Print dcm2niix output if verbose
            if result.stdout:
                print("\ndcm2niix output (last 5 lines):")
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    print(f"  {line}")
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for result in results:
        status = "✅" if result.get('success') else "❌"
        print(f"{status} {result['config']}")
        print(f"   Size: {result['size']}, {result['dims']}D")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    
    # Configuration
    dcm2niix_exe = r"F:\work\research\radiomics_TLSs\habit_project\demo_image_data\dcm2niix.exe"
    input_dir = r"F:\work\research\radiomics_TLSs\habit_project\demo_image_data\dicom\sub001\WATER_BHAxLAVA-Flex-2min_Series0012"
    
    # Check if paths exist
    if not Path(dcm2niix_exe).exists():
        print(f"❌ Error: dcm2niix executable not found: {dcm2niix_exe}")
        return
    
    if not Path(input_dir).exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return
    
    print("="*80)
    print("DCIM2NIIX 3D vs 4D DIAGNOSTIC TOOL")
    print("="*80)
    print(f"dcm2niix: {dcm2niix_exe}")
    print(f"Input: {input_dir}")
    print("="*80)
    
    # Step 1: Analyze DICOM tags
    print("\n🔍 Step 1: Analyzing DICOM tags...")
    try:
        check_dicom_tags(input_dir, num_samples=5)
    except Exception as e:
        print(f"Warning: Could not analyze DICOM tags: {e}")
        print("Installing pydicom might help: pip install pydicom")
    
    # Step 2: Test different parameter combinations
    print("\n🧪 Step 2: Testing different dcm2niix parameters...")
    
    test_configs = [
        {
            'name': 'No -m or -s parameters (dcm2niix defaults)',
            'params': {}
        },
        {
            'name': '-m n (no merge)',
            'params': {'-m': 'n'}
        },
        {
            'name': '-m y (default merge)',
            'params': {'-m': 'y'}
        },
        {
            'name': '-m 2 (merge by series)',
            'params': {'-m': '2'}
        },
        {
            'name': '-m n -s n (no merge, multiple files allowed)',
            'params': {'-m': 'n', '-s': 'n'}
        },
        {
            'name': '-m y -s n (merge, multiple files allowed)',
            'params': {'-m': 'y', '-s': 'n'}
        },
        {
            'name': '-m 0 (merge 2D, ignore trigger)',
            'params': {'-m': '0'}
        },
    ]
    
    test_dcm2niix_params(dcm2niix_exe, input_dir, test_configs)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. Check the test results above to find which config gives 3D 80-slice output")
    print("2. Update your dcm2nii.yaml with the successful parameters")
    print("3. If no config works, the DICOM data might truly be 2 separate volumes")
    print("4. You can manually merge the 2 volumes using SimpleITK or other tools")
    print("="*80)


if __name__ == "__main__":
    main()

