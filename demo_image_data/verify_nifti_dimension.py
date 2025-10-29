"""
Verify NIfTI Image Dimensions

This script helps verify whether the converted NIfTI images are 3D or 4D,
and displays their dimensions.

Usage:
    python verify_nifti_dimension.py
    
Or specify a file:
    python verify_nifti_dimension.py path/to/image.nii.gz
"""

import SimpleITK as sitk
from pathlib import Path
import sys


def check_image_dimension(image_path: str) -> None:
    """
    Check and display NIfTI image dimensions.
    
    Args:
        image_path (str): Path to NIfTI image file
    """
    try:
        # Read image
        image = sitk.ReadImage(image_path)
        
        # Get image information
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        pixel_type = image.GetPixelIDTypeAsString()
        
        # Determine if 3D or 4D
        n_dims = len(size)
        is_3d = n_dims == 3
        
        # Display results
        print("=" * 80)
        print(f"File: {image_path}")
        print("=" * 80)
        print(f"Dimensions: {n_dims}D {'✓ (3D as expected)' if is_3d else '✗ (4D, may need adjustment)'}")
        print(f"Size: {size}")
        
        if is_3d:
            print(f"  Width  (X): {size[0]} pixels")
            print(f"  Height (Y): {size[1]} pixels")
            print(f"  Slices (Z): {size[2]} slices")
        else:
            print(f"  Width  (X): {size[0]} pixels")
            print(f"  Height (Y): {size[1]} pixels")
            print(f"  Slices (Z): {size[2]} slices")
            print(f"  Time/Echo: {size[3]} volumes")
        
        print(f"\nSpacing: {spacing}")
        print(f"Origin: {origin}")
        print(f"Pixel Type: {pixel_type}")
        print(f"Direction: {direction}")
        
        # Calculate total number of voxels
        total_voxels = 1
        for s in size:
            total_voxels *= s
        print(f"\nTotal voxels: {total_voxels:,}")
        
        # Recommendations
        if not is_3d:
            print("\n" + "!" * 80)
            print("WARNING: This is a 4D image!")
            print("If you expected a 3D image, consider setting in your config:")
            print("  merge_slices: true")
            print("  single_file_mode: true")
            print("!" * 80)
        
        print()
        
    except Exception as e:
        print(f"Error reading image: {e}")


def find_nifti_files(directory: str) -> list:
    """
    Find all NIfTI files in a directory.
    
    Args:
        directory (str): Directory to search
        
    Returns:
        list: List of NIfTI file paths
    """
    nifti_files = []
    path = Path(directory)
    
    # Search for .nii and .nii.gz files
    for pattern in ['**/*.nii', '**/*.nii.gz']:
        nifti_files.extend(path.glob(pattern))
    
    return sorted(nifti_files)


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # File path provided as argument
        image_path = sys.argv[1]
        check_image_dimension(image_path)
    else:
        # Search for NIfTI files in current directory
        script_dir = Path(__file__).parent
        output_dir = script_dir / "nii" / "processed_images" / "images"
        
        print(f"Searching for NIfTI files in: {output_dir}")
        print()
        
        if output_dir.exists():
            nifti_files = find_nifti_files(str(output_dir))
            
            if nifti_files:
                print(f"Found {len(nifti_files)} NIfTI file(s)")
                print()
                
                for nifti_file in nifti_files:
                    check_image_dimension(str(nifti_file))
            else:
                print("No NIfTI files found.")
        else:
            print(f"Directory does not exist: {output_dir}")
            print("Please run the dcm2nii conversion first.")


if __name__ == "__main__":
    main()

