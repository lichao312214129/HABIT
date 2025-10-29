"""
Merge two 3D volumes into one

If dcm2niix outputs two 3D volumes (e.g., 512x512x40 each),
this script can merge them into a single 3D volume (512x512x80).

Usage:
    python merge_3d_volumes.py volume1.nii.gz volume2.nii.gz output.nii.gz
"""

import SimpleITK as sitk
import numpy as np
import sys
from pathlib import Path


def merge_volumes_along_z(volume1_path: str, volume2_path: str, output_path: str) -> None:
    """
    Merge two 3D volumes along Z-axis.
    
    Args:
        volume1_path: Path to first volume
        volume2_path: Path to second volume  
        output_path: Path for merged output
    """
    print("="*80)
    print("MERGING 3D VOLUMES")
    print("="*80)
    
    # Read volumes
    print(f"\n📖 Reading volume 1: {volume1_path}")
    vol1 = sitk.ReadImage(volume1_path)
    size1 = vol1.GetSize()
    print(f"   Size: {size1}")
    print(f"   Spacing: {vol1.GetSpacing()}")
    
    print(f"\n📖 Reading volume 2: {volume2_path}")
    vol2 = sitk.ReadImage(volume2_path)
    size2 = vol2.GetSize()
    print(f"   Size: {size2}")
    print(f"   Spacing: {vol2.GetSpacing()}")
    
    # Check compatibility
    print("\n🔍 Checking compatibility...")
    if size1[0] != size2[0] or size1[1] != size2[1]:
        print(f"❌ Error: Volumes have different X-Y dimensions!")
        print(f"   Volume 1: {size1[0]}x{size1[1]}")
        print(f"   Volume 2: {size2[0]}x{size2[1]}")
        return
    
    if vol1.GetSpacing() != vol2.GetSpacing():
        print(f"⚠️  Warning: Volumes have different spacing!")
        print(f"   Volume 1: {vol1.GetSpacing()}")
        print(f"   Volume 2: {vol2.GetSpacing()}")
        print(f"   Proceeding anyway...")
    
    # Convert to numpy arrays
    print("\n🔄 Converting to numpy arrays...")
    arr1 = sitk.GetArrayFromImage(vol1)  # Shape: (Z1, Y, X)
    arr2 = sitk.GetArrayFromImage(vol2)  # Shape: (Z2, Y, X)
    
    print(f"   Array 1 shape: {arr1.shape}")
    print(f"   Array 2 shape: {arr2.shape}")
    
    # Concatenate along Z-axis (axis 0 in numpy)
    print("\n🔗 Concatenating along Z-axis...")
    merged_arr = np.concatenate([arr1, arr2], axis=0)
    print(f"   Merged shape: {merged_arr.shape}")
    
    # Convert back to SimpleITK Image
    print("\n📦 Converting back to SimpleITK Image...")
    merged_img = sitk.GetImageFromArray(merged_arr)
    
    # Copy metadata from first volume
    merged_img.SetSpacing(vol1.GetSpacing())
    merged_img.SetOrigin(vol1.GetOrigin())
    merged_img.SetDirection(vol1.GetDirection())
    
    final_size = merged_img.GetSize()
    print(f"   Final size: {final_size}")
    print(f"   Expected: ({size1[0]}, {size1[1]}, {size1[2] + size2[2]})")
    
    # Save
    print(f"\n💾 Saving to: {output_path}")
    sitk.WriteImage(merged_img, output_path, useCompression=True)
    
    print("\n✅ Success!")
    print(f"   Merged {size1[2]} + {size2[2]} = {final_size[2]} slices")
    print("="*80)


def auto_find_and_merge(search_dir: str, output_name: str = "merged.nii.gz") -> None:
    """
    Automatically find two volumes and merge them.
    
    Args:
        search_dir: Directory to search for volumes
        output_name: Output filename
    """
    search_path = Path(search_dir)
    nifti_files = sorted(search_path.glob("*.nii*"))
    
    if len(nifti_files) < 2:
        print(f"❌ Error: Found only {len(nifti_files)} file(s) in {search_dir}")
        print("Need at least 2 files to merge.")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files:")
    for i, f in enumerate(nifti_files, 1):
        try:
            img = sitk.ReadImage(str(f))
            size = img.GetSize()
            dims = len(size)
            print(f"  {i}. {f.name}: {size} ({dims}D)")
        except:
            print(f"  {i}. {f.name}: (could not read)")
    
    # Find 3D volumes
    volumes_3d = []
    for f in nifti_files:
        try:
            img = sitk.ReadImage(str(f))
            if len(img.GetSize()) == 3:
                volumes_3d.append(f)
        except:
            pass
    
    if len(volumes_3d) < 2:
        print(f"\n❌ Error: Found only {len(volumes_3d)} 3D volume(s)")
        return
    
    print(f"\n🎯 Found {len(volumes_3d)} 3D volumes, merging first two...")
    
    output_path = search_path / output_name
    merge_volumes_along_z(str(volumes_3d[0]), str(volumes_3d[1]), str(output_path))


def main():
    """Main function."""
    
    if len(sys.argv) == 4:
        # Manual mode: merge_3d_volumes.py vol1 vol2 output
        volume1_path = sys.argv[1]
        volume2_path = sys.argv[2]
        output_path = sys.argv[3]
        
        merge_volumes_along_z(volume1_path, volume2_path, output_path)
    
    elif len(sys.argv) == 2:
        # Auto mode: merge_3d_volumes.py directory
        search_dir = sys.argv[1]
        auto_find_and_merge(search_dir)
    
    else:
        print("="*80)
        print("MERGE 3D VOLUMES TOOL")
        print("="*80)
        print("\nUsage:")
        print("  Manual mode:")
        print("    python merge_3d_volumes.py volume1.nii.gz volume2.nii.gz output.nii.gz")
        print("\n  Auto mode (finds and merges first 2 volumes in directory):")
        print("    python merge_3d_volumes.py /path/to/directory/")
        print("\nExample:")
        print("  python merge_3d_volumes.py output/")
        print("="*80)


if __name__ == "__main__":
    main()

