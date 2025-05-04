"""
Example script demonstrating how to use the batch processing module.

This script shows how to:
1. Configure preprocessing steps
2. Initialize the batch processor
3. Process multiple subjects in parallel
"""

import os
from pathlib import Path
import yaml

# Create example configuration
config = {
    "Preprocessing": {
        "resample": {
            "images": ["pre_contrast", "LAP", "PVP", "delay_3min"],
            "target_spacing": [1.0, 1.0, 1.0],
            "mode": "bilinear"
        },
        "n4_correction": {
            "images": ["pre_contrast", "LAP", "PVP", "delay_3min"],
            "shrink_factor": 4,
            "convergence_threshold": 0.001
        },
        "registration": {
            "images": ["pre_contrast", "LAP", "delay_3min"],
            "fixed_image": "PVP",
            "type_of_transform": "rigid"
        }
    }
}

def main():
    # Create example data directory structure
    data_root = Path("example_data")
    data_root.mkdir(exist_ok=True)
    
    # Create example subject
    subject_dir = data_root / "subject1"
    for time_point in ["pre_contrast", "LAP", "PVP", "delay_3min"]:
        # Create image directory
        image_dir = subject_dir / "images" / time_point
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mask directory
        mask_dir = subject_dir / "masks" / time_point
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy files (in real usage, these would be actual NIfTI files)
        (image_dir / "image.nii.gz").touch()
        (mask_dir / "mask.nii.gz").touch()
    
    # Save configuration
    config_path = data_root / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize and run batch processor
    from habit.core.batch_processor import BatchProcessor
    
    processor = BatchProcessor(
        config_path=config_path,
        num_workers=4,  # Use 4 worker processes
        log_level="INFO"
    )
    
    processor.process_batch(
        data_root=data_root,
        output_root=data_root / "processed"
    )

if __name__ == "__main__":
    main() 