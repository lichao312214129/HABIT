# HABIT Quick Start Guide

Welcome to HABIT! This guide will walk you through running your first tumor habitat analysis in just a few minutes.

---

### **Step 0: Installation**

Before you begin, make sure you have successfully installed HABIT and its dependencies. If not, please follow the [**INSTALL.md**](INSTALL.md) guide.

Once installed, activate your environment:
```bash
conda activate habit
```

### **Step 1: Prepare Your Data**

HABIT expects a specific directory structure for your data. Create a main data folder and organize your patient images and masks as shown below.

**Required Structure:**
```
your_project_folder/
├── data/                  # Your main data directory
│   ├── images/            # Contains one subfolder per patient
│   │   ├── patient_001/
│   │   │   ├── pre_contrast/  # Subfolder for each modality
│   │   │   │   └── image.nii.gz (or a series of DICOM files)
│   │   │   ├── LAP/
│   │   │   │   └── image.nii.gz
│   │   │   └── PVP/
│   │   │       └── image.nii.gz
│   │   └── patient_002/
│   │       └── ...
│   └── masks/             # Structure is identical to /images
│       ├── patient_001/
│       │   ├── pre_contrast/
│       │   │   └── mask.nii.gz
│       │   ├── LAP/
│       │   │   └── mask.nii.gz
│       │   └── PVP/
│       │       └── mask.nii.gz
│       └── patient_002/
│           └── ...
└── output/                # An empty directory for results
```
- The folder names (`pre_contrast`, `LAP`, `PVP`) are **keys** that you will reference in the configuration file.
- The `masks` directory mirrors the `images` directory structure. Each image file should have a corresponding mask file.
- Supported formats include `.nii.gz`, `.nii`, `.nrrd`, and `.mha`.

#### Alternative: Using a File Manifest

If your data is not organized in the structure above, you can instead create a YAML file (like `config/image_files.yaml`) to explicitly define the path for each image. This is useful when your files are in different locations.

**Example `image_files.yaml`:**
```yaml
images:
  subj003:
    T1: F:\\path\\to\\subj003\\T1_folder
    T2: F:\\path\\to\\subj003\\T2_folder
  subj004:
    T1: /another/path/to/subj004/t1_folder
    T2: /another/path/to/subj004/t2_folder
```

In your main analysis configuration, you would then reference this manifest file instead of specifying a `data_dir`.

### **Step 2: Configure Your Analysis**

All analyses in HABIT are controlled by YAML configuration files. Let's copy and edit a sample configuration.

1.  **Copy the example config**:
    ```bash
    cp config/config_getting_habitat.yaml my_first_analysis.yaml
    ```

2.  **Edit `my_first_analysis.yaml`** with a text editor and change the following critical paths:
    ```yaml
    # 1. Set your data and output directories
    data_dir: /path/to/your_project_folder/data  # Point to your main data folder
    out_dir: /path/to/your_project_folder/output # Point to your output folder
    
    # 2. Define the image keys for feature extraction
    # These must match the filenames from Step 1
    FeatureConstruction:
      voxel_level:
        method: concat(raw(pre_contrast), raw(LAP), raw(PVP))
        # ... other params
    
    # 3. (Optional) Adjust clustering parameters
    HabitatsSegmention:
      supervoxel:
        n_clusters: 50  # Number of initial supervoxels per patient
      habitat:
        mode: training
        max_clusters: 8 # Maximum number of final habitats to find
    
    # 4. (Optional) Adjust number of parallel processes based on your CPU cores
    processes: 4
    ```

### **Step 3: Run the Habitat Analysis**

Now, you can run the main analysis script using the configuration file you just created.

```bash
python scripts/app_getting_habitat_map.py --config my_first_analysis.yaml
```

This process will perform voxel feature extraction, supervoxel clustering, and habitat clustering. You will see progress bars for each stage.

### **Step 4: Understand the Output**

Once the analysis is complete, check your `output` directory. You will find several new folders:

-   `supervoxel_maps/`: Contains the intermediate supervoxel segmentation for each patient.
-   `habitat_maps/`: Contains the final habitat segmentation for each patient.
-   `features/`: Stores the raw voxel-level and processed supervoxel-level features in `.csv` files.
-   `clustering_models/`: The trained clustering models are saved here.
-   `plots/`: Visualizations, such as elbow plots for determining the optimal number of clusters.

**The most important output is the set of images in `habitat_maps/`.** You can view these with a medical image viewer (e.g., ITK-SNAP, 3D Slicer) to see the identified tumor sub-regions.

### **Step 5: What's Next?**

With the habitats identified, you can now proceed to the next steps in the radiomics pipeline.

**1. Extract High-Level Habitat Features:**
Use the generated habitat maps to extract advanced features like MSI (spatial relationships) and ITH (heterogeneity score).
```bash
# First, configure `config/config_extract_features.yaml` with your paths
python scripts/app_extracting_habitat_features.py --config config/config_extract_features.yaml
```

**2. Train a Predictive Model:**
Use the extracted features to train a machine learning model.
```bash
# First, configure `config/config_machine_learning.yaml` with your feature files
python scripts/app_of_machine_learning.py --config config/config_machine_learning.yaml
```

---

🎉 **Congratulations!** You have successfully run your first analysis with HABIT. Explore the other scripts and configuration files to discover the full power of the toolkit.