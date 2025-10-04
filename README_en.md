# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

<p align="center">
  <img src="https://www.imagilys.com/wp-content/uploads/2018/09/radiomics-illustration-500x500.png" alt="Radiomics" width="200"/>
</p>

<p align="center">
    <a href="https://github.com/your-repo/habit_project/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Active-green.svg"></a>
</p>

**HABIT** is a comprehensive, Python-based toolkit for tumor habitat analysis in medical imaging. It provides an end-to-end solution, from image preprocessing to machine learning, enabling researchers to investigate tumor heterogeneity through radiomics and advanced analytics.

---

## 📖 Conceptual Workflow

The core idea of HABIT is to identify and characterize sub-regions within a tumor, known as "habitats," which have distinct radiological phenotypes. This is achieved through a multi-stage pipeline:

<p align="center">
  <b>Image → Voxel Features → Supervoxels → Habitats → Habitat Features → Predictive Model</b>
</p>

1.  **Voxel-Level Feature Extraction**: Extracts rich features (e.g., intensity, texture, kinetic) for every single voxel within the tumor.
2.  **Supervoxel Clustering**: Groups spatially adjacent voxels with similar features into "supervoxels." This over-segmentation step simplifies the image while preserving local boundaries.
3.  **Habitat Clustering**: Groups the supervoxels across a patient cohort to identify common, recurring patterns, forming the final "habitats."
4.  **Feature Engineering**: Extracts high-level features from these habitats, such as their size, shape, spatial relationships (MSI features), and heterogeneity (ITH score).
5.  **Machine Learning**: Uses the engineered habitat features to train predictive models for clinical endpoints like patient survival, treatment response, or diagnosis.

---

## 🔬 Complete Research Workflow

A typical radiomics research project using HABIT involves the following steps. The HABIT toolkit provides powerful support for the steps marked with `[HABIT]`.

1.  **Data Acquisition and Download**:
    *   Obtain original imaging data (usually in DICOM format) from a hospital PACS or a public dataset.
    *   *This is a preparatory step performed outside of the HABIT toolkit.*

2.  **Data Organization and Anonymization**:
    *   Organize the data into a `Patient/Sequence/Files` structure.
    *   Anonymize patient-sensitive information.
    *   `[HABIT]` The `dcm2niix_converter` module supports anonymization during format conversion.

3.  **Format Conversion (DICOM to NIfTI)**:
    *   `[HABIT]` Use the `dcm2niix_converter` module or the `app_image_preprocessing.py` script to convert DICOM series into NIfTI format (`.nii.gz`).

4.  **Region of Interest (ROI) Segmentation**:
    *   A radiologist or researcher manually delineates the tumor region (ROI) using specialized software like ITK-SNAP or 3D Slicer, saving it as a mask file (e.g., `mask.nii.gz`).
    *   *This step is typically performed outside the HABIT toolkit and generates the `mask` file required for subsequent steps.*

5.  **Image Preprocessing**:
    *   `[HABIT]` Use the `app_image_preprocessing.py` script for a series of preprocessing steps, including:
        *   **Registration**: Aligning images from different sequences or modalities to the same space.
        *   **Resampling**: Standardizing all images to the same voxel spacing.
        *   **Intensity Normalization**: Such as Z-Score normalization.
        *   **N4 Bias Field Correction**: Correcting for signal inhomogeneity in MRI.

6.  **Habitat Analysis and Feature Extraction**:
    *   `[HABIT]` Run the core script `app_getting_habitat_map.py` to identify tumor habitats.
    *   `[HABIT]` Run `app_extracting_habitat_features.py` to extract high-level features (e.g., MSI, ITH score) from the habitats.

7.  **Building and Evaluating Predictive Models**:
    *   `[HABIT]` Use `app_of_machine_learning.py` for feature selection, model training, and internal validation.
    *   `[HABIT]` Use `app_model_comparison_plots.py` to compare the performance of different models and visualize the results.

8.  **Result Analysis and Manuscript Writing**:
    *   Interpret the findings from the model and write the research paper.
    *   *This step is performed outside of the HABIT toolkit.*

## 🚀 Key Features

| Category | Feature | Description |
| :--- | :--- | :--- |
| 🖼️ **Image Processing** | **Preprocessing Pipeline** | End-to-end tools for DICOM conversion, resampling, registration, and normalization. |
| | **N4 Bias Field Correction** | Corrects for intensity non-uniformity in MRI scans. |
| | **Histogram Standardization** | Standardizes intensity values across different patients or scanners. |
| 🧬 **Habitat Analysis** | **Multi-level Clustering** | A robust two-stage process (Supervoxel -> Habitat) to define tumor sub-regions. |
| | **Flexible Feature Input** | Supports various voxel-level features, including raw intensity, kinetic, and radiomics. |
| 🔬 **Feature Extraction** | **Advanced Feature Sets** | Extracts traditional radiomics, `non_radiomics` stats, `whole_habitat` features, `each_habitat` features, Multiregional Spatial Interaction (`msi`), and Intratumoral Heterogeneity (`ith_score`). |
| | **Configurable Engine** | Uses PyRadiomics with customizable parameter files for tailored feature extraction. |
| 🤖 **Machine Learning** | **Complete Workflow** | Includes data splitting, feature selection, model training, and evaluation. |
| | **Rich Algorithm Support** | Supports various models (Logistic Regression, SVM, RandomForest, XGBoost) and numerous feature selection methods (ICC, VIF, mRMR, LASSO, RFE). |
| | **Model Comparison** | Tools to generate ROC curves, Decision Curve Analysis (DCA), and perform DeLong tests. |
| 📊 **Validation & Utilities** | **Reproducibility Analysis** | Includes tools for Test-Retest and Inter-Class Correlation (ICC) analysis. |
| | **Modular & Configurable** | All steps are controlled via easy-to-edit YAML configuration files. |
| | **Robust Import System** | Ensures the toolkit remains operational even if some optional dependencies are missing. |

## 📁 Project Structure

```
habit_project/
├── habit/                      # Core Python source code package
│   ├── core/                   # Main modules for analysis
│   │   ├── habitat_analysis/   # Habitat identification logic
│   │   ├── machine_learning/   # ML modeling and evaluation
│   │   └── preprocessing/      # Image processing functions
│   └── utils/                  # Helper utilities (I/O, logging, etc.)
├── scripts/                    # Entry-point scripts for running analyses
├── config/                     # YAML configuration files for all scripts
├── doc/                        # Detailed documentation for each module
├── requirements.txt            # Python dependencies
├── INSTALL.md                  # Detailed installation guide
└── QUICKSTART.md               # 5-minute tutorial for new users
```

## 🛠️ Installation

A detailed guide is available in [**INSTALL.md**](INSTALL.md).

For a quick setup:
```bash
# 1. Clone the repository
git clone <repository_url>
cd habit_project

# 2. Create and activate a Conda environment
conda create -n habit python=3.8
conda activate habit

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the HABIT package in editable mode
pip install -e .
```

## 📖 Quick Start

New to HABIT? Follow our [**QUICKSTART.md**](QUICKSTART.md) guide to run your first habitat analysis in minutes!

### Basic Usage Example

All workflows in HABIT are driven by running a script from the `scripts/` directory with a corresponding configuration file from the `config/` directory.

**1. Run Habitat Analysis:**
```bash
python scripts/app_getting_habitat_map.py --config config/config_getting_habitat.yaml
```

**2. Extract Habitat Features:**
```bash
python scripts/app_extracting_habitat_features.py --config config/config_extract_features.yaml
```

**3. Train a Machine Learning Model:**
```bash
python scripts/app_of_machine_learning.py --config config/config_machine_learning.yaml
```

## 🤝 Contributing

Contributions are welcome! Please refer to the contribution guidelines (to be added) or open an issue to discuss your ideas.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Citation

If you use HABIT in your research, please consider citing:
> [Citation details to be added here]

## 🙋‍♀️ Support

If you encounter any problems or have suggestions, please:
1.  Read the detailed documentation in the `doc/` folder.
2.  Open an [Issue](https://github.com/your-repo/habit_project/issues) on GitHub.
