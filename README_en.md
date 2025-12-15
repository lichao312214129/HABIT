# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

<p align="center">
  <img src="https://www.imagilys.com/wp-content/uploads/2018/09/radiomics-illustration-500x500.png" alt="Radiomics" width="200"/>
</p>

<p align="center">
  <strong>📖 Language / 语言</strong><br>
  <a href="README_en.md">🇬🇧 English</a> | <a href="README.md">🇨🇳 简体中文</a>
</p>

<p align="center">
    <a href="https://github.com/your-repo/habit_project/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Active-green.svg"></a>
</p>

**HABIT** is a comprehensive, Python-based toolkit for tumor habitat analysis in medical imaging. It provides an end-to-end solution, from image preprocessing to machine learning, enabling researchers to investigate tumor heterogeneity through radiomics and advanced analytics.

> 🎯 **New Feature**: HABIT now supports a unified command-line interface (CLI)! After installation, simply use the `habit` command to access all features. See the [Quick Start](#-quick-start) section for details.

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
    *   `[HABIT]` Use the `habit preprocess` command (or `app_image_preprocessing.py` script) for a series of preprocessing steps, including:
        *   **Registration**: Aligning images from different sequences or modalities to the same space.
        *   **Resampling**: Standardizing all images to the same voxel spacing.
        *   **Intensity Normalization**: Such as Z-Score normalization.
        *   **N4 Bias Field Correction**: Correcting for signal inhomogeneity in MRI.

6.  **Habitat Analysis and Feature Extraction**:
    *   `[HABIT]` Run `habit get-habitat` (or the core script `app_getting_habitat_map.py`) to identify tumor habitats.
        *   **Two Clustering Modes Supported**:
            *   **One-Step**: Direct voxel-to-habitat clustering, optimal cluster count auto-determined per tumor, independent habitat labels
            *   **Two-Step**: Individual-level supervoxel clustering followed by population-level habitat clustering, unified habitat labels across all patients
    *   `[HABIT]` Run `habit extract` (or `app_extracting_habitat_features.py`) to extract high-level features (e.g., MSI, ITH score) from the habitats.

7.  **Building and Evaluating Predictive Models**:
    *   `[HABIT]` Use `habit model` (or `app_of_machine_learning.py`) for feature selection, model training, and internal validation.
    *   `[HABIT]` Use `habit compare` (or `app_model_comparison_plots.py`) to compare the performance of different models and visualize the results.

8.  **Result Analysis and Manuscript Writing**:
    *   Interpret the findings from the model and write the research paper.
    *   *This step is performed outside of the HABIT toolkit.*

## 🚀 Key Features

| Category | Feature | Description | Docs |
| :--- | :--- | :--- | :---: |
| 🖼️ **Image Processing** | **Preprocessing Pipeline** | End-to-end tools for DICOM conversion, resampling, registration, and normalization. | [📖](doc_en/app_image_preprocessing.md) |
| | **N4 Bias Field Correction** | Corrects for intensity non-uniformity in MRI scans. | [📖](doc_en/app_image_preprocessing.md) |
| | **Histogram Standardization** | Standardizes intensity values across different patients or scanners. | [📖](doc_en/app_image_preprocessing.md) |
| 🧬 **Habitat Analysis** | **One-Step Clustering** | Direct clustering to habitats, cluster count determined per tumor, independent habitat labels. | [📖](doc_en/app_habitat_analysis.md) |
| | **Two-Step Clustering** | Two-stage clustering (individual supervoxels → population habitats), unified habitat labeling system. | [📖](doc_en/app_habitat_analysis.md) |
| | **Flexible Feature Input** | Supports various voxel-level features, including raw intensity, kinetic, and radiomics. | [📖](doc_en/app_habitat_analysis.md) |
| 🔬 **Feature Extraction** | **Advanced Feature Sets** | Extracts traditional radiomics, `non_radiomics` stats, `whole_habitat` features, `each_habitat` features, Multiregional Spatial Interaction (`msi`), and Intratumoral Heterogeneity (`ith_score`). | [📖](doc_en/app_extracting_habitat_features.md) |
| | **Configurable Engine** | Uses PyRadiomics with customizable parameter files for tailored feature extraction. | [📖](doc_en/app_extracting_habitat_features.md) |
| 🤖 **Machine Learning** | **Complete Workflow** | Includes data splitting, feature selection, model training, and evaluation. | [📖](doc_en/app_of_machine_learning.md) |
| | **Rich Algorithm Support** | Supports various models (Logistic Regression, SVM, RandomForest, XGBoost) and numerous feature selection methods (ICC, VIF, mRMR, LASSO, RFE). | [📖](doc_en/app_of_machine_learning.md) |
| | **K-Fold Cross-Validation** | Comprehensive K-fold cross-validation with multi-model evaluation and visualization. | [📖](doc_en/app_kfold_cross_validation.md) |
| | **Model Comparison** | Tools to generate ROC curves, Decision Curve Analysis (DCA), and perform DeLong tests. | [📖](doc_en/app_model_comparison_plots.md) |
| 📊 **Validation & Utilities** | **Reproducibility Analysis** | Includes tools for Test-Retest and Inter-Class Correlation (ICC) analysis. | [📖](doc_en/app_icc_analysis.md) |
| | **DICOM Conversion** | DICOM to NIfTI format conversion tools. | [📖](doc_en/app_dcm2nii.md) |
| | **Modular & Configurable** | All steps are controlled via easy-to-edit YAML configuration files. | [📖](HABIT_CLI.md) |

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

### 🎯 Unified Command Line Interface (CLI) - **Recommended Usage**

**HABIT provides a unified, streamlined command-line interface!** ✨

Using a CLI system built with **Click**, you only need the `habit` command to access all functionality—no need to remember complex script paths.

#### Ready to Use After Installation

After completing `pip install -e .`, the `habit` command is globally available in your environment:

```bash
# View all available commands
habit --help

# Get help for specific commands
habit model --help
habit cv --help
```

#### Core Command Examples

```bash
# 1️⃣ Image Preprocessing - Resampling, registration, normalization
habit preprocess --config config/config_image_preprocessing.yaml
# 📖 Documentation: doc_en/app_image_preprocessing.md

# 2️⃣ Generate Habitat Maps - Identify tumor sub-regions
# Supports one-step (personalized) or two-step (cohort study) mode
habit get-habitat --config config/config_getting_habitat.yaml
# 📖 Documentation: doc_en/app_habitat_analysis.md

# 3️⃣ Extract Habitat Features - MSI, ITH, and other advanced features
habit extract --config config/config_extract_features.yaml
# 📖 Documentation: doc_en/app_extracting_habitat_features.md

# 4️⃣ Machine Learning - Train predictive models
habit model --config config/config_machine_learning.yaml --mode train
# 📖 Documentation: doc_en/app_of_machine_learning.md

# 5️⃣ Model Prediction - Use trained models
habit model --mode predict \
  --model ./ml_data/model_package.pkl \
  --data ./new_data.csv \
  --output ./predictions/
# 📖 Documentation: doc_en/app_of_machine_learning.md

# 6️⃣ K-Fold Cross-Validation - Robust model evaluation
habit cv --config config/config_machine_learning_kfold.yaml
# 📖 Documentation: doc_en/app_kfold_cross_validation.md

# 7️⃣ Model Comparison - ROC, DCA, calibration curves visualization
habit compare --config config/config_model_comparison.yaml
# 📖 Documentation: doc_en/app_model_comparison_plots.md

# 8️⃣ ICC Analysis - Feature reproducibility assessment
habit icc --config config/config_icc_analysis.yaml
# 📖 Documentation: doc_en/app_icc_analysis.md

# 9️⃣ Traditional Radiomics Feature Extraction
habit radiomics --config config/config_traditional_radiomics.yaml

# 🔟 Test-Retest Habitat Mapping
habit retest --config config/config_habitat_test_retest.yaml
```

#### Quick Reference Table

| Command | Function | Config File | Docs |
|---------|----------|-------------|:---:|
| `habit preprocess` | Image preprocessing | `config_image_preprocessing.yaml` | [📖](doc_en/app_image_preprocessing.md) |
| `habit get-habitat` | Generate Habitat maps | `config_getting_habitat.yaml` | [📖](doc_en/app_habitat_analysis.md) |
| `habit extract` | Extract Habitat features | `config_extract_features.yaml` | [📖](doc_en/app_extracting_habitat_features.md) |
| `habit model` | ML training/prediction | `config_machine_learning.yaml` | [📖](doc_en/app_of_machine_learning.md) |
| `habit cv` | K-fold cross-validation | `config_machine_learning_kfold.yaml` | [📖](doc_en/app_kfold_cross_validation.md) |
| `habit compare` | Model comparison & viz | `config_model_comparison.yaml` | [📖](doc_en/app_model_comparison_plots.md) |
| `habit icc` | ICC reproducibility | `config_icc_analysis.yaml` | [📖](doc_en/app_icc_analysis.md) |
| `habit radiomics` | Traditional radiomics | `config_traditional_radiomics.yaml` | [📖](HABIT_CLI.md) |
| `habit retest` | Test-retest mapping | `config_habitat_test_retest.yaml` | [📖](doc_en/app_habitat_test_retest.md) |

#### Advantages

✅ **Unified & Clean** - All features accessible via `habit` command  
✅ **Ready to Use** - No path configuration needed after installation  
✅ **Built-in Help** - Every command has a `--help` option  
✅ **Colored Output** - Clear success/error messages  
✅ **Parameter Validation** - Automatic checking of required parameters  

📚 **Complete CLI Guide**: See [**HABIT_CLI.md**](HABIT_CLI.md) for comprehensive command-line documentation, including installation instructions, troubleshooting, and advanced usage.

---


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
1.  Read the detailed documentation in the `doc_en/` folder.
2.  Open an [Issue](https://github.com/your-repo/habit_project/issues) on GitHub.

### 📖 Multilingual Documentation

HABIT provides complete bilingual documentation in Chinese and English:
- **Chinese Documentation**: Located in `doc/` directory
- **English Documentation**: Located in `doc_en/` directory

💡 **Language Switch**: Click the "🇬🇧 English" or "🇨🇳 简体中文" link at the top of the page to quickly switch languages.

---

# Installation Guide

# HABIT Installation Guide

This guide provides detailed instructions for installing the HABIT toolkit and its dependencies.

---

## 1. System Requirements

-   **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS 10.15+.
-   **Python**: Version 3.8, 3.9, or 3.10 are recommended.
-   **Memory (RAM)**: Minimum 16 GB, **32 GB or more is highly recommended** for processing large datasets.
-   **Storage**: At least 10 GB of free disk space.

## 2. External Dependencies

Before installing the Python packages, you must install these external tools:

### A. Conda

It is **highly recommended** to use `conda` (from Anaconda or Miniconda) for environment management.
-   Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### B. Git

Required for cloning the project repository.
-   Install `git` from the [official website](https://git-scm.com/downloads).

### C. dcm2niix (Required for DICOM conversion)

If you plan to convert DICOM images to NIfTI, `dcm2niix` is required.
1.  Go to the [dcm2niix GitHub releases page](https://github.com/rordenlab/dcm2niix/releases).
2.  Download the pre-compiled version for your operating system.
3.  Extract the executable (`dcm2niix.exe` on Windows) and add its location to your system's **PATH environment variable**.
4.  Verify by opening a new terminal and running: `dcm2niix --version`.

### D. R Language (Optional)

An R installation is required **only if you plan to use the `stepwise` feature selection method** in the machine learning pipeline.
1.  Download and install R from the [official R-project website](https://cran.r-project.org/).
2.  During installation, take note of the installation path.
3.  You may need to specify this path in your machine learning configuration file.

## 3. Installation Steps

This is the recommended installation procedure using Conda.

### Step 1: Clone the Repository

Open a terminal (or Anaconda Prompt on Windows) and run:
```bash
git clone <repository_url>
cd habit_project
```

### Step 2: Create and Activate Conda Environment

Create a dedicated environment for HABIT to avoid dependency conflicts.
```bash
# Create an environment named 'habit' with Python 3.8
conda create -n habit python=3.8

# Activate the new environment
conda activate habit
```

### Step 3: Install Python Dependencies

Install all required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### Step 4: Install HABIT Package

Finally, install the HABIT toolkit itself in "editable" mode. This allows you to modify the source code without needing to reinstall.
```bash
pip install -e .
```

## 4. Verifying the Installation

To ensure everything is set up correctly, run the following checks from your terminal (with the `habit` environment activated).

1.  **Check basic package import:**
    ```bash
    python -c "import habit; print(f'HABIT version {habit.__version__} installed successfully!')"
    ```

2.  **Check core module availability:**
    ```bash
    python -c "from habit.utils.import_utils import check_dependencies; check_dependencies(['SimpleITK', 'antspyx', 'torch', 'sklearn', 'pyradiomics'])"
    ```
    This should report that all listed modules are available.

3.  **Check script entry points:**
    ```bash
    habit get-habitat --help
    ```
    This should display the help menu for the main analysis script.

## 5. Troubleshooting

-   **`antspyx` or `SimpleITK` installation fails**: These packages can sometimes have compilation issues. Try installing them separately with `conda` before running `pip install -r requirements.txt`:
    ```bash
    conda install -c conda-forge antspyx simpleitk -y
    ```

-   **R-related errors for `stepwise` selection**: If you see errors related to `rpy2` or R, ensure R is installed correctly and that the `Rhome` path in your configuration file (e.g., `config/config_machine_learning.yaml`) points to the correct R installation directory if needed.

-   **Memory Errors**: If you encounter `MemoryError` during analysis, try reducing the `processes` number in your YAML configuration file.

-   **CUDA/GPU Errors**: If you have a compatible NVIDIA GPU and want to use it, ensure you have the correct NVIDIA driver and CUDA Toolkit version installed. Then, install the GPU-enabled version of PyTorch by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

---

Your installation is now complete. Proceed to the [**QUICKSTART.md**](QUICKSTART.md) guide to run your first analysis.