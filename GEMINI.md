# HABIT (Habitat Analysis: Biomedical Imaging Toolkit)

## Project Overview
**HABIT** is a comprehensive Python toolkit designed for tumor "habitat" analysis in medical imaging. It facilitates the entire radiomics pipeline, from raw image preprocessing to advanced machine learning modeling, enabling researchers to explore tumor heterogeneity.

The core concept involves identifying sub-regions (habitats) within tumors that exhibit distinct radiological phenotypes, extracting features from these habitats, and using them for clinical predictions.

### Key Features
*   **Image Preprocessing:** DICOM-to-NIfTI conversion, resampling, registration, intensity normalization, and N4 bias field correction.
*   **Habitat Analysis:** Supports three clustering strategies:
    *   **One-Step:** Direct voxel-to-habitat clustering (personalized).
    *   **Two-Step (Default):** Voxel -> Supervoxel -> Habitat (cohort-consistent).
    *   **Direct Pooling:** All voxels pooled -> Habitat (cohort-consistent).
*   **Feature Extraction:** Extracts traditional radiomics, Multiregional Spatial Interaction (MSI), and Intratumoral Heterogeneity (ITH) features.
*   **Machine Learning:** End-to-end workflow for feature selection, model training (XGBoost, etc.), and evaluation.
*   **Validation:** Tools for Inter-Class Correlation (ICC) and Test-Retest reliability analysis.
*   **Visualization:** Automated generation of 2D/3D habitat maps and statistical plots.

## Architecture & Structure
The project is structured as a Python package with a CLI interface.

*   **`habit/`**: Main source code package.
    *   `cli.py` & `cli_commands/`: Implementation of the `habit` CLI using `click`.
    *   `core/`: Core logic for preprocessing, habitat analysis, and ML.
    *   `utils/`: Utility functions for I/O, config parsing, and visualization.
*   **`config/`**: Default configuration files (YAML) for various pipeline stages.
*   **`demo_data/`**: Complete set of sample data (DICOM/NIfTI) and configs for quick testing.
*   **`doc/` & `doc_en/`**: Documentation in Chinese and English.
*   **`scripts/`**: Standalone scripts (likely precursors to the CLI or for specific tasks).
*   **`tests/`**: Unit and integration tests.

## Development & Installation

### Prerequisites
*   Python 3.8+ (Python 3.10 recommended for `autogluon` support)
*   Conda (recommended)

### Installation
1.  **Create Environment:**
    ```bash
    conda create -n habit python=3.8
    conda activate habit
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install Package (Editable Mode):**
    ```bash
    pip install -e .
    ```

## Usage
The project is primarily used via the `habit` command-line interface. All commands require a configuration file (YAML).

### Basic Command Structure
```bash
habit [COMMAND] --config [CONFIG_FILE] [OPTIONS]
```

### Key Commands
| Task | Command | Config Example |
| :--- | :--- | :--- |
| **Preprocessing** | `habit preprocess` | `config/config_image_preprocessing.yaml` |
| **Habitat Analysis** | `habit get-habitat` | `demo_data/config_habitat.yaml` |
| **Feature Extraction** | `habit extract` | `config/config_extract_features.yaml` |
| **Machine Learning** | `habit model` | `config/config_machine_learning.yaml` |
| **Cross-Validation** | `habit cv` | `config/config_machine_learning_kfold.yaml` |
| **Model Comparison** | `habit compare` | `config/config_model_comparison.yaml` |
| **ICC Analysis** | `habit icc` | `config/config_icc_analysis.yaml` |

### Quick Start (Demo)
To run a full habitat analysis on the provided demo data:
```bash
habit get-habitat --config demo_data/config_habitat.yaml
```
Results will be generated in `demo_data/results/habitat/`.

## Configuration
Configuration is central to HABIT. Users should modify the YAML files in `config/` or `demo_data/` to point to their data and adjust parameters (e.g., clustering method, feature sets).

*   **`data_dir`**: Path to input images.
*   **`out_dir`**: Path for output.
*   **`FeatureConstruction.voxel_level.method`**: Feature extraction algorithm.
*   **`HabitatsSegmention.clustering_mode`**: `one_step`, `two_step`, or `direct_pooling`.

## Testing
The project uses `pytest`.
```bash
pytest
```
See `tests/README.md` for more details.

## Technologies
*   **Core:** Python, Click, Poetry
*   **Imaging:** SimpleITK, ANTsPy, PyRadiomics, OpenCV
*   **Data/ML:** NumPy, Pandas, Scikit-learn, XGBoost, SciPy, Statsmodels
*   **Visualization:** Matplotlib, Seaborn
