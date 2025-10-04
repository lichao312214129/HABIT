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
    python scripts/app_getting_habitat_map.py --help
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
