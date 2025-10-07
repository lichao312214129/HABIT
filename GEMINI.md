# Gemini Project Companion: HABIT

This document provides a summary of the HABIT project to help Gemini (the AI assistant) understand the codebase and assist with development tasks.

## 1. Project Summary

**HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** is a Python-based, end-to-end toolkit for tumor habitat analysis in medical imaging. It enables researchers to investigate tumor heterogeneity through radiomics and machine learning.

The core workflow is:
**Image → Voxel Features → Supervoxels → Habitats → Habitat Features → Predictive Model**

- **Technology Stack**: Python 3.8+
- **Core Libraries**: PyRadiomics, SimpleITK, scikit-learn, pandas, numpy.
- **Configuration**: Driven by YAML files located in the `config/` directory.
- **Execution**: Performed via Python scripts in the `scripts/` directory.

## 2. Project Structure

- `habit/`: The core Python source code package.
    - `core/`: Main modules for analysis.
        - `habitat_analysis/`: Habitat identification logic.
        - `machine_learning/`: ML modeling and evaluation.
        - `preprocessing/`: Image processing functions.
    - `utils/`: Helper utilities (I/O, logging, etc.).
- `scripts/`: Entry-point scripts for running analyses. These are the main executables for the user.
- `config/`: YAML configuration files for all scripts.
- `doc/` & `doc_en/`: Detailed documentation for each module.
- `requirements.txt`: Python dependencies.
- `INSTALL.md`: Detailed installation guide.
- `QUICKSTART.md`: A 5-minute tutorial for new users.

## 3. Key Files & Scripts

- **Configuration Files (`config/`)**:
    - `config_getting_habitat.yaml`: Main configuration for habitat analysis.
    - `config_extract_features.yaml`: Configuration for extracting high-level habitat features.
    - `config_machine_learning.yaml`: Configuration for training and evaluating machine learning models.
    - `image_files.yaml`: Alternative to directory-based data structure, allows specifying file paths.

- **Main Scripts (`scripts/`)**:
    - `app_getting_habitat_map.py`: Core script to identify tumor habitats.
    - `app_extracting_habitat_features.py`: Extracts features (MSI, ITH) from habitats.
    - `app_of_machine_learning.py`: Trains and evaluates predictive models.
    - `app_image_preprocessing.py`: Pipeline for image preprocessing (conversion, registration, normalization).

## 4. Common Workflows

All workflows are initiated by running a script from the `scripts/` directory with a corresponding configuration file.

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

## 5. Instructions for Gemini

- **Code Style**: Follow PEP 8 for Python. Adhere to the existing code style in the project.
- **Configuration**: When asked to perform a task, first inspect the relevant YAML configuration file in the `config/` directory.
- **Execution**: Use the scripts in the `scripts/` directory to run analyses. Do not directly call the functions in the `habit/` package unless necessary for a specific task.
- **File Paths**: When modifying configuration files, be mindful of file paths. The user might be on Windows or Linux. Use forward slashes for paths to maintain compatibility, or use `os.path.join`.
- **Dependencies**: If new dependencies are needed, add them to `requirements.txt`.
- **Linting/Formatting**: The project uses `.pre-commit-config.yaml`. Before committing, it would be good to run `pre-commit run --all-files`.
- **Documentation**: If new features are added, update the corresponding documentation in the `doc/` and `doc_en/` directories.

## 6. Gemini Capabilities (IDE-Connected)

When connected to the IDE, Gemini acts as a hands-on programming assistant with the following capabilities:

- **Full Project Access**: Gemini can read and understand the entire project structure and file contents, providing context-aware assistance.
- **Direct Action**: Gemini can directly perform actions like:
    - **Writing and modifying code**.
    - **Creating, deleting, and managing files**.
    - **Executing shell commands** (e.g., running scripts, installing dependencies, running tests, using git).
- **Seamless Workflow**: Gemini is integrated into the development environment, allowing for a smooth and efficient workflow without context switching.
- **Verification**: Gemini can run tests and linters to verify its own work, ensuring code quality and correctness.

In essence, when connected to the IDE, Gemini transitions from a "knowledge consultant" to a "developer assistant" that can actively participate in the development lifecycle.
