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

---

## 📖 Conceptual Workflow

The core idea of HABIT is to identify and characterize sub-regions within a tumor, known as "habitats," which have distinct radiological phenotypes. This is achieved through a multi-stage pipeline:

<p align="center">
  <b>Image → Voxel Features → Supervoxels (Optional) → Habitats → Habitat Features → Predictive Model (Optional)</b>
</p>

### Core Concept Hierarchy
*Abstraction process from micro-voxels to macro-habitats*

```
      [Tumor Image]          [Micro-Level]          [Meso-Level]          [Macro-Level]
     +------------+         +------------+         +------------+         +------------+
     |   Tumor    |         |   Voxels   |         | Supervoxels|         |  Habitats  |
     |  (Region)  |         | (Features) |         | (Clusters) |         | (Patterns) |
     +-----+------+         +-----+------+         +-----+------+         +-----+------+
           |                      |                      |                      |
           v                      v                      v                      v
     +------------+         +------------+         +------------+         +------------+
     |            |         | . . . . . .|         | AA BB CC DD|         | ## ** @@   |
     |  (Image)   |  ---->  | . . . . . .|  ---->  | AA BB CC DD|  ---->  | ## ** @@   |
     |            |         | . . . . . .|         | EE FF GG HH|         | $$ %% &&   |
     +------------+         +------------+         +------------+         +------------+
      Original Image        Voxel Features          Supervoxels           Habitat Map
                                                  (Over-segmentation)    (Biological Meaning)
```

### Detailed Workflow

1. **Voxel-Level Feature Extraction**: Extracts rich features (e.g., intensity, texture, kinetic) for every single voxel within the tumor.
2. **Supervoxel Clustering**: Groups spatially adjacent voxels with similar features into "supervoxels." This over-segmentation step simplifies the image while preserving local boundaries.
3. **Habitat Clustering**: Groups the supervoxels across a patient cohort to identify common, recurring patterns, forming the final "habitats."
4. **Feature Engineering**: Extracts high-level features from these habitats, such as their size, shape, spatial relationships (MSI features), and heterogeneity (ITH score).
5. **Machine Learning**: Uses the engineered habitat features to train predictive models for clinical endpoints like patient survival, treatment response, or diagnosis.

### Three Clustering Strategies

HABIT supports three different clustering strategies, each suitable for different research scenarios:

#### 1️⃣ One-Step Clustering
- **Process**: Voxel → Habitat (direct clustering)
- **Characteristics**: Each patient independently determines optimal cluster count, independent habitat labels
- **Use Cases**: Individual heterogeneity analysis, small sample studies, personalized analysis per patient

#### 2️⃣ Two-Step Clustering ⭐ Default Method
- **Process**: Voxel → Supervoxel → Habitat
  - **Step 1**: Cluster voxels for each patient to generate supervoxels (e.g., 50 supervoxels per patient)
  - **Step 2**: Pool supervoxels from all patients and perform population-level clustering to identify unified habitat patterns
- **Characteristics**: Individual clustering first, then population clustering, all patients share unified habitat labels
- **Use Cases**: Cohort studies, cross-patient habitat pattern recognition, unified labeling for comparison

#### 3️⃣ Direct Pooling
- **Process**: Concatenate all voxels from all patients → Direct population clustering
- **Characteristics**: Skips supervoxel step, directly clusters all voxels at population level, all patients share unified labels
- **Use Cases**: Moderate data size, need unified labels but don't need supervoxel intermediate step

### 🔍 Visual Comparison of Clustering Strategies

#### 1. One-Step Clustering (Personalized)
*Independent clustering for each patient, suitable for individual heterogeneity.*

```
      Patient 1 (P1)              Patient 2 (P2)
   +------------------+        +------------------+
   |  P1 Tumor Image  |        |  P2 Tumor Image  |
   +--------+---------+        +--------+---------+
            |                           |
            v (Extract Voxels)          v
   +--------+---------+        +--------+---------+
   | Voxels: . . . .  |        | Voxels: . . . .  |
   +--------+---------+        +--------+---------+
            |                           |
            v (Cluster Individually)    v
   +--------+---------+        +--------+---------+
   | Habitats: # * @  |        | Habitats: & % $  |
   +------------------+        +------------------+
     Unique to P1                Unique to P2
    (Labels Not Shared)         (Labels Not Shared)
```

#### 2. Two-Step Clustering (Cohort Study) ⭐ Recommended
*Cluster voxels to supervoxels first, then cluster populations. Balances local detail with population consistency.*

```
      Patient 1 (P1)              Patient 2 (P2)
   +------------------+        +------------------+
   |  P1 Tumor Image  |        |  P2 Tumor Image  |
   +--------+---------+        +--------+---------+
            |                           |
            v                           v
   +--------+---------+        +--------+---------+
   | Voxels: . . . .  |        | Voxels: . . . .  |
   +--------+---------+        +--------+---------+
            | (Local Clustering)        |
            v                           v
   +--------+---------+        +--------+---------+
   | Supervoxels:     |        | Supervoxels:     |
   | AA BB CC DD      |        | EE FF GG HH      |
   +--------+---------+        +--------+---------+
            \                         /
             \  (Pool All Supervoxels) /
              \                     /
               v                   v
           +---------------------------+
           |   Population Clustering   |
           |   (Cluster Supervoxels)   |
           +-------------+-------------+
                         |
                         v
           +---------------------------+
           |  Unified Habitats (Shared)|
           |  Type 1: # (e.g. Necrosis)|
           |  Type 2: * (e.g. Active)  |
           |  Type 3: @ (e.g. Edema)   |
           +---------------------------+
             (Consistent Labels for Cohort)
```

#### 3. Direct Pooling
*Skip supervoxels, directly cluster all voxels from population.*

```
      Patient 1 (P1)              Patient 2 (P2)
   +------------------+        +------------------+
   |  P1 Tumor Image  |        |  P2 Tumor Image  |
   +--------+---------+        +--------+---------+
            |                           |
            v                           v
   +--------+---------+        +--------+---------+
   | Voxels: . . . .  |        | Voxels: . . . .  |
   +--------+---------+        +--------+---------+
             \                         /
              \  (Pool All Voxels)    /
               \                     /
                v                   v
           +---------------------------+
           |   Population Clustering   |
           |    (Cluster All Voxels)   |
           +-------------+-------------+
                         |
                         v
           +---------------------------+
           |  Unified Habitats (Shared)|
           |     Type 1: #, 2: *, 3: @ |
           +---------------------------+
```


### 📊 Strategy Selection Guide

**Choose One-Step if:**
- You want to analyze each tumor individually
- Sample sizes vary greatly between patients
- You're interested in personalized habitat patterns
- Computational resources are limited

**Choose Two-Step if:**
- You're conducting a cohort study
- You need comparable habitats across patients ⭐ **Most studies**
- You want to balance computational efficiency with biological relevance
- You need interpretable intermediate results (supervoxels)

**Choose Direct Pooling if:**
- You have moderate computational resources
- You want unified habitats but don't need supervoxel intermediate step
- You're working with datasets where voxel-level clustering is feasible

**Comparison Table**:

| Feature | One-Step | Two-Step | Direct Pooling |
|---------|----------|----------|----------------|
| **Clustering Process** | Voxel→Habitat | Voxel→Supervoxel→Habitat | Pool all voxels→Habitat |
| **Clustering Level** | Single (individual) | Two-level (individual+population) | Single (population) |
| **Habitat Labels** | Independent per patient | Unified across patients | Unified across patients |
| **Computational Cost** | Low | Medium | High (depends on total voxels) |
| **Use Cases** | Individual heterogeneity | Cohort studies (recommended) | Moderate-scale data |

---

## 🧪 Quick Test (Using Demo Data)

**🎯 Important**: HABIT provides complete sample data—you can quickly experience all features without preparing your own data!

### Quick Start with Demo Data

The `demo_data/` directory in the project contains:
- ✅ Sample DICOM imaging data (2 subjects)
- ✅ Preprocessed images and masks
- ✅ Complete example configuration files
- ✅ Sample analysis results

### Three-Step Quick Experience

```bash
# 1. Ensure HABIT is installed (see Installation section below)
# 2. Activate environment
conda activate habit

# 3. Run Habitat analysis using demo data
habit get-habitat --config demo_data/config_habitat.yaml
```

**Expected Results**:
- After analysis completes, results will be saved in `demo_data/results/habitat/` directory
- You will see:
  - `habitats.csv` - Habitat label results
  - `subj001_habitats.nrrd` and `subj002_habitats.nrrd` - Habitat maps (viewable with ITK-SNAP or 3D Slicer)
  - `visualizations/` - Automatically generated visualization charts
  - `supervoxel2habitat_clustering_strategy_bundle.pkl` - Trained model

### Reference Example Configuration Files

All example configuration files are in the `demo_data/` directory:
- `config_habitat.yaml` - Habitat analysis configuration (recommended starting point)
- `config_preprocessing.yaml` - Image preprocessing configuration
- `config_icc.yaml` - ICC analysis configuration

**💡 Tip**: You can copy these configuration files and modify paths and parameters according to your own data.

---

## 🛠️ Installation

A detailed guide is available in [**INSTALL.md**](INSTALL.md).

### Quick Installation Steps

```bash
# 1. Clone the repository
git clone <repository_url>
cd habit_project

# 2. Create and activate Conda environment
conda create -n habit python=3.8
conda activate habit

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the HABIT package in editable mode
pip install -e .
```

### Verify Installation

```bash
# Check if command is available
habit --help

# If you see the command list, installation is successful!
```

---

## 📖 Quick Start

### 🎯 Unified Command Line Interface (CLI) - **Recommended Usage**

**HABIT provides a unified, streamlined command-line interface!** ✨

Using a CLI system built with **Click**, you only need the `habit` command to access all functionality—no need to remember complex script paths.

#### Ready to Use After Installation

After completing `pip install -e .`, the `habit` command is globally available in your environment:

```bash
# View all available commands
habit --help

# Get help for specific commands
habit get-habitat --help
```

#### Core Command Examples

```bash
# 1️⃣ Image Preprocessing - Resampling, registration, normalization
habit preprocess --config config/config_image_preprocessing.yaml

# 2️⃣ Generate Habitat Maps - Identify tumor sub-regions
# Supports one-step, two-step, or direct pooling methods
habit get-habitat --config demo_data/config_habitat.yaml

# 3️⃣ Extract Habitat Features - MSI, ITH, and other advanced features
habit extract --config config/config_extract_features.yaml

# 4️⃣ Machine Learning - Train predictive models
habit model --config config/config_machine_learning.yaml --mode train

# 5️⃣ Model Prediction - Use trained models
habit model --mode predict \
  --model ./ml_data/model_package.pkl \
  --data ./new_data.csv \
  --output ./predictions/

# 6️⃣ K-Fold Cross-Validation - Robust model evaluation
habit cv --config config/config_machine_learning_kfold.yaml

# 7️⃣ Model Comparison - ROC, DCA, calibration curves visualization
habit compare --config config/config_model_comparison.yaml

# 8️⃣ ICC Analysis - Feature reproducibility assessment
habit icc --config config/config_icc_analysis.yaml
```

#### Quick Reference Table

| Command | Function | Example Config File | Docs |
|---------|----------|---------------------|:---:|
| `habit preprocess` | Image preprocessing | `config_image_preprocessing.yaml` | [📖](doc_en/app_image_preprocessing.md) |
| `habit get-habitat` | Generate Habitat maps | `demo_data/config_habitat.yaml` ⭐ | [📖](doc_en/app_habitat_analysis.md) |
| `habit extract` | Extract Habitat features | `config_extract_features.yaml` | [📖](doc_en/app_extracting_habitat_features.md) |
| `habit model` | ML training/prediction | `config_machine_learning.yaml` | [📖](doc_en/app_of_machine_learning.md) |
| `habit cv` | K-fold cross-validation | `config_machine_learning_kfold.yaml` | [📖](doc_en/app_kfold_cross_validation.md) |
| `habit compare` | Model comparison & viz | `config_model_comparison.yaml` | [📖](doc_en/app_model_comparison_plots.md) |
| `habit icc` | ICC reproducibility | `config_icc_analysis.yaml` | [📖](doc_en/app_icc_analysis.md) |

---

## 🔬 Complete Research Workflow

A typical radiomics research project using HABIT involves the following steps. The HABIT toolkit provides powerful support for the steps marked with `[HABIT]`.

1. **Data Acquisition and Download**: Obtain original imaging data (usually in DICOM format) from a hospital PACS or a public dataset.
2. **Data Organization and Anonymization**: Organize the data into a `Patient/Sequence/Files` structure, anonymize patient-sensitive information.
3. **Format Conversion (DICOM to NIfTI)**: `[HABIT]` Use the `habit preprocess` command to convert DICOM series into NIfTI format.
4. **Region of Interest (ROI) Segmentation**: A radiologist or researcher manually delineates the tumor region (ROI) using specialized software like ITK-SNAP or 3D Slicer, saving it as a mask file.
5. **Image Preprocessing**: `[HABIT]` Use the `habit preprocess` command for registration, resampling, intensity normalization, N4 bias field correction, etc.
6. **Habitat Analysis and Feature Extraction**: 
   - `[HABIT]` Run `habit get-habitat` to identify tumor habitats (supports one-step, two-step, or direct pooling methods)
   - `[HABIT]` Run `habit extract` to extract high-level features (e.g., MSI, ITH score) from the habitats
7. **Building and Evaluating Predictive Models**: 
   - `[HABIT]` Use `habit model` for feature selection, model training, and internal validation
   - `[HABIT]` Use `habit compare` to compare the performance of different models and visualize the results
8. **Result Analysis and Manuscript Writing**: Interpret the findings from the model and write the research paper.

---

## 🚀 Key Features

| Category | Feature | Description | Docs |
| :--- | :--- | :--- | :---: |
| 🖼️ **Image Processing** | **Preprocessing Pipeline** | End-to-end tools for DICOM conversion, resampling, registration, normalization, and N4 bias field correction. | [📖](doc_en/app_image_preprocessing.md) |
| 🧬 **Habitat Analysis** | **One-Step Clustering** | Direct clustering to habitats, cluster count determined per tumor, independent habitat labels. | [📖](doc_en/app_habitat_analysis.md) |
| | **Two-Step Clustering** | Two-stage clustering (individual supervoxels → population habitats), unified habitat labeling system. | [📖](doc_en/app_habitat_analysis.md) |
| | **Direct Pooling** | Pool all voxels and cluster directly, skipping supervoxel step. | [📖](doc_en/app_habitat_analysis.md) |
| | **🎨 Auto Visualization** | Automatically generates high-quality 2D/3D clustering scatter plots, optimal cluster number curves, etc. | [📖](doc_en/app_habitat_analysis.md) |
| 🔬 **Feature Extraction** | **Advanced Feature Sets** | Extracts traditional radiomics, Multiregional Spatial Interaction (MSI), and Intratumoral Heterogeneity (ITH) features. | [📖](doc_en/app_extracting_habitat_features.md) |
| 🤖 **Machine Learning** | **Complete Workflow** | Includes data splitting, feature selection, model training, and evaluation. | [📖](doc_en/app_of_machine_learning.md) |
| | **K-Fold Cross-Validation** | Comprehensive K-fold cross-validation with multi-model evaluation and visualization. | [📖](doc_en/app_kfold_cross_validation.md) |
| | **Model Comparison** | Tools to generate ROC curves, Decision Curve Analysis (DCA), and perform DeLong tests. | [📖](doc_en/app_model_comparison_plots.md) |
| 📊 **Validation & Utilities** | **Reproducibility Analysis** | Includes tools for Test-Retest and Inter-Class Correlation (ICC) analysis. | [📖](doc_en/app_icc_analysis.md) |

---

## ❓ Frequently Asked Questions

### Q1: How do I get started with HABIT?

**Recommended approach**: Use the sample data in `demo_data` for a quick experience!

```bash
# 1. Ensure installation (see Installation section)
conda activate habit

# 2. Run example
habit get-habitat --config demo_data/config_habitat.yaml

# 3. Check results
# Results are in demo_data/results/habitat/ directory
```

### Q2: Command `habit` not found?

**Solution**:
```bash
# Ensure correct environment is activated
conda activate habit

# Reinstall
pip install -e .

# Verify installation
habit --help
```

### Q3: How do I modify configuration files?

**Recommended approach**:
1. Copy `demo_data/config_habitat.yaml` as a template
2. Modify paths and parameters in it
3. Main parameters to modify:
   - `data_dir`: Your data path
   - `out_dir`: Output results path
   - `FeatureConstruction.voxel_level.method`: Feature extraction method
   - `HabitatsSegmention.clustering_mode`: Choose clustering strategy (one_step/two_step/direct_pooling)

### Q4: How do I view analysis results?

**Result locations**:
- CSV files: `{out_dir}/habitats.csv` - Can be opened with Excel
- Image files: `{out_dir}/*_habitats.nrrd` - Viewable with ITK-SNAP or 3D Slicer
- Visualization charts: `{out_dir}/visualizations/` - PNG format, directly viewable

### Q5: How to choose among the three clustering strategies?

- **One-Step**: Suitable when each patient needs personalized analysis, large sample differences
- **Two-Step**: Suitable for cohort studies, need unified labels for comparison (**recommended for most studies**)
- **Direct Pooling**: Suitable for moderate data size, need unified labels but don't need supervoxel intermediate step

### Q6: How to understand output results?

- **habitats.csv**: Contains habitat labels for each supervoxel (or voxel)
- **Habitat maps**: 3D images, different colors represent different habitats
- **Visualization charts**: Help understand clustering effectiveness and optimal cluster numbers

---

## 🤝 Contributing

Contributions are welcome! Please refer to the contribution guidelines (to be added) or open an issue to discuss your ideas.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Citation

If you use HABIT in your research, please consider citing:
> [Citation details to be added here]

## 🙋‍♀️ Support

If you encounter any problems or have suggestions, please:
1. Read the detailed documentation in the `doc_en/` folder
2. Open an [Issue](https://github.com/your-repo/habit_project/issues) on GitHub

### 📖 Multilingual Documentation

HABIT provides complete bilingual documentation in Chinese and English:
- **Chinese Documentation**: Located in `doc/` directory
- **English Documentation**: Located in `doc_en/` directory

💡 **Language Switch**: Click the "🇬🇧 English" or "🇨🇳 简体中文" link at the top of the page to quickly switch languages.

---

**Happy using!** 🎉
