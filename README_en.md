# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

<p align="center">
  <img src="https://www.imagilys.com/wp-content/uploads/2018/09/radiomics-illustration-500x500.png" alt="Radiomics" width="200"/>
</p>

<p align="center">
  <strong>📖 Language / 语言</strong><br>
  <a href="README_en.md">🇬🇧 English</a> | <a href="README.md">🇨🇳 简体中文</a>
</p>

<p align="center">
    <a href="https://github.com/lichao312214129/HABIT/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Active-green.svg"></a>
</p>

## What is HABIT?

**HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** is a comprehensive toolkit for tumor habitat analysis in medical imaging.

**What are "Habitats"?**
Tumors are not homogeneous; they consist of multiple sub-regions with distinct radiological phenotypes, known as "habitats." For example, a tumor may contain necrotic zones, active tumor zones, edema zones, etc., each with different treatment responses and prognoses. HABIT can automatically identify these regions and extract their features, helping clinicians and researchers better understand tumor heterogeneity.

**Why Habitat Analysis?**
- Tumor heterogeneity is a key factor affecting treatment outcomes and prognosis
- Traditional radiomics only considers average features of the entire tumor, ignoring internal differences
- Habitat analysis can reveal complex structures within tumors, providing more accurate predictive models

---

## 📖 Conceptual Workflow

**Image + ROI → Voxel Features → Supervoxels (Optional) → Supervoxel Features (Optional) → Habitats → Habitat Features → Predictive Model (Optional)**

---

## ⚡ 5-Minute Quick Start

**No data preparation needed, try it now!**

```bash
# 1. Install HABIT (if not already installed)
git clone https://github.com/lichao312214129/HABIT.git
cd HABIT
conda create -n habit python=3.8
conda activate habit
pip install -r requirements.txt
pip install -e .

# 2. Run example analysis
habit get-habitat --config demo_data/config_habitat.yaml

# 3. View results
# Results are saved in demo_data/results/habitat/ directory
# Includes: habitat maps, visualization charts, CSV result files
```

**Expected Results**:
- `habitats.csv` - Habitat labels for each voxel
- `*_habitats.nrrd` - 3D habitat maps (viewable with ITK-SNAP or 3D Slicer)
- `visualizations/` - Automatically generated visualization charts

---

## 🎯 Use Cases

HABIT can be used for the following clinical and research scenarios:

- **Tumor Heterogeneity Assessment**: Quantify the degree of heterogeneity within tumors
- **Prognosis Prediction**: Predict patient survival based on habitat features
- **Treatment Response Prediction**: Predict patient response to specific treatments
- **Treatment Efficacy Evaluation**: Compare habitat changes before and after treatment
- **Biomarker Discovery**: Discover new imaging biomarkers

---

## 📋 Data Requirements

### Input Data
- **Image Formats**: DICOM or NIfTI (.nii/.nii.gz)
- **ROI Files**: NIfTI format tumor segmentation masks
- **Data Volume**: Supports single case to large cohort studies

### Output Results
- **Habitat Maps**: 3D images with different colors representing different habitats
- **Feature Files**: CSV format containing detailed features for each habitat
- **Visualization Charts**: PNG format clustering results and statistical charts

---

## 🛠️ Installation

### System Requirements
- **Operating System**: Windows, macOS, Linux
- **Python Version**: 3.8 or higher
- **Memory**: Recommended 8GB or more (depends on data volume)
- **Dependencies**: See [requirements.txt](requirements.txt)

### Detailed Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/lichao312214129/HABIT.git
cd HABIT

# 2. Create Conda environment
conda create -n habit python=3.8
conda activate habit

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install HABIT
pip install -e .

# 5. Verify installation
habit --help
```

**Common Issues**:
- If `conda` command is not found, please install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first
- If `pip install` fails, try upgrading pip: `pip install --upgrade pip`
- If you encounter permission issues, use `--user` flag: `pip install --user -r requirements.txt`

---

## 📖 Full Documentation

For detailed usage guides and API documentation, please see:

- **Online Docs**: [https://lichao312214129.github.io/HABIT](https://lichao312214129.github.io/HABIT)

### Documentation Contents

- **Getting Started**: Installation, configuration, quick testing
- **User Guide**: Image preprocessing, habitat analysis, feature extraction, machine learning
- **Configuration Reference**: Complete configuration file documentation
- **API Reference**: Detailed documentation of modules and classes
- **Development Guide**: Architecture design, contribution guide

### Common Links

- Image Preprocessing: [Documentation - Image Preprocessing](https://lichao312214129.github.io/HABIT/user_guide/app_image_preprocessing.html)
- Habitat Analysis: [Documentation - Habitat Analysis](https://lichao312214129.github.io/HABIT/user_guide/habit_analysis.html)
- Feature Extraction: [Documentation - Feature Extraction](https://lichao312214129.github.io/HABIT/user_guide/feature_extraction.html)
- Machine Learning: [Documentation - Machine Learning](https://lichao312214129.github.io/HABIT/user_guide/machine_learning.html)

---

## 🚀 Key Features

| Category | Feature | Description |
| :--- | :--- | :--- |
| 🖼️ **Image Processing** | **Preprocessing Pipeline** | Provides DICOM conversion, resampling, registration, standardization, and N4 bias field correction |
| 🧬 **Habitat Analysis** | **Clustering Strategies** | Supports one-step, two-step, and direct pooling clustering strategies |
| 🔬 **Feature Extraction** | **Advanced Feature Set** | Extracts traditional radiomics, multi-scale spatial interaction (MSI), and intra-tumor heterogeneity (ITH) features |
| 🤖 **Machine Learning** | **Complete Workflow** | Includes data splitting, feature selection, model training, and evaluation |
| 📊 **Validation & Tools** | **Reproducibility Analysis** | Includes test-retest and intraclass correlation coefficient (ICC) analysis tools |

---

## ❓ Frequently Asked Questions (FAQ)

### Q1: Do I need programming skills?
A: No. HABIT provides complete command-line tools; you only need to modify configuration files to use them. However, if you have Python basics, you can perform more flexible customizations.

### Q2: Which imaging modalities are supported?
A: HABIT supports all medical imaging modalities, including CT, MRI, PET, etc.

### Q3: How long does analysis take?
A: Depends on data volume and computing resources. Habitat analysis for a single case typically takes 5-15 minutes.

### Q4: How do I interpret results?
A: HABIT generates visualization results and detailed feature files. Habitat maps can be viewed with ITK-SNAP or 3D Slicer, and feature files can be opened and analyzed with Excel.

### Q5: Can it be used for clinical research?
A: Yes. HABIT has been used in multiple clinical studies, including tumor prognosis prediction, treatment response evaluation, etc.

---

## 🤝 Contributing

Contributions of all forms are welcome! Please refer to contribution guide or open an Issue to discuss your ideas.

---

## 📄 License

This project is licensed under MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔬 Citation

If you use HABIT in your research, please consider citing:
> [Citation information to be added]

---

## 🙋‍♀️ Support

If you encounter any issues or have suggestions for improvement, please:
1. Read full documentation
2. Check [FAQ](#frequently-asked-questions-faq)
3. Submit an [Issue](https://github.com/lichao312214129/HABIT/issues) on GitHub