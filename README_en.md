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

**HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** is a comprehensive, Python-based toolkit for tumor habitat analysis in medical imaging. It provides an end-to-end solution, from image preprocessing to machine learning, enabling researchers to investigate tumor heterogeneity through radiomics and advanced analytics.

---

## 📖 Conceptual Workflow

The core idea of HABIT is to identify and characterize sub-regions within a tumor, known as "habitats," which have distinct radiological phenotypes.

**Image → Voxel Features → Supervoxels (Optional) → Habitats → Habitat Features → Predictive Model (Optional)**

## 🧪 Quick Test

**HABIT provides complete sample data, so you can quickly experience all features without preparing your own data!**

```bash
# 1. Ensure HABIT is installed (see installation guide below)
conda activate habit

# 2. Run Habitat analysis with sample data
habit get-habitat --config demo_data/config_habitat.yaml
```

**Expected results**: After analysis completes, results will be saved in the `demo_data/results/habitat/` directory.

**More usage examples**:
- Image Preprocessing: See [Documentation - Image Preprocessing](https://lichao312214129.github.io/HABIT/user_guide/app_image_preprocessing.html)
- Habitat Analysis: See [Documentation - Habitat Analysis](https://lichao312214129.github.io/HABIT/user_guide/habit_analysis.html)
- Feature Extraction: See [Documentation - Feature Extraction](https://lichao312214129.github.io/HABIT/user_guide/feature_extraction.html)
- Machine Learning: See [Documentation - Machine Learning](https://lichao312214129.github.io/HABIT/user_guide/machine_learning.html)

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone <repository_url>
cd habit_project

# 2. Create and activate Conda environment
conda create -n habit python=3.8
conda activate habit

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install HABIT package in editable mode
pip install -e .
```

## 📖 Full Documentation

For detailed usage guides and API documentation, please see:

- **Sphinx Documentation System** (Recommended): Located in `docs/` directory
  - Online docs: [https://lichao312214129.github.io/HABIT](https://lichao312214129.github.io/HABIT)
  - Local build: `cd docs && make html`

### Documentation Contents

- **Getting Started**: Installation, configuration, quick testing
- **User Guide**: Image preprocessing, habitat analysis, feature extraction, machine learning
- **Configuration Reference**: Complete configuration file documentation
- **API Reference**: Detailed documentation of modules and classes
- **Development Guide**: Architecture design, contribution guide

## 🚀 Key Features

| Category | Feature | Description |
| :--- | :--- | :--- |
| 🖼️ **Image Processing** | **Preprocessing Pipeline** | Provides DICOM conversion, resampling, registration, standardization, and N4 bias field correction |
| 🧬 **Habitat Analysis** | **Clustering Strategies** | Supports one-step, two-step, and direct pooling clustering strategies |
| 🔬 **Feature Extraction** | **Advanced Feature Set** | Extracts traditional radiomics, multi-scale spatial interaction (MSI), and intra-tumor heterogeneity (ITH) features |
| 🤖 **Machine Learning** | **Complete Workflow** | Includes data splitting, feature selection, model training, and evaluation |
| 📊 **Validation & Tools** | **Reproducibility Analysis** | Includes test-retest and intraclass correlation coefficient (ICC) analysis tools |

## 🤝 Contributing

Contributions of all forms are welcome! Please refer to the contribution guide or open an Issue to discuss your ideas.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Citation

If you use HABIT in your research, please consider citing:
> [Citation information to be added]

## 🙋‍♀️ Support

If you encounter any issues or have suggestions for improvement, please:
1. Read the full documentation
2. Submit an [Issue](https://github.com/lichao312214129/HABIT/issues) on GitHub