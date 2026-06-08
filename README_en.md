# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**Tumor habitat analysis and intratumoral heterogeneity quantification** for clinical and radiomics research. Workflows are driven by YAML configs: preprocessing, habitat segmentation, feature extraction, and optional machine learning.

**Language / 语言**：[English](README_en.md) | [简体中文](README.md)

---

## Documentation (read details here)

**Online docs (recommended)**：[https://lichao312214129.github.io/HABIT](https://lichao312214129.github.io/HABIT)

*(Primary language: Chinese; English README is an index only.)*

Local build: `cd docs && make html`, then open `docs/build/html/index.html`.

### Suggested learning path

| Step | Topic | Link |
|------|--------|------|
| 1 | Install HABIT (Windows: portable pack in [Installation](https://lichao312214129.github.io/HABIT/getting_started/installation_zh.html); source build: same page, section 2) | [Installation](https://lichao312214129.github.io/HABIT/getting_started/installation_zh.html) |
| 2 | Demo data & 5-step workflow | [Quickstart / Demo](https://lichao312214129.github.io/HABIT/getting_started/quickstart_zh.html) |
| 3 | Per-step commands | [User guide index](https://lichao312214129.github.io/HABIT/user_guide/index_zh.html) |
| 4 | YAML parameters | [Configuration reference](https://lichao312214129.github.io/HABIT/configuration_zh.html) |

### Workflow chapters

| Step | Link |
|------|------|
| Data layout | [Data structure](https://lichao312214129.github.io/HABIT/data_structure_zh.html) |
| Preprocessing | [Image preprocessing](https://lichao312214129.github.io/HABIT/user_guide/image_preprocessing_zh.html) |
| Habitat segmentation | [Habitat segmentation](https://lichao312214129.github.io/HABIT/user_guide/habitat_segmentation_zh.html) |
| Feature extraction | [Habitat features](https://lichao312214129.github.io/HABIT/user_guide/habitat_feature_extraction_zh.html) |
| Machine learning | [ML modeling](https://lichao312214129.github.io/HABIT/user_guide/machine_learning_modeling_zh.html) |
| Model comparison | [Model comparison](https://lichao312214129.github.io/HABIT/user_guide/model_comparison_zh.html) |

### Tools & more

| Topic | Link |
|--------|------|
| CLI overview | [CLI reference](https://lichao312214129.github.io/HABIT/cli_zh.html) |
| DICOM metadata | [habit dicom-info](https://lichao312214129.github.io/HABIT/app_dicom_info_zh.html) |
| Contributing | [Contributing](https://lichao312214129.github.io/HABIT/development/contributing.html) |

---

## Bundled config templates

After cloning or unpacking the repo, use the **`config/`** folder at the **project root** (sibling to the `habit/` Python package, not inside the import package). It contains reference YAMLs for preprocessing, habitat, ML, radiomics, etc. Start with [`config/README_CONFIG.md`](config/README_CONFIG.md) for a scenario index; copy a template and edit paths in `#%%====` blocks. Field reference: [Configuration (ZH)](https://lichao312214129.github.io/HABIT/configuration_zh.html).

---

## Source & demo data

- **Windows portable pack (recommended)**：[Installation — section 1 (ZH)](https://lichao312214129.github.io/HABIT/getting_started/installation_zh.html)
  - **CPU** (smaller download): [Baidu Netdisk `habit_portable.tar.gz`](https://pan.baidu.com/s/1zTbYhDm3VnhHP-cMo5bN0A?pwd=8yks) — code **8yks**
  - **GPU full pack** (optional, ~3 GB): [Baidu Netdisk `HABIT-win-py310-gpu-v0.1.0.tar.gz`](https://pan.baidu.com/s/1xaMy69z-2dZH4nFEwhd4tg?pwd=fxnh) — code **fxnh**
  - Extract: empty folder → portable pack → `setup_habit.bat`
  - **Also from Baidu Netdisk** (into same pack root): `config.zip`, `demo_data.rar`; optional `tests.zip` — see [Installation (ZH)](https://lichao312214129.github.io/HABIT/getting_started/installation_zh.html)
  - CPU + NVIDIA: wheel + `install_gpu_torch.bat`
- **Resources (separate from portable pack)**: `config.zip`, `demo_data.rar`, `tests.zip` — [Installation](https://lichao312214129.github.io/HABIT/getting_started/installation_zh.html) · [Quickstart](https://lichao312214129.github.io/HABIT/getting_started/quickstart_zh.html)
- **Source**：[GitHub](https://github.com/lichao312214129/HABIT) · [Download ZIP](https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip) — see [Installation](https://lichao312214129.github.io/HABIT/getting_started/installation_zh.html)
- **Demo bundle**：see [Quickstart](https://lichao312214129.github.io/HABIT/getting_started/quickstart_zh.html)

---

## Support & license

- **Issues**：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- **License**：[HABIT Software License](LICENSE) (non-commercial use with attribution; commercial use requires prior written consent)

**Core developers**：Li Chao, Dong Mengshi
