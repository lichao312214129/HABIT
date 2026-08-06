# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**Tumor habitat analysis and intratumoral heterogeneity quantification** for clinical and radiomics research. Workflows are driven by YAML configs: preprocessing, habitat segmentation, feature extraction, and optional machine learning.

**Language / 语言**：[English](https://github.com/lichao312214129/HABIT/blob/main/README_en.md) | [简体中文](https://github.com/lichao312214129/HABIT/blob/main/README.md)

---

## Documentation

**Online docs**: [https://lichao312214129.github.io/HABIT](https://lichao312214129.github.io/HABIT)

Local build: `cd docs && make html` → `docs/build/html/index.html`

### Suggested learning path

| Step | Topic | Link |
|------|--------|------|
| 1 | Install HABIT | [Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html) |
| 2 | Demo workflow | [Quickstart](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) |
| 3 | Web GUI (under development) | [Web GUI](https://lichao312214129.github.io/HABIT/gui/index.html) |
| 4 | Step-by-step how-to | [How-to index](https://lichao312214129.github.io/HABIT/how_to/index.html) |
| 5 | YAML parameters | [Configuration](https://lichao312214129.github.io/HABIT/configuration/index.html) |

### Workflow chapters

| Step | Link |
|------|------|
| Prepare data | [Prepare data](https://lichao312214129.github.io/HABIT/how_to/prepare_data.html) |
| Preprocessing | [Preprocess](https://lichao312214129.github.io/HABIT/how_to/preprocess.html) |
| Habitat segmentation | [Segment habitat](https://lichao312214129.github.io/HABIT/how_to/segment_habitat.html) |
| Feature extraction | [Extract features](https://lichao312214129.github.io/HABIT/how_to/extract_features.html) |
| Machine learning | [Train model](https://lichao312214129.github.io/HABIT/how_to/train_model.html) |
| Model comparison | [Compare models](https://lichao312214129.github.io/HABIT/how_to/compare_models.html) |
| FAQ | [FAQ](https://lichao312214129.github.io/HABIT/troubleshooting/faq.html) |

### Tools & more

| Topic | Link |
|--------|------|
| CLI overview | [CLI reference](https://lichao312214129.github.io/HABIT/reference/cli.html) |
| Contributing | [Contributing](https://lichao312214129.github.io/HABIT/development/contributing.html) |

---

## Bundled config templates

After cloning or unpacking the repo, use the **`config/`** folder at the **project root** (sibling to the `habit/` Python package). See [`config/README_CONFIG.md`](https://github.com/lichao312214129/HABIT/blob/main/config/README_CONFIG.md) and [Configuration reference](https://lichao312214129.github.io/HABIT/configuration/index.html).

---

## Source & demo data

- **Windows lightweight one-click installer (recommended)**: [Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html)
  - Extract to a short ASCII-only path, open `launchers/`, run `一键安装HABIT.bat`, then use `启动HABIT命令行.bat`
  - The default environment contains imaging, habitat, and standard ML dependencies only; run `一键启用HABIT-AutoML.bat` or `一键启用HABIT-进阶分析.bat` from `launchers/` only when those features are needed
  - Optional NVIDIA acceleration is installed separately with `launchers/一键启用HABIT-GPU.bat`, after the CPU environment passes verification
- **Source**: [GitHub](https://github.com/lichao312214129/HABIT)
- **Demo data**: [Quickstart](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)

---

## Support & citation

- **Issues**: [GitHub Issues](https://github.com/lichao312214129/HABIT/issues)
- **Citation**: see [CITATION.cff](https://github.com/lichao312214129/HABIT/blob/main/CITATION.cff)
- **License**: [Apache License 2.0](https://github.com/lichao312214129/HABIT/blob/main/LICENSE). Free for academic and commercial use; the only obligation is to retain the copyright and license notices and to ship [NOTICE](https://github.com/lichao312214129/HABIT/blob/main/NOTICE) with redistributions. When HABIT supports scientific work, the authors request -- but do not require as a license condition -- that you cite it
