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

## Install & demo data

Two supported install methods (full steps: [Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html)). Python **3.10–3.14** (numpy 1.26 and 2.x).

- **(A) pip** (Miniconda / venv / etc.)
  ```bash
  pip install habitat-analysis
  habit --version
  # import name remains: import habit
  ```
- **(B) from Git source**
  ```bash
  git clone https://github.com/lichao312214129/HABIT.git
  cd HABIT
  pip install .
  # editable: pip install -e .
  ```

PyRadiomics is **not** a default dependency. When you need radiomics features:
```bash
pip install "habitat-analysis[radiomics]"
python -m habit.install_radiomics
python -c "import radiomics; print(radiomics.__version__)"
```
On Windows this installs the matching prebuilt wheel from the HABIT GitHub Release (avoids the broken PyPI sdist). On macOS / Linux it installs `pyradiomics` from PyPI. Other extras: `pip install "habitat-analysis[ml,analysis,registration]"`.

- **Source**: [GitHub](https://github.com/lichao312214129/HABIT)
- **Demo data**: [Quickstart](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)

---

## Support & citation

- **Issues**: [GitHub Issues](https://github.com/lichao312214129/HABIT/issues)
- **Citation**: see [CITATION.cff](https://github.com/lichao312214129/HABIT/blob/main/CITATION.cff)
- **License**: [Apache License 2.0](https://github.com/lichao312214129/HABIT/blob/main/LICENSE). Free for academic and commercial use; the only obligation is to retain the copyright and license notices and to ship [NOTICE](https://github.com/lichao312214129/HABIT/blob/main/NOTICE) with redistributions. When HABIT supports scientific work, the authors request -- but do not require as a license condition -- that you cite it
