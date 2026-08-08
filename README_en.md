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
| 3 | Step-by-step how-to | [How-to index](https://lichao312214129.github.io/HABIT/how_to/index.html) |
| 4 | YAML parameters | [Configuration](https://lichao312214129.github.io/HABIT/configuration/index.html) |

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

Python **3.10–3.14**. Full steps: [Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html).

```bash
conda create -n habit python=3.10 -y
conda activate habit
pip install -U pip
pip install habitat-analysis -i https://pypi.org/simple
habit --version
# import name: import habit
```

Optional: napari for `habit view` (see Installation). PyRadiomics is separate
when you need radiomics. Other capabilities are extras — missing ones raise
`OptionalDependencyError` with the exact `pip install` command, e.g.:

```bash
pip install "habitat-analysis[ml,analysis]"
```

- **Source**: [GitHub](https://github.com/lichao312214129/HABIT) (dev: `pip install -e .`)
- **Demo data (two packs)**:
  1. **Imaging** (habitat / preprocess / feature extract): [`preprocessed.zip`](https://pan.baidu.com/s/1w8r0IUJ8YXVDrkFYCAOQWw?pwd=9bi3) (code **9bi3**). After extract you must have `demo_data/preprocessed/images/` and `demo_data/preprocessed/masks/` next to `config/` (no nested `processed_images`). If zip top level is `preprocessed/`, extract into `demo_data/`; if `images/`+`masks/`, put under `demo_data/preprocessed/`.
  2. **Tabular ML** (`habit model` / `habit cv`): [`ml_data.zip`](https://pan.baidu.com/s/1qOmZJ3uDgkDKHpHGVRpcEA?pwd=atnp) (code **atnp**). Extract at project root to get `demo_data/ml_data/` (e.g. `breast_cancer_dataset.csv`). If zip top level is `ml_data/`, extract into `demo_data/`.

  Habitat-only demos need pack 1; add pack 2 only for ML demos. See [Quickstart](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)

---

## Support & citation

- **Issues**: [GitHub Issues](https://github.com/lichao312214129/HABIT/issues)
- **Citation**: see [CITATION.cff](https://github.com/lichao312214129/HABIT/blob/main/CITATION.cff)
- **License**: [Apache License 2.0](https://github.com/lichao312214129/HABIT/blob/main/LICENSE). Free for academic and commercial use; the only obligation is to retain the copyright and license notices and to ship [NOTICE](https://github.com/lichao312214129/HABIT/blob/main/NOTICE) with redistributions. When HABIT supports scientific work, the authors request -- but do not require as a license condition -- that you cite it
