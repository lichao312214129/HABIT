# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**Tumor habitat analysis toolkit** — build habitat label maps from images + ROI, then quantify habitats (volume, heterogeneity, graph metrics, radiomics, …). The product focus is **habitat analysis**.

**Language / 语言**：[English](https://github.com/lichao312214129/HABIT/blob/main/README_en.md) | [简体中文](https://github.com/lichao312214129/HABIT/blob/main/README.md)

---

## Documentation (learn here — this README is not a tutorial)

**Online docs**: [https://lichao312214129.github.io/HABIT/](https://lichao312214129.github.io/HABIT/)

| Entry | Link |
|------|------|
| Install | [Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html) |
| First habitat map (Python) | [Quickstart (Python)](https://lichao312214129.github.io/HABIT/auto_quickstart/plot_quickstart_python.html) |
| First habitat map (CLI / YAML) | [Quickstart (CLI)](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) |
| **Habitat Guide** (one scientific task per page) | [Habitat Guide](https://lichao312214129.github.io/HABIT/auto_examples/index.html) |
| API / Spec / feature formulas | [Reference](https://lichao312214129.github.io/HABIT/api/index.html) |

Local Sphinx build: see [`docs/README.md`](docs/README.md) (use the py310 Sphinx; do not use the base-env `sphinx-build` on PATH).

---

## Install

Python **3.10–3.14**. Full steps and optional extras:
[Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html).

```bash
conda create -n habit python=3.10 -y
conda activate habit
pip install -U pip
pip install habitat-analysis -i https://pypi.org/simple
habit --version
# import name: import habit
```

Common extras (install as needed; missing ones raise `OptionalDependencyError`
with the exact `pip install` line):

```bash
pip install "habitat-analysis[tables,viz]"
```

- **Source**: [GitHub](https://github.com/lichao312214129/HABIT) (dev: `pip install -e .`)
- **Demo data**: see [Quickstart](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) and
  `habit.datasets.fetch_demo` (Guide scripts download once into a cache)

Root [`config/`](config/) ships example habitat YAML templates; see
[`config/README_CONFIG.md`](config/README_CONFIG.md).

---

## Support & citation

- **Issues**: [GitHub Issues](https://github.com/lichao312214129/HABIT/issues)
- **Citation**: [CITATION.cff](CITATION.cff) and [Acknowledgments](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- **License**: [Apache License 2.0](LICENSE). Free for academic and commercial use; retain copyright and license notices and ship [NOTICE](NOTICE) with redistributions. When HABIT supports scientific work, the authors request — but do not require as a license condition — that you cite it

**Team**: HABIT development team (see [Acknowledgments](https://lichao312214129.github.io/HABIT/acknowledgments.html))
