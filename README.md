# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**肿瘤生境（Habitat）分析与瘤内异质性评估工具箱** — 面向临床与影像组学研究，通过 YAML 配置驱动预处理、生境分割、特征提取与机器学习。

**语言 / Language**：[简体中文](https://github.com/lichao312214129/HABIT/blob/main/README.md) | [English](https://github.com/lichao312214129/HABIT/blob/main/README_en.md)

---

## 文档（主要内容请在此阅读）

**在线文档（推荐，英文）**：[https://lichao312214129.github.io/HABIT](https://lichao312214129.github.io/HABIT)

本地构建：进入 `docs/` 目录执行 `make html`，在 `docs/build/html/index.html` 打开。

### 推荐学习路径

| 顺序 | 说明 | 链接 |
|------|------|------|
| 1 | **安装** HABIT（pip 或 Git 源码） | [安装](https://lichao312214129.github.io/HABIT/tutorial/installation.html) |
| 2 | 跑通 Demo | [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) |
| 3 | 用自己的数据逐步操作 | [操作指南](https://lichao312214129.github.io/HABIT/how_to/index.html) |
| 4 | 修改 YAML 参数 | [配置参考](https://lichao312214129.github.io/HABIT/configuration/index.html) |
| 5 | 查特征公式（写论文） | [特征参考](https://lichao312214129.github.io/HABIT/reference/features/index.html) |

### 按流程查阅

| 步骤 | 文档 |
|------|------|
| 准备数据 | [准备数据](https://lichao312214129.github.io/HABIT/how_to/prepare_data.html) |
| 影像预处理 | [预处理](https://lichao312214129.github.io/HABIT/how_to/preprocess.html) |
| 生境分割 | [生境分割](https://lichao312214129.github.io/HABIT/how_to/segment_habitat.html) |
| 特征提取 | [特征提取](https://lichao312214129.github.io/HABIT/how_to/extract_features.html) |
| 机器学习 | [机器学习](https://lichao312214129.github.io/HABIT/how_to/train_model.html) |
| 模型对比 | [模型对比](https://lichao312214129.github.io/HABIT/how_to/compare_models.html) |
| 遇到问题 | [常见问题](https://lichao312214129.github.io/HABIT/troubleshooting/faq.html) |

### 其它

| 主题 | 文档 |
|------|------|
| 命令索引 | [命令参考](https://lichao312214129.github.io/HABIT/reference/cli.html) |
| 参与开发 | [贡献指南](https://lichao312214129.github.io/HABIT/development/contributing.html) |

---

## 内置配置模板

获取源码后，在**项目根目录**（与 Python 包 `habit/` 同级，而非 `habit/` 包内部）提供 [`config/`](config/) 目录，内含预处理、生境分割、特征提取、机器学习等**可参考的示例 YAML**。建议先阅读 [`config/README_CONFIG.md`](config/README_CONFIG.md) 中的场景索引，再复制对应文件并按 `#%%====` 块修改路径；各字段含义见 [配置参考](https://lichao312214129.github.io/HABIT/configuration/index.html)。

---

## 安装与演示数据

Python **3.10–3.14**。完整说明见[安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)。

```bash
conda create -n habit python=3.10 -y
conda activate habit
pip install -U pip
pip install habitat-analysis -i https://pypi.org/simple
habit --version
# 代码里：import habit
```

可选：`habit view` 需要 napari（见安装指南）。组学特征需**单独**装 PyRadiomics；其它能力按需装 extra（缺什么会报错并给出 `pip install` 命令），例如：

```bash
pip install "habitat-analysis[ml,analysis]"
```

- **源码**：[GitHub 仓库](https://github.com/lichao312214129/HABIT)（开发：`pip install -e .`）
- **演示数据（分两个包）**：
  1. **影像**（生境 / 预处理 / 特征提取）：[`preprocessed.zip`](https://pan.baidu.com/s/1w8r0IUJ8YXVDrkFYCAOQWw?pwd=9bi3)（提取码 **9bi3**）。解压后须得到 `demo_data/preprocessed/images/` 与 `demo_data/preprocessed/masks/`（与 `config/` 同级；**无**嵌套的 `processed_images` 层）。若 zip 顶层是 `preprocessed/`，解压到 `demo_data/`；若顶层直接是 `images/`+`masks/`，放到 `demo_data/preprocessed/` 下。
  2. **表格 ML**（`habit model` / `habit cv`）：[`ml_data.zip`](https://pan.baidu.com/s/1qOmZJ3uDgkDKHpHGVRpcEA?pwd=atnp)（提取码 **atnp**）。解压到项目根，得到 `demo_data/ml_data/`（含 `breast_cancer_dataset.csv` 等）。若 zip 顶层是 `ml_data/`，解压进 `demo_data/`。

  仅跑生境 demo 只需包 1；跑 ML demo 再下包 2。见 [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)

---

## 支持与引用

- **问题反馈**：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- **引用**：见 [CITATION.cff](CITATION.cff) 与文档 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- **许可**：[Apache License 2.0](LICENSE)。可自由用于学术与商业用途，唯一义务是保留版权与许可声明，并在再分发时附带 [NOTICE](NOTICE)。用于科研工作时，作者恳请（但不作为许可条件）引用 HABIT

**开发团队**：HABIT 开发团队（详见 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)）
