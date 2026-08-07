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
| 3 | 图形界面（开发中，暂不建议日常使用） | [Web GUI](https://lichao312214129.github.io/HABIT/gui/index.html) |
| 4 | 用自己的数据逐步操作 | [操作指南](https://lichao312214129.github.io/HABIT/how_to/index.html) |
| 5 | 修改 YAML 参数 | [配置参考](https://lichao312214129.github.io/HABIT/configuration/index.html) |
| 6 | 查特征公式（写论文） | [特征参考](https://lichao312214129.github.io/HABIT/reference/features/index.html) |

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

仅支持两种安装方式（详见[安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)）。需要 Python **3.10–3.14**（numpy 1.26 与 2.x 均支持）。

- **(A) pip 安装**（Miniconda / venv 等）
  ```bash
  pip install habitat-analysis
  habit --version
  # 代码中仍然：import habit
  ```
- **(B) 从 Git 源码安装**
  ```bash
  git clone https://github.com/lichao312214129/HABIT.git
  cd HABIT
  pip install .
  # 开发常用：pip install -e .
  ```

默认安装只包含生境内核跑不起来就不行的 11 个包（numpy / scipy / pandas / scikit-learn / SimpleITK / pydantic / PyYAML / click / tqdm / joblib / kneed）：实测 CPython 3.10 / Linux 下 20 个 wheel、129 MB 下载、635 MB 安装体积。生境分析全流程（体素特征 → 超体素 → 队列生境拟合 → 生境指派 → 生境特征 → CSV 结果表）不需要任何 extra。

其余能力按需装 extra，装漏了不会静默降级：HABIT 会抛 `OptionalDependencyError`，消息里直接给出可复制的 `pip install` 命令。

| Extra | 用途 |
| --- | --- |
| `viz` | 全部出图（`habit.viz`、ML 报告图、聚类图、KM 曲线） |
| `tables` | 读写 `.parquet`（`habitats_results_format` 的默认值）与 `.xlsx` |
| `dicom` | `habit dicom-info` / `habit sort-dicom`；NIfTI / NRRD 输入不需要 |
| `slic` | SLIC 超体素后端；默认的 `kmeans` / `gmm` 后端不需要 |
| `ml` | XGBoost、SMOTE、mRMR / VIF / 逐步回归特征筛选（含 `viz`、`tables`） |
| `analysis` | SHAP、Plotly、ICC、生存分析（含 `viz`、`tables`） |
| `registration` | 预处理里的 ANTs 配准后端 |
| `automl` | AutoGluon Tabular |
| `torch` | TorchRadiomics / GPU 纹理后端 |
| `gui` | Web GUI 服务端（预览） |
| `all` | 除 `torch` 与 PyRadiomics 外的全部可选能力 |
| `full` | 1.0.x 用户的迁移别名，见下 |

```bash
pip install "habitat-analysis[ml,analysis]"
```

**从 1.0.x 升级**：1.1.0 把 `matplotlib`、`seaborn`、`scikit-image`、`pydicom`、`pyarrow`、`openpyxl` 从必装下放到 extras（`chardet` 直接移除，已无处使用），默认安装从 212 MB 下载 / 931 MB 安装 / 43 个包降到 129 MB / 635 MB / 23 个包。Python API 的公开符号与签名**没有任何变化**，变的只是 `pip install habitat-analysis` 会装到什么。一条命令恢复旧行为：

```bash
pip install -U "habitat-analysis[full]"
```

注意 `habitats_results_format` **仍然默认 parquet**（不改默认值，避免输出文件名从 `habitats.parquet` 悄悄变成 `habitats.csv`）。缺 pyarrow 时会报错并同时给出两条出路：装 `[tables]`，或在 YAML 里设 `habitats_results_format: csv`。

PyRadiomics **不是**默认依赖，也**不会**由 HABIT extras 拉取——需要组学特征时请**单独安装**：

- **Windows**：从 [Release v1.0.2](https://github.com/lichao312214129/HABIT/releases/tag/v1.0.2) 安装对应 CPython 的预编译 wheel（勿用裸 `pip install pyradiomics`，PyPI sdist 会编译失败），例如 Python 3.10：
  ```bash
  pip install https://github.com/lichao312214129/HABIT/releases/download/v1.0.2/pyradiomics-3.1.0-cp310-cp310-win_amd64.whl
  ```
- **macOS / Linux**：`pip install "pyradiomics>=3.0.1,<3.2"`，或 `conda install -c conda-forge pyradiomics`

完整 wheel 对照表与 extras 矩阵见[安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)。

- **源码**：[GitHub 仓库](https://github.com/lichao312214129/HABIT)
- **演示数据 / 测试**：[`demo_data.rar`](https://pan.baidu.com/s/1K1m8U47wUWV9CCUNahNZuw?pwd=9ws9)（**9ws9**）；可选 [`tests.zip` 打包目录](https://pan.baidu.com/s/1EAcC2s4qIKGp1h08UtbApA?pwd=vv2c)（**vv2c**）。`config/` 已内置于源码 — 见 [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)

---

## 支持与引用

- **问题反馈**：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- **引用**：见 [CITATION.cff](CITATION.cff) 与文档 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- **许可**：[Apache License 2.0](LICENSE)。可自由用于学术与商业用途，唯一义务是保留版权与许可声明，并在再分发时附带 [NOTICE](NOTICE)。用于科研工作时，作者恳请（但不作为许可条件）引用 HABIT

**开发团队**：HABIT 开发团队（详见 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)）
