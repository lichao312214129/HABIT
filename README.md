# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**肿瘤生境（Habitat）分析与瘤内异质性评估工具箱** — 面向临床与影像组学研究，通过 YAML 配置驱动预处理、生境分割、特征提取与机器学习。

**语言 / Language**：[简体中文](README.md) | [English](README_en.md)

---

## 文档（主要内容请在此阅读）

**在线文档（推荐，英文）**：[https://lichao312214129.github.io/HABIT](https://lichao312214129.github.io/HABIT)

本地构建：进入 `docs/` 目录执行 `make html`，在 `docs/build/html/index.html` 打开。

### 推荐学习路径

| 顺序 | 说明 | 链接 |
|------|------|------|
| 1 | **安装** HABIT（Windows 推荐便携包） | [安装](https://lichao312214129.github.io/HABIT/tutorial/installation.html) |
| 2 | 跑通 Demo | [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) |
| 3 | 图形界面（可选） | [Web GUI](https://lichao312214129.github.io/HABIT/gui/index.html) |
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

## 源码与演示数据

- **Windows 便携包（推荐）**：[安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)
  - **GPU 整包**（可选，约 3 GB）：百度网盘 [`HABIT-win-py310-gpu-v0.1.0.tar.gz`](https://pan.baidu.com/s/1bzh3DvNmiL4m-Wdw7K0Tcg?pwd=8wzx) ，提取码 **8wzx**
  - **CPU 版**（优先，体积小）：百度网盘 [`HABIT-win-py310-cpu-v0.1.0.tar.gz`](https://pan.baidu.com/s/1dG4ibQONxvMOFZm1mOKpFw?pwd=ycva) ，提取码 **ycva**
  - 解压：新建空文件夹 → 便携包解压到当前目录 → `setup_habit.bat`
  - 跑 Demo 时从网盘下载 `config.rar`、`demo_data.rar` — 见 [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)
  - **GPU 整包已内置 GPU 加速，无需额外安装**；仅 CPU 版 + NVIDIA 显卡时才可选 wheel + `install_gpu_torch.bat`（见安装指南）
- **源码**：[GitHub 仓库](https://github.com/lichao312214129/HABIT) · [下载 ZIP](https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip)（安装见 [安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)）
- **配置 / 演示数据 / 测试**：`config.rar`、`demo_data.rar`（Demo 教程）、`tests.zip`（可选）— 见 [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)

---

## 支持与引用

- **问题反馈**：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- **引用**：见文档 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- **许可**：[HABIT Software License](LICENSE)（非商业免费使用须署名申明；商业使用须事先书面授权）

**开发团队**：HABIT 开发团队（详见 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)）
