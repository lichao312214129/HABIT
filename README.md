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

## 源码与演示数据

- **Windows 轻量一键安装（推荐）**：[安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)
  - 解压轻量 ZIP 到纯英文、无空格的短路径（例如 `D:\HABIT`）
  - 打开 `launchers/`，双击 `一键安装HABIT.bat`；安装器会在项目内创建锁定的 Python 3.10 环境，不要求预装 Python 或 Conda，也不会修改用户 PATH
  - 安装完成后，在 `launchers/` 中双击 `启动HABIT命令行.bat`
  - 默认环境只包含基础影像、habitat 与常规模型依赖；仅在确实使用对应功能时，在 `launchers/` 中运行 `一键启用HABIT-AutoML.bat` 或 `一键启用HABIT-进阶分析.bat`
  - NVIDIA GPU 为可选增强：先保证 CPU 环境通过自检，再在 `launchers/` 中双击 `一键启用HABIT-GPU.bat`；失败时基础 CPU 环境仍可用
  - 轻量 ZIP 已包含 `config/`；跑 Demo 时仅需从网盘下载 [`demo_data.rar`](https://pan.baidu.com/s/1K1m8U47wUWV9CCUNahNZuw?pwd=9ws9)（提取码 **9ws9**）— 见 [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)
- **源码**：[GitHub 仓库](https://github.com/lichao312214129/HABIT) · [下载 ZIP](https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip)（安装见 [安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)）
- **演示数据 / 测试**：[`demo_data.rar`](https://pan.baidu.com/s/1K1m8U47wUWV9CCUNahNZuw?pwd=9ws9)（**9ws9**）；可选 [`tests.zip` 打包目录](https://pan.baidu.com/s/1EAcC2s4qIKGp1h08UtbApA?pwd=vv2c)（**vv2c**）。`config/` 已内置于源码和轻量 ZIP — 见 [Demo 教程](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)

---

## 支持与引用

- **问题反馈**：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- **引用**：见 [CITATION.cff](CITATION.cff) 与文档 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- **许可**：[Apache License 2.0](LICENSE)。可自由用于学术与商业用途，唯一义务是保留版权与许可声明，并在再分发时附带 [NOTICE](NOTICE)。用于科研工作时，作者恳请（但不作为许可条件）引用 HABIT

**开发团队**：HABIT 开发团队（详见 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)）
