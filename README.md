# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

**肿瘤生境（Habitat）分析工具箱** — 从影像与 ROI 得到生境标签图，再对生境做体积、异质性、图网络与放射组学等定量。产品聚焦生境分析。

**语言 / Language**：[简体中文](https://github.com/lichao312214129/HABIT/blob/main/README.md) | [English](https://github.com/lichao312214129/HABIT/blob/main/README_en.md)

---

## 文档（请在此学习，不要依赖本 README 当教程）

**在线文档（推荐）**：[https://lichao312214129.github.io/HABIT/](https://lichao312214129.github.io/HABIT/)

| 入口 | 链接 |
|------|------|
| 安装 | [Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html) |
| 第一张生境图（Python） | [Quickstart (Python)](https://lichao312214129.github.io/HABIT/auto_quickstart/plot_quickstart_python.html) |
| 第一张生境图（CLI / YAML） | [Quickstart (CLI)](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) |
| **Habitat Guide**（按研究阶段，一页一事） | [Habitat Guide](https://lichao312214129.github.io/HABIT/auto_examples/index.html) |
| API / Spec / 特征公式 | [Reference](https://lichao312214129.github.io/HABIT/api/index.html) |

本地构建文档：见 [`docs/README.md`](docs/README.md)（须用 py310 的 Sphinx，勿用 PATH 上的 base `sphinx-build`）。

---

## 安装

Python **3.10–3.14**。完整说明与可选依赖见[安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)。

```bash
conda create -n habit python=3.10 -y
conda activate habit
pip install -U pip
pip install habitat-analysis -i https://pypi.org/simple
habit --version
# 代码里：import habit
```

常用 extras 示例（按需安装；缺什么会报错并给出 `pip install` 命令）：

```bash
pip install "habitat-analysis[tables,viz]"
```

- **源码**：[GitHub](https://github.com/lichao312214129/HABIT)（开发：`pip install -e .`）
- **演示数据**：见文档 [Quickstart](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) 与 `habit.datasets.fetch_demo`（Guide 脚本默认走缓存下载）

仓库根目录的 [`config/`](config/) 提供可参考的生境 YAML 模板；场景索引见 [`config/README_CONFIG.md`](config/README_CONFIG.md)。

---

## 支持与引用

- **问题反馈**：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- **引用**：见 [CITATION.cff](CITATION.cff) 与文档 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- **许可**：[Apache License 2.0](LICENSE)。可自由用于学术与商业用途；再分发时保留版权与许可声明，并附带 [NOTICE](NOTICE)。用于科研工作时，作者恳请（但不作为许可条件）引用 HABIT

**开发团队**：HABIT 开发团队（详见 [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)）
