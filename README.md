# HABIT

肿瘤生境影像（habitat imaging）：由医学影像与 ROI 生成生境标签图，再计算体积、瘤内异质性（intratumoral heterogeneity）、图网络与影像组学（radiomics）。

**语言 / Language**：[简体中文](https://github.com/lichao312214129/HABIT/blob/main/README.md) | [English](https://github.com/lichao312214129/HABIT/blob/main/README_en.md)

**文档**：[https://lichao312214129.github.io/HABIT/](https://lichao312214129.github.io/HABIT/)

| 入口 | 链接 |
|------|------|
| 安装 | [Installation](https://lichao312214129.github.io/HABIT/tutorial/installation.html) |
| 第一张生境图（Python） | [Quickstart (Python)](https://lichao312214129.github.io/HABIT/auto_quickstart/plot_quickstart_python.html) |
| 第一张生境图（CLI / YAML） | [Quickstart (CLI)](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html) |
| Habitat Guide | [Habitat Guide](https://lichao312214129.github.io/HABIT/auto_examples/index.html) |
| API | [Reference](https://lichao312214129.github.io/HABIT/api/index.html) |

## 安装

Python **3.10–3.14**。说明见[安装指南](https://lichao312214129.github.io/HABIT/tutorial/installation.html)。

```bash
pip install habitat-analysis
```

代码里 `import habit`。源码安装：`pip install -e .`。演示数据见 [Quickstart](https://lichao312214129.github.io/HABIT/tutorial/quickstart.html)（`habit.datasets.fetch_demo`）。生境 YAML 示例在 [`config/`](config/)。

## 引用与许可

- 问题：[GitHub Issues](https://github.com/lichao312214129/HABIT/issues) · [lichao19870617@163.com](mailto:lichao19870617@163.com)
- 引用：[CITATION.cff](CITATION.cff) · [致谢](https://lichao312214129.github.io/HABIT/acknowledgments.html)
- 许可：[Apache-2.0](LICENSE)。学术与商业使用均可；再分发时保留版权、许可声明与 [NOTICE](NOTICE)。用于科研时请引用 HABIT。
