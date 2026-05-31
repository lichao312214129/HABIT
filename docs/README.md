# HABIT 文档

官方 Sphinx 文档。YAML 字段权威参考：**`docs/source/configuration_zh.rst`**（构建后为「配置参考」）。配置模板索引见 **`config/README_CONFIG.md`**。

## 构建 HTML（与 HABIT 相同的 py310 环境）

```bash
conda activate habit
cd docs
pip install -r requirements.txt
make html
```

Windows 可用 ``make.bat html``。预览：``make livehtml``。本地打开 ``docs/_build/html/index.html``。

## 文档结构

- **getting_started/** — 安装、快速开始
- **user_guide/** — 工作流（字段见 configuration_zh）
- **configuration_zh.rst** — YAML 参数全集
- **api/**、**tutorials/**、**development/**

欢迎贡献文档，见开发指南中的贡献说明。
