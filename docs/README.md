# HABIT 文档

这是 HABIT 项目的官方文档。

## YAML 配置参数（权威参考）

所有 CLI 功能的 **YAML 字段说明**（类型、默认值、子块参数、路径解析、`config_hash` 等）集中在 Sphinx 源文件：

- **`docs/source/configuration_zh.rst`** → 构建后为 HTML 目录 **「配置参考」**

仓库内配置模板索引见 **`config/README_CONFIG.md`**。用户指南（`docs/source/user_guide/`）侧重操作流程，字段细节以配置参考为准。

## 构建文档

### 安装依赖

```bash
pip install -r requirements.txt
```

### 构建 HTML 文档

```bash
make html
```

或 Windows:

```cmd
make.bat html
```

### 实时预览

```bash
make livehtml
```

## 文档结构

- **getting_started/**: 快速开始指南
- **user_guide/**: 用户指南（工作流与示例；YAML 字段见 **configuration_zh**）
- **configuration_zh.rst**: **配置参考** — 各功能 YAML 参数全集
- **api/**: API 文档
- **algorithms/**: 算法文档
- **tutorials/**: 教程
- **development/**: 开发者指南

本地构建后打开 `docs/_build/html/configuration_zh.html` 查看配置参考。

## 贡献

欢迎贡献文档！请参考 `开发指南 <source/development/contributing.html>`_。
