# 安装指南

本文档提供了项目安装和开发设置的详细步骤。

## 系统要求

- Python 3.8 或更高版本
- 足够的磁盘空间（取决于您的数据集大小）
- 适用于多进程处理的多核 CPU（推荐）

## 依赖包

主要依赖包包括：

- numpy
- pandas
- matplotlib
- scikit-learn
- SimpleITK
- tqdm

## 安装步骤

### 1. 克隆代码库

```bash
git clone https://github.com/yourusername/habitat-clustering.git
cd habitat-clustering
```

### 2. 创建虚拟环境（推荐）

#### 使用 conda：

```bash
conda create -n habitat-env python=3.8
conda activate habitat-env
```

#### 或使用 venv：

```bash
python -m venv habitat-env
# Linux/Mac
source habitat-env/bin/activate
# Windows
habitat-env\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 开发模式安装

```bash
pip install -e .
```

## 验证安装

运行以下命令验证安装是否成功：

```bash
python -c "import habitat_clustering; print(habitat_clustering.__file__)"
```

如果显示了 habitat_clustering 模块的路径，说明安装成功。

## 常见问题

### ImportError: No module named 'habitat_clustering'

这通常是因为模块没有正确安装。请确保：

1. 您已经运行了 `pip install -e .` 命令
2. 您的 Python 环境变量 PYTHONPATH 设置正确
3. 您在正确的虚拟环境中

解决方法：

```bash
# 确认当前工作目录是项目根目录
cd /path/to/habitat-clustering

# 重新安装
pip install -e .
```

### SimpleITK 安装问题

如果 SimpleITK 安装出错，可以尝试：

```bash
conda install -c simpleitk simpleitk
```

### 针对 Windows 用户的说明

Windows 用户可能需要额外安装 Microsoft Visual C++ Build Tools：

1. 访问 https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 下载并安装 Build Tools（确保选择 "C++ build tools"）
3. 然后再次尝试安装依赖

## 开发工作流程

1. 创建新的分支进行开发
2. 实现功能或修复问题
3. 运行测试确保代码正常工作
4. 提交 Pull Request

## 代码风格指南

请遵循以下代码风格规范：

- 遵循 PEP 8 规范
- 使用有意义的变量名和函数名
- 为所有函数和类添加文档字符串
- 避免不必要的复杂性和嵌套

## 构建文档

项目文档使用 Sphinx 构建。要生成文档，请执行：

```bash
cd docs
make html
```

生成的文档将在 `docs/_build/html` 目录中。 