# HABIT 安装指南

## 环境要求
- Python 3.8.16
- Conda (Anaconda或Miniconda)
- pip (Python包管理器)

## 安装步骤

1. 克隆仓库
```bash
git clone [repository_url]
cd HABIT
```

2. 创建并激活Conda虚拟环境
```bash
# 创建名为habit的conda环境，指定Python版本为3.8.16
conda create -n habit python=3.8.16

# 激活环境
# Windows
conda activate habit

# Linux/Mac
conda activate habit
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

4. 安装HABIT包
```bash
pip install -e .
```

## 验证安装

安装完成后，您可以通过以下命令验证安装是否成功：
```bash
python -c "import habit"
```

如果没有报错，说明安装成功。

## 注意事项
- 确保您的Python版本为3.8.16
- 如果安装过程中遇到依赖包版本冲突，请尝试先升级pip：
  ```bash
  pip install --upgrade pip
  ```
- 如果遇到权限问题，可能需要使用管理员权限运行命令
- 使用conda环境时，请确保已经安装了Anaconda或Miniconda
- 可以通过以下命令查看当前激活的conda环境：
  ```bash
  conda info --envs
  ```
