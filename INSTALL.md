# HABIT 安装指南

## 📋 系统要求

### 操作系统支持
- Windows 10/11
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS 10.14+

### 硬件要求
- **CPU**: 多核处理器 (推荐 8 核以上)
- **内存**: 最小 16GB RAM (推荐 32GB 或更多)
- **存储**: 至少 10GB 可用磁盘空间
- **GPU**: 可选，支持CUDA的NVIDIA GPU (用于深度学习加速)

### 软件依赖
- **Python**: 3.8或更高版本（推荐3.8-3.10）
- **Conda**: Anaconda 或 Miniconda（推荐）
- **Git**: 用于克隆仓库
- **R语言**（可选）：部分特征选择方法（如逐步回归）需要R环境

## 🚀 快速安装

### 方法一：使用 Conda (推荐)

```bash
# 1. 克隆仓库
git clone <repository_url>
cd habit_project

# 2. 创建并激活Conda虚拟环境
conda create -n habit python=3.8
conda activate habit

# 3. 安装依赖包
pip install -r requirements.txt

# 4. 安装HABIT包（开发模式）
pip install -e .
```

**注意**：`pip install -e .` 会以开发模式安装HABIT包，这样您可以直接修改代码而无需重新安装。

### 方法二：使用 pip 和 venv

```bash
# 1. 克隆仓库
git clone <repository_url>
cd habit_project

# 2. 创建虚拟环境
python -m venv habit_env

# 3. 激活虚拟环境
# Windows
habit_env\Scripts\activate
# Linux/macOS
source habit_env/bin/activate

# 4. 升级pip
pip install --upgrade pip

# 5. 安装依赖包
pip install -r requirements.txt

# 6. 安装HABIT包
pip install -e .
```

## 📦 详细依赖说明

### 核心依赖包

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| SimpleITK | 2.2.1 | 医学影像处理和格式转换 |
| antspyx | 0.4.2 | 高级影像配准和处理 |
| numpy | - | 数值计算 |
| pandas | - | 数据处理和分析 |
| scikit-learn | - | 机器学习算法 |
| pyradiomics | - | 影像组学特征提取 |
| xgboost | - | 梯度提升算法 |
| matplotlib | - | 数据可视化 |
| seaborn | - | 统计可视化 |
| scipy | - | 科学计算 |
| statsmodels | - | 统计模型 |
| PyYAML | - | YAML配置文件解析 |
| tqdm | - | 进度条显示 |
| openpyxl | - | Excel文件处理 |
| mrmr_selection | - | mRMR特征选择 |
| pingouin | - | 统计分析 |
| shap | - | 模型解释 |
| lifelines | - | 生存分析 |
| opencv-python | - | 图像处理 |
| trimesh | - | 网格处理 |
| torch | - | 深度学习框架 |

### 可选依赖

```bash
# AutoGluon自动机器学习 (可选，用于高级建模)
pip install autogluon

# Jupyter notebook 支持 (可选)
pip install jupyter ipykernel
python -m ipykernel install --user --name habit --display-name "HABIT"

# R语言接口 (可选，用于某些特征选择方法)
pip install rpy2

# 开发工具 (可选)
pip install black pytest mypy pylint pre-commit
```

**注意**：
- AutoGluon较大且安装时间较长，仅在需要使用AutoGluon模型时安装
- R语言接口（rpy2）需要先安装R语言环境
- torch已包含在requirements.txt中，如需GPU支持请根据CUDA版本安装对应版本

## ✅ 验证安装

### 基本验证
```bash
# 激活环境
conda activate habit

# 验证Python包导入
python -c "import habit; print('HABIT installed successfully!')"

# 检查核心模块
python -c "from habit.core.habitat_analysis import HabitatAnalysis; print('Core modules OK!')"
python -c "from habit.core.machine_learning.machine_learning import Modeling; print('ML modules OK!')"
```

### 功能验证
```bash
# 查看各应用脚本帮助信息
python scripts/app_getting_habitat_map.py --help
python scripts/app_image_preprocessing.py --help
python scripts/app_of_machine_learning.py --help

# 检查配置文件加载
python -c "from habit.utils.io_utils import load_config; config = load_config('./config/config_getting_habitat.yaml'); print('Config file loaded successfully!')"
```

## 🔧 环境配置

### 设置环境变量 (可选)
```bash
# Linux/macOS
export HABIT_DATA_DIR="/path/to/your/data"
export HABIT_OUTPUT_DIR="/path/to/output"

# Windows (PowerShell)
$env:HABIT_DATA_DIR="C:\path\to\your\data"
$env:HABIT_OUTPUT_DIR="C:\path\to\output"
```

### 配置文件设置
修改`config`文件夹下的相应配置文件：

```yaml
# 示例：config/config_getting_habitat.yaml
data_dir: "/path/to/your/data"
out_dir: "/path/to/output"
processes: 4  # 根据您的CPU核心数调整
# 其他配置项...
```

## 🔍 故障排除

### 常见问题

#### 1. SimpleITK 安装失败
```bash
# 解决方案：使用conda安装
conda activate habit
conda install -c conda-forge simpleitk=2.2.1
```

#### 2. antspyx 安装失败
```bash
# 解决方案：确保编译工具可用
# Windows: 安装 Visual Studio Build Tools
# Linux: sudo apt-get install build-essential
# macOS: xcode-select --install

# 或使用预编译版本
conda install -c conda-forge antspyx
```

#### 3. 内存错误
```bash
# 解决方案：增加虚拟内存或使用较小的数据集
# 在配置文件中设置较小的batch size或并行进程数
```

#### 4. CUDA相关错误 (使用GPU时)
```bash
# 检查CUDA版本兼容性
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 安装对应CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 5. 权限问题
```bash
# Linux/macOS: 使用用户目录安装
pip install --user -r requirements.txt

# Windows: 以管理员身份运行命令提示符
```

#### 6. R语言相关错误（使用逐步特征选择时）
```bash
# 确保已安装R语言
# Windows: 从 https://cran.r-project.org/bin/windows/base/ 下载安装
# Linux: sudo apt-get install r-base
# macOS: brew install r

# 安装rpy2
pip install rpy2

# 在配置文件中指定R路径（Windows示例）
# feature_selection_methods:
#   - method: stepwise
#     params:
#       Rhome: 'C:/Program Files/R/R-4.3.0'  # 根据实际安装路径调整
```

### 依赖版本冲突
如果遇到依赖版本冲突，可以尝试：

```bash
# 1. 清理环境
conda deactivate
conda remove -n habit --all

# 2. 重新创建环境
conda create -n habit python=3.8
conda activate habit

# 3. 分步安装核心依赖
pip install numpy pandas matplotlib
pip install SimpleITK==2.2.1
pip install antspyx==0.4.2
pip install scikit-learn
pip install -r requirements.txt
```

## 📝 开发环境设置

如果您计划为 HABIT 项目贡献代码：

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/yourusername/habit_project.git
cd habit_project

# 2. 创建开发环境
conda create -n habit-dev python=3.8
conda activate habit-dev

# 3. 安装开发依赖
pip install -r requirements.txt

# 4. 以开发模式安装
pip install -e .

# 5. 安装pre-commit hooks（可选）
pip install pre-commit
pre-commit install

# 6. 运行测试（如果有）
pytest tests/
```

## 🐳 Docker 安装 (高级)

如果您熟悉 Docker，可以使用容器化部署：

```dockerfile
# Dockerfile 示例
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "scripts/app_getting_habitat_map.py"]
```

```bash
# 构建和运行
docker build -t habit .
docker run -v /path/to/data:/app/data habit
```

## 📞 获取帮助

如果安装过程中遇到问题：

1. **查看错误日志**: 仔细阅读错误信息
2. **检查系统要求**: 确保满足最低硬件和软件要求
3. **更新系统**: 确保系统包管理器是最新的
4. **清理缓存**: 
   ```bash
   pip cache purge
   conda clean --all
   ```
5. **重新创建环境**: 删除环境后重新安装
6. **查看文档**: 参考doc文件夹下的应用文档
7. **提交Issue**: 在项目GitHub页面提交详细的问题报告

## 🔄 卸载

```bash
# 删除conda环境
conda deactivate
conda remove -n habit --all

# 或删除pip虚拟环境
deactivate
rm -rf habit_env/  # Linux/macOS
rmdir /s habit_env  # Windows
```

---

**注意**: 建议定期更新依赖包以获得最新功能和安全修复：
```bash
conda activate habit
pip install --upgrade -r requirements.txt
```

**下一步**: 安装完成后，请参考`QUICKSTART.md`快速开始使用，或查看`doc`文件夹下的详细文档了解各功能模块的使用方法。
