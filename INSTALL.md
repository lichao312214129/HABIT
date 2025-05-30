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
- **Python**: 3.8.16 (必需)
- **Conda**: Anaconda 或 Miniconda (推荐)
- **Git**: 用于克隆仓库

## 🚀 快速安装

### 方法一：使用 Conda (推荐)

```bash
# 1. 克隆仓库
git clone <repository_url>
cd habit_project

# 2. 创建并激活Conda虚拟环境
conda create -n habit python=3.8.16
conda activate habit

# 3. 安装依赖包
pip install -r requirements.txt

# 4. 安装HABIT包
pip install -e .
```

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

| 包名 | 版本 | 用途 |
|------|------|------|
| SimpleITK | 2.2.1 | 医学影像处理 |
| antspyx | 0.4.2 | 高级影像配准和处理 |
| numpy | latest | 数值计算 |
| pandas | latest | 数据处理 |
| scikit-learn | latest | 机器学习 |
| pyradiomics | latest | 影像组学特征提取 |
| xgboost | latest | 梯度提升算法 |
| matplotlib | latest | 数据可视化 |
| seaborn | latest | 统计可视化 |

### 可选依赖

```bash
# 深度学习支持 (可选)
pip install torch torchvision

# Jupyter notebook 支持 (可选)
pip install jupyter ipykernel
python -m ipykernel install --user --name habit --display-name "HABIT"

# 开发工具 (可选)
pip install black pytest mypy pylint pre-commit
```

## ✅ 验证安装

### 基本验证
```bash
# 激活环境
conda activate habit

# 验证Python包导入
python -c "import habit; print('HABIT installed successfully!')"

# 检查版本
python -c "import habit; print(f'HABIT version: {habit.__version__}')"
```

### 功能验证
```bash
# 运行示例脚本 (需要配置文件)
python scripts/app_getting_habitat_map.py --help

# 检查配置文件
python -c "from habit.utils.config_utils import load_config; print('Config utilities working!')"
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
创建或修改配置文件 `habit/utils/example_paths_config.yaml`:
```yaml
data_dir: "/path/to/your/data"
out_dir: "/path/to/output"
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

### 依赖版本冲突
如果遇到依赖版本冲突，可以尝试：

```bash
# 1. 清理环境
conda deactivate
conda remove -n habit --all
conda create -n habit python=3.8.16

# 2. 分步安装核心依赖
conda activate habit
pip install numpy pandas matplotlib
pip install SimpleITK==2.2.1
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
conda create -n habit-dev python=3.8.16
conda activate habit-dev

# 3. 安装开发依赖
pip install -r requirements.txt
pip install -e ".[dev]"

# 4. 安装pre-commit hooks
pre-commit install

# 5. 运行测试
pytest tests/
```

## 🐳 Docker 安装 (高级)

如果您熟悉 Docker，可以使用容器化部署：

```dockerfile
# Dockerfile 示例
FROM python:3.8.16-slim

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
6. **提交Issue**: 在项目GitHub页面提交详细的问题报告

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
pip install --upgrade -r requirements.txt
```
