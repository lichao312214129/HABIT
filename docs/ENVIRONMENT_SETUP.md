# HABIT 环境设置指南

## 问题说明

如果您在安装 HABIT 时看到依赖冲突警告，这通常是因为：
1. 当前环境中已安装了其他项目的包（如 `clarifai`、`tensorflow`、`dcapy` 等）
2. 这些包对某些依赖有严格的版本要求，与 HABIT 的依赖不完全兼容

**重要提示**：这些警告通常不会影响 HABIT 的核心功能，因为 HABIT 的核心依赖已经成功安装。

## 解决方案

### 方案 1：创建独立的 Conda 环境（强烈推荐）⭐

这是最干净、最安全的解决方案，可以避免所有依赖冲突。

#### Windows 用户：

```powershell
# 1. 创建新的 conda 环境（Python 3.10，因为 AutoGluon 需要）
conda create -n habit python=3.10 -y

# 2. 激活环境
conda activate habit

# 3. 进入项目目录
cd F:\work\habit_project

# 4. 安装 HABIT 依赖
pip install -r requirements.txt

# 5. 安装 HABIT 包（开发模式）
pip install -e .
```

#### macOS/Linux 用户：

```bash
# 1. 创建新的 conda 环境
conda create -n habit python=3.10 -y

# 2. 激活环境
conda activate habit

# 3. 进入项目目录
cd /path/to/habit_project

# 4. 安装 HABIT 依赖
pip install -r requirements.txt

# 5. 安装 HABIT 包（开发模式）
pip install -e .
```

### 方案 2：使用当前环境（如果必须）

如果您必须使用当前环境，可以：

1. **忽略警告**：如果 HABIT 功能正常，这些警告可以忽略
2. **测试功能**：运行以下命令验证 HABIT 是否正常工作：

```bash
# 测试 CLI 是否正常
habit -h

# 测试导入是否正常
python -c "from habit.cli import cli; print('HABIT imported successfully')"
```

如果上述命令都能正常执行，说明 HABIT 可以正常使用，依赖冲突不影响核心功能。

### 方案 3：修复特定冲突（高级用户）

如果某个冲突确实影响了功能，可以尝试：

```bash
# 例如，如果 pandas 版本冲突影响功能
pip install pandas==2.0.3 --force-reinstall

# 或者，如果 pydantic 版本冲突
pip install pydantic==2.10.6 --force-reinstall
```

**注意**：强制重新安装可能会影响其他已安装的包，请谨慎操作。

## 验证安装

安装完成后，运行以下命令验证：

```bash
# 1. 检查 CLI 是否可用
habit -h

# 2. 检查核心模块是否可以导入
python -c "from habit.core.habitat_analysis import HabitatAnalysis; print('✓ Core modules OK')"

# 3. 检查机器学习模块（如果使用）
python -c "from habit.core.machine_learning import MachineLearningWorkflow; print('✓ ML modules OK')"
```

## 常见问题

### Q1: 为什么推荐 Python 3.10？

A: HABIT 的机器学习模块使用了 AutoGluon，它需要 Python 3.10+。如果您不需要使用 AutoGluon，可以使用 Python 3.8+。

### Q2: 如果我只使用预处理和生境分析，不需要机器学习功能，怎么办？

A: 您可以：
1. 使用 Python 3.8 环境
2. 不安装 `torch`、`xgboost`、`shap` 等机器学习相关包
3. 只安装核心依赖：

```bash
pip install SimpleITK==2.2.1 antspyx==0.4.2 opencv-python numpy matplotlib scipy pandas pyyaml click pydicom
```

### Q3: 依赖冲突警告会影响功能吗？

A: 大多数情况下不会。这些警告只是 pip 提醒您某些已安装的包可能有版本不兼容，但如果 HABIT 的核心依赖已经正确安装，功能通常不受影响。

### Q4: 如何完全清理环境重新安装？

A: 

```bash
# 1. 删除旧环境（如果使用独立环境）
conda deactivate
conda env remove -n habit

# 2. 重新创建环境
conda create -n habit python=3.10 -y
conda activate habit

# 3. 重新安装
cd F:\work\habit_project
pip install -r requirements.txt
pip install -e .
```

## 联系支持

如果遇到无法解决的问题，请：
1. 在 GitHub 上提交 Issue：https://github.com/lichao312214129/HABIT/issues
2. 发送邮件至：lichao19870617@163.com

请在问题描述中附上：
- Python 版本：`python --version`
- 操作系统：Windows/macOS/Linux
- 完整的错误信息
- 您使用的安装方法
