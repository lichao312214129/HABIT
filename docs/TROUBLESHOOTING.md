# HABIT 故障排除指南

## 常见错误及解决方案

### 1. ModuleNotFoundError: No module named 'click'

**错误信息：**
```
ModuleNotFoundError: No module named 'click'
```

**原因：** 环境中缺少 `click` 包，这是 HABIT CLI 的核心依赖。

**解决方案：**

#### 方案 A：安装缺失的依赖（快速）

```bash
# 直接安装 click
pip install click>=8.0

# 或者重新安装所有依赖
pip install -r requirements.txt
```

#### 方案 B：使用 setup.py 安装（推荐）

```bash
# 这会自动安装 setup.py 中定义的所有依赖
pip install -e .
```

#### 方案 C：创建新环境（最干净）

```bash
# 1. 创建新环境
conda create -n habit python=3.10 -y
conda activate habit

# 2. 安装 HABIT
cd F:\work\habit_project
pip install -r requirements.txt
pip install -e .
```

---

### 2. AttributeError: module 'scipy.fft' has no attribute 'next_fast_len'

**错误信息：**
```
AttributeError: module 'scipy.fft' has no attribute 'next_fast_len'
```

**原因：** `pywt` (PyWavelets) 与 `scipy` 版本不兼容。

**解决方案：**

这个问题已经在最新版本中修复（使用延迟导入）。如果仍然遇到，可以：

```bash
# 更新相关包
pip install --upgrade pywavelets scipy

# 或者重新安装
pip uninstall pywavelets scipy
pip install pywavelets scipy
```

---

### 3. UserWarning: loaded more than 1 DLL from .libs

**警告信息：**
```
UserWarning: loaded more than 1 DLL from .libs:
```

**原因：** numpy 加载了多个 DLL 文件，这是无害的警告，但会减慢启动速度。

**解决方案：**

这个问题已经在最新版本中修复（在 `habit/cli.py` 中添加了警告过滤器）。如果仍然看到警告，请更新到最新版本：

```bash
git pull origin V_0_1_0
```

---

### 4. 依赖冲突警告

**警告信息：**
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
clarifai 9.10.4 requires opencv-python==4.7.0.68, but you have opencv-python 4.13.0.90...
```

**原因：** 环境中已安装的其他包（如 `clarifai`、`tensorflow` 等）与 HABIT 的依赖版本不完全匹配。

**解决方案：**

这些警告通常**不影响 HABIT 的功能**。如果 HABIT 能正常运行，可以忽略。

如果确实需要解决，建议创建独立环境：

```bash
# 创建新环境
conda create -n habit python=3.10 -y
conda activate habit

# 安装 HABIT
pip install -r requirements.txt
pip install -e .
```

详细说明请参考 [环境设置指南](ENVIRONMENT_SETUP.md)。

---

### 5. 验证安装是否成功

运行以下命令验证：

```bash
# 1. 检查 CLI 是否可用
habit -h

# 2. 检查导入是否正常
python -c "from habit.cli import cli; print('✓ HABIT 导入成功')"

# 3. 检查核心模块
python -c "from habit.core.habitat_analysis import HabitatAnalysis; print('✓ Core modules OK')"
```

如果所有命令都能正常执行，说明安装成功！

---

## 获取帮助

如果以上方案都无法解决问题，请：

1. **在 GitHub 上提交 Issue**：https://github.com/lichao312214129/HABIT/issues
2. **发送邮件**：lichao19870617@163.com

请在问题描述中附上：
- Python 版本：`python --version`
- 操作系统：Windows/macOS/Linux
- 完整的错误信息
- 您使用的安装方法
- 您尝试过的解决方案
