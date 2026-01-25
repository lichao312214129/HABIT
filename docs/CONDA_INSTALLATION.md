# Conda 批量安装包的方法

Conda 提供了多种批量安装包的方法，比逐个安装更高效。

## 方法 1：使用 environment.yml 文件（推荐）⭐

这是 conda 的标准方式，可以一次性创建环境并安装所有依赖。

### 创建 environment.yml 文件

项目根目录已经包含了 `environment.yml` 文件，内容如下：

```yaml
name: habit
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy==1.10.1
  - pandas
  - matplotlib
  - scikit-learn
  - seaborn
  - opencv
  - pyyaml
  - tqdm
  - openpyxl
  - xgboost
  - pytorch
  - pip:
    - click>=8.0
    - SimpleITK==2.2.1
    - antspyx==0.4.2
    - trimesh
    - pyradiomics
    - mrmr_selection
    - pingouin
    - statsmodels
    - shap
    - lifelines
    - pydicom
    - pydantic
    - -e .
```

### 使用方法

```bash
# 1. 从 environment.yml 创建新环境（推荐）
conda env create -f environment.yml

# 2. 激活环境
conda activate habit

# 3. 验证安装
habit -h
```

### 更新现有环境

如果环境已存在，可以更新：

```bash
# 更新环境（添加新包，不删除现有包）
conda env update -f environment.yml --prune
```

---

## 方法 2：使用 conda install 一次安装多个包

可以在一个命令中安装多个包：

```bash
# 基本语法
conda install package1 package2 package3

# 示例：安装多个科学计算包
conda install numpy scipy pandas matplotlib scikit-learn

# 从特定 channel 安装
conda install -c conda-forge opencv pyyaml

# 指定版本
conda install numpy=1.23.5 scipy=1.9.3
```

### 实际示例（HABIT 项目）

```bash
# 激活环境
conda activate habit

# 批量安装 conda 可用的包
conda install -c conda-forge \
  numpy scipy pandas matplotlib scikit-learn \
  seaborn opencv pyyaml tqdm openpyxl xgboost pytorch

# 然后用 pip 安装 conda 中没有的包
pip install click>=8.0 SimpleITK==2.2.1 antspyx==0.4.2 \
  trimesh pyradiomics mrmr_selection pingouin statsmodels \
  shap lifelines pydicom pydantic

# 最后安装 HABIT 包
pip install -e .
```

---

## 方法 3：从 requirements.txt 转换为 conda 安装

### 方法 A：使用 pip 在 conda 环境中安装

```bash
# 激活环境
conda activate habit

# 使用 pip 安装 requirements.txt 中的所有包
pip install -r requirements.txt

# 安装 HABIT 包
pip install -e .
```

### 方法 B：混合使用 conda 和 pip

```bash
# 1. 先用 conda 安装尽可能多的包（conda 管理更好）
conda install -c conda-forge numpy scipy pandas matplotlib scikit-learn

# 2. 然后用 pip 安装剩余的包
pip install -r requirements.txt

# 3. 安装 HABIT
pip install -e .
```

---

## 方法 4：导出和共享环境

### 导出当前环境

```bash
# 导出当前环境到 environment.yml
conda env export > environment.yml

# 只导出明确安装的包（不包括依赖）
conda env export --from-history > environment.yml

# 导出为 requirements.txt（pip 格式）
pip freeze > requirements.txt
```

### 从导出的文件创建环境

```bash
# 从 environment.yml 创建
conda env create -f environment.yml

# 从 requirements.txt 安装（在已激活的环境中）
pip install -r requirements.txt
```

---

## 方法 5：使用 conda 的 --file 参数

创建一个包含包名的文本文件：

```bash
# 创建 packages.txt
cat > packages.txt << EOF
numpy
scipy
pandas
matplotlib
scikit-learn
EOF

# 批量安装
conda install --file packages.txt
```

---

## 推荐的工作流程（HABIT 项目）

### 方案 A：使用 environment.yml（最推荐）

```bash
# 1. 创建环境并安装所有依赖
conda env create -f environment.yml

# 2. 激活环境
conda activate habit

# 3. 验证
habit -h
```

### 方案 B：手动安装（如果 environment.yml 有问题）

```bash
# 1. 创建环境
conda create -n habit python=3.10 -y
conda activate habit

# 2. 安装 conda 包
conda install -c conda-forge \
  numpy scipy pandas matplotlib scikit-learn \
  seaborn opencv pyyaml tqdm openpyxl xgboost pytorch

# 3. 安装 pip 包
pip install -r requirements.txt

# 4. 安装 HABIT
pip install -e .
```

---

## 常见问题

### Q1: conda 和 pip 混用安全吗？

A: 一般来说是安全的，但建议：
- 优先使用 conda 安装包（如果可用）
- 只在 conda 没有的包时使用 pip
- 避免对同一个包同时使用 conda 和 pip

### Q2: 如何知道包在 conda 中是否可用？

```bash
# 搜索包
conda search package_name

# 搜索特定 channel
conda search -c conda-forge package_name
```

### Q3: 如何更新所有包？

```bash
# 更新 conda 包
conda update --all

# 更新 pip 包
pip list --outdated
pip install --upgrade package_name
```

### Q4: 如何清理 conda 缓存？

```bash
# 清理未使用的包和缓存
conda clean --all
```

---

## 总结

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| environment.yml | 标准化、可重复、完整 | 需要维护文件 | **推荐**：新环境创建 |
| conda install 多包 | 快速、灵活 | 命令较长 | 临时安装几个包 |
| pip install -r | 简单直接 | 不管理 conda 依赖 | 已有环境补充安装 |
| conda + pip 混合 | 充分利用两个工具 | 需要了解包来源 | 复杂项目 |

**对于 HABIT 项目，推荐使用方法 1（environment.yml）**，这是最标准和可重复的方式。
