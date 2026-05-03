# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)



**📖 语言 / Language**  
[🇨🇳 简体中文](README.md) | [🇬🇧 English](README_en.md)



> **专为临床医生和研究人员设计的肿瘤"生境"（Habitat）分析与异质性评估工具箱。**
>
> *无需复杂的编程知识，即可开展顶刊级别的影像组学与生境分析研究。*

---

## 🌟 为什么选择 HABIT？

在精准医疗时代，**肿瘤异质性**是影响患者预后和治疗反应的关键因素。传统的影像组学往往提取整个肿瘤的平均特征，掩盖了肿瘤内部的空间异质性。

**HABIT 致力于解决这一痛点：**

1. **自动划分生境**：自动识别肿瘤内部的坏死区、活性肿瘤区、水肿区等具有不同生物学行为的亚区域。
2. **无缝对接临床**：完美支持由 **ITK-SNAP** 或 **3D Slicer** 勾画的 ROI，生成的生境地图可直接拖回软件中叠加显示。
3. **零代码门槛**：通过修改简单的配置文件（类似填表）即可运行，无需编写复杂的代码。
4. **一站式流程**：从 DICOM 预处理 -> 生境聚类 -> 特征提取 -> 机器学习建模 -> 绘制图表，全流程覆盖。

---

## 👨‍⚕️ 临床工作流：从影像到发表

HABIT 将复杂的计算过程封装为简单的四个步骤：

1. **准备数据 (Prepare)**
  - 使用您熟悉的 **ITK-SNAP** 或 **3D Slicer** 勾画肿瘤 ROI，保存为 NIfTI (`.nii.gz`) 格式。
2. **填写配置 (Config)**
  - 修改 `.yaml` 配置文件，指定您的影像文件夹路径（就像填写 Excel 表格一样简单）。
3. **一键运行 (Run)**
  - 在终端运行一行命令，HABIT 自动完成所有计算。
4. **获取结果 (Result)**
  - 获得 **彩色生境地图**（用于直观展示）和 **量化特征表格**（用于统计分析）。

---

## ⚡ 5分钟快速开始

### Step 0: 准备基础环境 (仅需一次)

> 如果您是第一次使用 Python 软件，请先安装 **Miniconda**。
> *如果您已经安装了 Anaconda 或 Miniconda，请跳过此步骤。*

#### 💻 Windows 用户

1. **下载**：👉 [点击下载 Windows 版 Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
2. **安装**：双击下载的 `.exe` 文件，**一路点击 "Next"**，直到安装完成。
  - *建议勾选 "Add Miniconda3 to my PATH environment variable"（如果出现该选项）。*
3. **打开终端**：
  - 点击屏幕左下角的 **"开始"** 菜单。
  - 搜索 **"Anaconda Prompt"**（或 **"Miniconda Prompt"**）。
  - 点击出现的黑色图标。

#### 🍎 macOS (苹果电脑) 用户

1. **下载**：
  - **M1/M2/M3 芯片** (2020年后的新款): 👉 [点击下载 Apple Silicon 版](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)
  - **Intel 芯片** (旧款): 👉 [点击下载 Intel 版](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)
  - *如果不确定，点击屏幕左上角苹果图标 -> "关于本机" 查看。*
2. **安装**：双击下载的 `.pkg` 文件，按照提示一路点击 "继续" 或 "安装"。
3. **打开终端**：
  - 按键盘上的 `Command ⌘` + `空格`。
  - 输入 **"Terminal"** (或 **"终端"**) 并回车。
  - 在终端中输入 `conda init` 并回车，然后**重启终端**。

---

*✅ 验证成功示例：*

> 您应该看到类似下面的黑色窗口，最前面有 `**(base)`** 字样：
>
> ```text
> (base) C:\Users\YourName> _
> ```

### Step 1: 下载与安装 HABIT

**1. 下载工具包**
👉 **[点击下载最新版 HABIT 压缩包](https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip)**

下载后，请将压缩包**解压**到一个您容易找到的文件夹（例如 `D:\HABIT`）。

**2. 进入文件夹**
在刚才打开的 **黑色终端窗口** 中，使用 `cd` 命令进入解压后的文件夹。

*例如，如果您解压在 `D:\HABIT`，请输入：*

```bash
cd D:\HABIT
```

*(如果按下回车后发现目录没变，可能因为您在 C 盘。此时请先输入 `D:` 并回车，再输入上面的命令)*

**3. 一键安装**
复制并粘贴以下命令到终端中运行（网络通畅情况下约需 5-10 分钟）：

```bash
# 1. 创建一个名为 "habit" 的专属环境
# 推荐使用 Python 3.8。如果您计划使用 AutoGluon 进行自动机器学习，请使用 Python 3.10
conda create -n habit python=3.8 -y

# 2. 激活这个环境 (注意观察命令行前缀的变化)
conda activate habit
```

> *✅ 激活成功示例：*
> 命令行前缀会从 `(base)` 变成 `**(habit)*`*：
>
> ```text
> (base) D:\HABIT> conda activate habit
> (habit) D:\HABIT> _
> ```

```bash
# 3. 安装依赖包
pip install -r requirements.txt

# 4. 安装 HABIT 工具本身
pip install -e .
```

> *✅ 安装成功示例：*
> 当屏幕停止滚动，并出现类似下方的文字，说明安装完成：
>
> ```text
> Successfully installed habit-1.0.0
> (habit) D:\HABIT> _
> ```

**❌ 如果安装失败怎么办？**
如果运行 `pip install` 时出现红色报错，请尝试以下步骤：

1. **逐个排查**：打开项目文件夹下的 `requirements.txt`，手动逐行运行 `pip install <包名>`（例如 `pip install SimpleITK`），找出具体是哪个包安装失败。
2. **寻求帮助**：如果无法解决，请在 GitHub 上 [提交 Issue](https://github.com/lichao312214129/HABIT/issues) 或者直接发送邮件联系作者：**[lichao19870617@163.com](mailto:lichao19870617@163.com)**。

### Step 2: 下载演示数据 (Download Demo Data)

演示数据已上传至百度网盘，请先下载并解压：

**📦 下载演示数据**

- **文件**: `demo_data.rar`
- **链接**: [https://pan.baidu.com/s/1bHTLvVMHnfiApArmZf8wrQ](https://pan.baidu.com/s/1bHTLvVMHnfiApArmZf8wrQ)
- **提取码**: `kbmd`

**⚠️ 重要说明**：

- 所有隐私信息已完全去除
- **严禁商业用途，仅供学术研究和Demo演示使用**
- 下载后请将文件解压到 `demo_data` 目录下

解压完成后，按照以下步骤体验完整的研究流程：

---

#### 🔬 **完整研究流程演示**

下面将带您走过一个典型的影像组学研究全流程，从原始影像到最终的预测模型。每一步都会说明其**临床意义**。

---

#### **步骤 1：影像预处理 (Image Preprocessing)**

**📋 临床意义**：

- 就像我们在看片前需要调整窗宽窗位一样，影像预处理确保所有病例的影像在同一"标尺"下进行分析
- 消除不同设备、不同扫描参数带来的系统性差异
- 类似于实验室检查前的标准化操作（如血糖检测前的空腹要求）

```bash
# 运行预处理（包括：DICOM转换、重采样、配准、标准化）
habit preprocess --config demo_data/config_preprocessing.yaml
```

**⏱️ 预计用时**：约 2-5 分钟
**📁 结果位置**：`demo_data/preprocessed/`（按流水线阶段分目录，最终产物在 `processed_images/`）

**💡 重要提示**：

- **关于 `preprocessed` 文件夹**：您从百度网盘下载解压的 demo_data 中已包含此文件夹，里面已有预处理好的影像和 ROI mask 文件，可直接进入下一步
- **预处理输出按"阶段 + 模态"组织**，目录结构为：
  - `preprocessed/<stage>_NN/images/<subject>/<modality>/<modality>.nii.gz`
  - `preprocessed/<stage>_NN/masks/<subject>/<modality>/<modality>.nii.gz`
  - 其中 `<stage>_NN` 为 `dcm2nii_01`、`resample_02`、`registration_03`、`zscore_normalization_04` 等阶段编号；最终聚合产物在 `processed_images/` 下，供下游生境分析直接使用
- **运行预处理的影响**：
  - ✅ 会**覆盖**已有阶段目录中的影像文件（如 `delay2.nii.gz`、`delay3.nii.gz`、`delay5.nii.gz`），生成新的标准化影像
  - ✅ **不会影响** `masks/` 子目录下的 ROI 文件（仅会因配准被重采样到与影像一致的空间）
  - ⚠️ 原因：预处理流程只重写影像与配准后的 mask，不会修改原始的 ROI 勾画语义
- **为什么要保留 mask？** 这些 ROI mask 文件是临床医生用 ITK-SNAP 或 3D Slicer 手工勾画的肿瘤边界，是后续生境分析的**必需输入文件**，请勿删除

---

#### **步骤 2：生境聚类分析 (Habitat Segmentation)**

**📋 临床意义**：

- 传统影像诊断只能看到"整个肿瘤"，就像只知道一个城市的总人口
- 生境分析能自动识别肿瘤内部的不同亚区（如坏死区、活跃增殖区、缺氧区）
- 这些亚区对治疗的反应不同，预后也不同
- **实际应用举例**：某些亚区可能对放疗敏感，而另一些亚区可能需要靶向治疗

```bash
# 二步法生境分析（demo 默认演示策略：先个体级超体素，再群体级生境）
habit get-habitat --config demo_data/config_habitat_two_step.yaml

# （可选）一步法 / direct-pooling 策略
# habit get-habitat --config demo_data/config_habitat_one_step.yaml
# habit get-habitat --config demo_data/config_habitat_direct_pooling.yaml
```

**⏱️ 预计用时**：约 5-10 分钟
**📁 结果位置**：`demo_data/results/habitat_two_step/`（结果目录名由配置中的 `out_dir` 决定，one-step / direct-pooling 会分别落在 `habitat_one_step/` / `habitat_direct_pooling/`）
**🔍 如何查看**：将生成的 `subj001_habitats.nrrd`、`subj001_supervoxel.nrrd` 等文件拖入 ITK-SNAP，叠加在原始影像上查看

---

#### **步骤 3：特征提取 (Feature Extraction)**

**📋 临床意义**：

- 从影像中提取数百个定量指标（纹理、形状、强度等）
- 就像血常规检查能得到白细胞、红细胞、血红蛋白等多个指标一样
- 这些特征能量化肿瘤的异质性，捕捉肉眼无法识别的信息
- **实际应用举例**：某些纹理特征可能与肿瘤的恶性程度、侵袭性相关

```bash
# 同时提取传统影像组学 + 生境组学 + MSI + ITH 等特征
# 提取的特征类型在 config 中通过 feature_types 字段控制（traditional / whole_habitat / msi / ith_score 等）
habit extract --config demo_data/config_extract_features.yaml
```

**⏱️ 预计用时**：约 3-8 分钟
**📁 结果位置**：`demo_data/results/features/`（每类特征一个 CSV，详见下方「结果文件夹结构」）
**📊 结果格式**：CSV 表格，可用 Excel 打开查看
**⚠️ 前置条件**：`config_extract_features.yaml` 中 `habitats_map_folder` 需指向步骤 2 实际生成的目录（默认是 `./results/habitat_direct_pooling`，若您用了 two-step 策略，请改为 `./results/habitat_two_step`）

---

#### **步骤 4：机器学习建模 (Machine Learning)**

**📋 临床意义**：

- 从众多特征中找出与预后/疗效最相关的"生物标志物"
- 构建预测模型（如预测患者5年生存率、治疗反应等）
- 就像多因素回归分析，但能处理更复杂的非线性关系
- **实际应用举例**：帮助识别高危患者，指导个体化治疗方案

```bash
# 运行机器学习建模（包括特征选择、模型训练、性能评估）
# 该 demo 使用乳腺癌数据 (demo_data/ml_data/breast_cancer_dataset.csv) 训练 LogisticRegression
habit model --config demo_data/config_machine_learning_radiomics.yaml --mode train

# （可选）K 折交叉验证版本
# habit model --config demo_data/config_machine_learning_kfold.yaml --mode train
```

**⏱️ 预计用时**：约 2-5 分钟
**📁 结果位置**：`demo_data/ml_data/radiomics/`（路径由 config 中的 `output` 字段决定）
**📈 生成内容**：

- `prediction_results.csv` / `all_prediction_results.csv`（每个被试的预测概率与预测标签）
- `evaluation_metrics.csv`（AUC、准确率、灵敏度、特异度等指标）
- `roc_curve.pdf` / `calibration_curve.pdf` / `decision_curve.pdf` / `pr_curve.pdf`（性能曲线）
- `LogisticRegression_confusion_matrix.pdf`（混淆矩阵）
- 特征筛选过程图（如方差阈值、相关性矩阵等，由 `feature_selection_methods` 触发）

---

#### **步骤 5：模型比较 (Model Comparison)**

**📋 临床意义**：

- 比较不同模型（如传统影像组学 vs 生境特征）的预测性能
- 类似于比较不同诊断方法的敏感性和特异性
- 帮助您找到最佳的预测方案
- **实际应用举例**：证明生境分析比传统方法能提供更多有价值的信息

```bash
# 比较多个模型的性能（demo 中对比 radiomics 与 clinical 两个模型）
# 注意：运行前需要分别用 config_machine_learning_radiomics.yaml 与
# config_machine_learning_clinical.yaml 训练好两个模型，生成各自的
# all_prediction_results.csv
habit compare --config demo_data/config_model_comparison.yaml
```

**⏱️ 预计用时**：约 1-3 分钟
**📁 结果位置**：`demo_data/ml_data/model_comparison/`（路径由 config 中的 `output_dir` 字段决定）
**📊 生成内容**：

- `roc_curves.pdf`（多模型 ROC 曲线对比）
- `decision_curves.pdf`（多模型 DCA 决策曲线）
- `calibration_curves.pdf`（多模型校准曲线）
- `precision_recall_curves.pdf`（多模型 PR 曲线）
- `delong_results.json`（DeLong 检验：模型间 AUC 差异的统计学显著性）
- `combined_predictions.csv`（多模型预测结果合并表）

---

#### **🎯 快速运行全流程**

如果您想一次性运行所有步骤，可以使用以下命令：

```bash
# 依次运行所有步骤（适合熟悉流程后使用）
# 注意：模型对比 (habit compare) 需要 radiomics 与 clinical 两个模型都训练完毕，
# 因此下面分别用两个 ML 配置训练后再做对比
habit preprocess --config demo_data/config_preprocessing.yaml && \
habit get-habitat --config demo_data/config_habitat_two_step.yaml && \
habit extract --config demo_data/config_extract_features.yaml && \
habit model --config demo_data/config_machine_learning_radiomics.yaml --mode train && \
habit model --config demo_data/config_machine_learning_clinical.yaml --mode train && \
habit compare --config demo_data/config_model_comparison.yaml
```

**⚠️ 注意**：全流程运行约需 15-30 分钟，建议首次使用时逐步运行，熟悉每一步的输出结果。

---

### Step 3: 查看和理解结果 (View Results)

运行完成后，`demo_data/` 下会形成 **三块** 输出目录：`preprocessed/`（预处理产物）、`results/`（生境分析与特征）、`ml_data/`（机器学习与模型对比）。完整层次如下：

#### **📁 结果文件夹结构**

```text
demo_data/
│
├── 📂 preprocessed/                                <-- 预处理产物（与 results/ 同级，不在 results/ 下）
│   ├── 📄 processing.log                           <-- 预处理总日志
│   ├── 📂 dcm2nii_01/                              <-- 阶段 1：DICOM → NIfTI
│   │   └── images/<subject>/<modality>/<modality>.nii.gz
│   ├── 📂 resample_02/                             <-- 阶段 2：重采样（含 images/ 与 masks/）
│   ├── 📂 registration_03/                         <-- 阶段 3：模态间配准
│   ├── 📂 zscore_normalization_04/                 <-- 阶段 4：Z-Score 标准化
│   │   ├── images/<subject>/<modality>/<modality>.nii.gz
│   │   └── masks/<subject>/<modality>/<modality>.nii.gz
│   └── 📂 processed_images/                        <-- 最终预处理产物（供下游生境分析使用）
│       ├── images/<subject>/<modality>/<modality>.nii.gz
│       └── masks/<subject>/<modality>/<modality>.nii.gz
│       # 示例：images/subj001/delay2/delay2.nii.gz
│
├── 📂 results/                                     <-- 生境分析 + 特征提取结果
│   │
│   ├── 📂 habitat_two_step/                        <-- 生境分析结果（two-step 策略）
│   │   ├── 🖼️ subj001_habitats.nrrd                <-- 3D 生境地图（可用 ITK-SNAP 查看）
│   │   ├── 🖼️ subj001_supervoxel.nrrd              <-- 超体素地图（仅 two-step 有）
│   │   ├── 📊 habitats.csv                         <-- 各被试 × 各生境的统计明细
│   │   ├── 💾 habitat_pipeline.pkl                 <-- 训练好的生境流水线（用于 predict 模式）
│   │   ├── 📄 habitat_analysis.log                 <-- 运行日志
│   │   └── 📈 visualizations/
│   │       ├── habitat_clustering/                 <-- 群体级生境聚类
│   │       │   ├── habitat_clustering_2D.png       <-- PCA/t-SNE 2D 投影
│   │       │   ├── habitat_clustering_3D.png       <-- 3D 投影
│   │       │   └── kmeans_inertia_cluster_validation_scores.png   <-- 聚类有效性曲线
│   │       └── supervoxel_clustering/              <-- 个体级超体素聚类
│   │           ├── subj001_supervoxel_clustering_2D.png
│   │           └── subj001_supervoxel_clustering_3D.png
│   │   # 若使用 one-step / direct-pooling 策略，对应目录名为
│   │   #   results/habitat_one_step/  或  results/habitat_direct_pooling/
│   │
│   └── 📂 features/                                <-- 提取的特征（每类一个 CSV）
│       ├── 📊 raw_image_radiomics.csv              <-- 原始影像传统影像组学特征
│       ├── 📊 whole_habitat_radiomics.csv          <-- 全肿瘤生境组学特征
│       ├── 📊 msi_features.csv                     <-- 多区域空间交互（MSI）特征
│       ├── 📊 ith_scores.csv                       <-- 肿瘤内异质性（ITH）评分
│       ├── 📊 habitat_basic_features.csv           <-- 生境基础形态/计数特征
│       └── 📄 processing.log                       <-- 提取日志
│
└── 📂 ml_data/                                     <-- 机器学习与模型对比结果
    ├── 📊 breast_cancer_dataset.csv                <-- 演示用特征/标签表（demo 输入）
    ├── 📄 train_ids.txt / test_ids.txt             <-- 训练/测试集划分
    │
    ├── 📂 radiomics/                               <-- habit model（radiomics 配置）输出
    │   ├── 📊 all_prediction_results.csv           <-- 训练+测试合并的预测结果
    │   ├── 📊 prediction_results.csv               <-- 测试集预测结果
    │   ├── 📊 evaluation_metrics.csv               <-- AUC / 准确率 / 灵敏度 / 特异度
    │   ├── 🖼️ roc_curve.pdf                        <-- ROC 曲线
    │   ├── 🖼️ calibration_curve.pdf                <-- 校准曲线
    │   ├── 🖼️ decision_curve.pdf                   <-- DCA 决策曲线
    │   ├── 🖼️ pr_curve.pdf                         <-- 精确率-召回率曲线
    │   ├── 🖼️ LogisticRegression_confusion_matrix.pdf
    │   └── 💾 *_model.pkl                          <-- 训练好的模型
    │
    ├── 📂 clinical/                                <-- habit model（clinical 配置）输出（同上结构）
    │
    └── 📂 model_comparison/                        <-- habit compare 输出
        ├── 📊 combined_predictions.csv             <-- 多模型预测合并表
        ├── 🖼️ roc_curves.pdf                       <-- 多模型 ROC 对比
        ├── 🖼️ decision_curves.pdf                  <-- 多模型 DCA 对比
        ├── 🖼️ calibration_curves.pdf               <-- 多模型校准对比
        ├── 🖼️ precision_recall_curves.pdf          <-- 多模型 PR 对比
        └── 📄 delong_results.json                  <-- DeLong 检验结果（统计学差异检验）
```

#### **🔍 如何查看和使用这些结果？**

**1. 查看生境地图（临床最直观的结果）**

- 打开 ITK-SNAP 或 3D Slicer
- 加载原始影像（任选一个模态，例如 delay2）：
  `demo_data/preprocessed/processed_images/images/subj001/delay2/delay2.nii.gz`
- 叠加生境地图：`demo_data/results/habitat_two_step/subj001_habitats.nrrd`
- 您将看到肿瘤被自动分为不同颜色的区域，每种颜色代表一种生境
- 也可以加载 `subj001_supervoxel.nrrd` 查看更精细的超体素分割

**2. 查看特征数据（用于统计分析）**

- 用 Excel 或 SPSS 打开 `demo_data/results/features/` 下的任一 CSV，例如：
  - `raw_image_radiomics.csv`：原始影像的传统影像组学特征
  - `whole_habitat_radiomics.csv`：基于生境的影像组学特征
  - `msi_features.csv` / `ith_scores.csv`：多区域空间交互与肿瘤内异质性指标
- 每一行是一个病例（`subjID`），每一列是一个特征
- 可直接用于后续的统计分析或机器学习

**3. 查看模型性能（评估预测能力）**

- 性能曲线：`demo_data/ml_data/radiomics/roc_curve.pdf`、`calibration_curve.pdf`、`decision_curve.pdf`、`pr_curve.pdf`
- 混淆矩阵：`demo_data/ml_data/radiomics/LogisticRegression_confusion_matrix.pdf`
- 数值指标：`demo_data/ml_data/radiomics/evaluation_metrics.csv`（AUC、准确率、灵敏度、特异度）
- 经验：ROC 曲线越接近左上角越好，AUC > 0.8 通常视为具备较好的预测能力

**4. 用于论文撰写**

- 所有图表均为高分辨率 PDF 格式（矢量图，可无损放大），可直接插入论文
- `demo_data/ml_data/model_comparison/` 中的多模型对比图可用于展示方法学优势
- `delong_results.json` 中的 DeLong 检验 p 值可用于在文中报告"AUC 差异是否显著"
- CSV 文件可用于绘制更多自定义图表

#### **💡 结果解读小贴士**

**对于医生**：

- 生境地图可以帮助您直观理解肿瘤内部的异质性
- 不同颜色的区域可能代表不同的生物学行为（如增殖活跃区 vs 坏死区）
- 这些信息可以辅助治疗决策（如放疗剂量分布的优化）

**对于研究人员**：

- 特征文件可用于进一步的统计建模
- 模型性能指标可用于方法学对比
- 所有结果都有详细的日志记录，保证研究的可重复性

---

## 📚 核心功能模块


| 功能模块      | 临床应用价值                                                |
| --------- | ----------------------------------------------------- |
| **影像预处理** | 自动进行重采样、配准、N4 偏置场校正，**省去手动处理的繁琐**，保证数据一致性。            |
| **生境聚类**  | 提供一步法、二步法等策略，**自动发现**肉眼难以区分的肿瘤亚区（生境）。                 |
| **特征提取**  | 提取 **多区域空间交互 (MSI)** 和 **肿瘤内异质性 (ITH)** 特征，挖掘深层生物学信息。 |
| **机器学习**  | 内置全流程建模工具，从特征筛选到 AUC 曲线绘制，**助力高分文章发表**。               |


> **💡 术语小贴士**：
>
> - **体素 (Voxel)**：3D 影像中的像素点，影像分析的基本单位。
> - **超体素 (Supervoxel)**：将性质相似的相邻体素打包成的小方块。相比传统体素，它能更稳定地反映纹理特征，抗噪性更强，计算效率更高。

---

## 📖 文档与支持

详细的使用指南、参数说明和 API 文档请访问：

👉 **[HABIT 在线文档](https://lichao312214129.github.io/HABIT)**

- [如何准备数据？](https://lichao312214129.github.io/HABIT/user_guide/image_preprocessing_zh.html)
- [如何解读生境图？](https://lichao312214129.github.io/HABIT/user_guide/habitat_segmentation_zh.html)
- [常见问题 (FAQ)](https://lichao312214129.github.io/HABIT/index.html#faq)

---

## 👥 核心开发者

本项目的核心开发者为 **黎超** 和 **董梦实**。

---

## 🤝 引用与许可

如果您在研究中使用了 HABIT，请引用我们的工作：

> [引用信息待添加]

本项目采用 [MIT 许可证](LICENSE) 开源。

---

## 🙏 致谢

我们衷心感谢以下专家和机构对 HABIT 项目的支持与贡献：

- **黄腾医生** - 赣州市人民医院
- **王亚奇医生** -  安徽医科大学第三附属医院
- **李梦思博士** - 中南大学湘雅医院

他们在临床应用、算法改进和项目优化方面提供了宝贵的意见和反馈，为 HABIT 的发展做出了重要贡献。

---

由 HABIT 团队维护 | 欢迎提交 Issue 或 Pull Request