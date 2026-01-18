# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

<p align="center">
  <img src="https://www.imagilys.com/wp-content/uploads/2018/09/radiomics-illustration-500x500.png" alt="Radiomics" width="200"/>
</p>

<p align="center">
  <strong>📖 语言 / Language</strong><br>
  <a href="README.md">🇨🇳 简体中文</a> | <a href="README_en.md">🇬🇧 English</a>
</p>

<p align="center">
    <a href="https://github.com/your-repo/habit_project/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Active-green.svg"></a>
</p>

**HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** 是一个专为医学影像设计的、基于Python的综合性肿瘤"生境"分析工具包。它提供从影像预处理到机器学习的端到端解决方案，使研究人员能够通过影像组学和高级分析方法深入探究肿瘤的异质性。

---

## 📖 核心工作流

HABIT的核心思想是识别和表征肿瘤内部具有不同影像表型的亚区，即"生境"。这一目标通过一个多阶段的流水线实现：

<p align="center">
  <b>影像 → 体素特征 → 超体素(可选) → 生境 → 生境特征 → 预测模型(可选)</b>
</p>

### 核心概念层级图
*从微观体素到宏观生境的抽象过程*

```
       [肿瘤整体]             [微观结构]             [中观结构]             [宏观模式]
     +------------+         +------------+         +------------+         +------------+
     |   Tumor    |         |   Voxels   |         | Supervoxels|         |  Habitats  |
     |  (Image)   |         | (Features) |         | (Clusters) |         | (Patterns) |
     +-----+------+         +-----+------+         +-----+------+         +-----+------+
           |                      |                      |                      |
           v                      v                      v                      v
     +------------+         +------------+         +------------+         +------------+
     |            |         | . . . . . .|         | AA BB CC DD|         | ## ** @@   |
     |  (Image)   |  ---->  | . . . . . .|  ---->  | AA BB CC DD|  ---->  | ## ** @@   |
     |            |         | . . . . . .|         | EE FF GG HH|         | $$ %% &&   |
     +------------+         +------------+         +------------+         +------------+
        原始图像               体素特征               超体素                生境图
                                                  (局部过分割)           (具有生物学意义)
```

### 详细流程说明

1. **体素级特征提取**: 为肿瘤内的每一个体素提取丰富的特征（如信号强度、纹理、动态增强特征等）。
2. **超体素聚类**: 将空间上相邻且特征相似的体素分组，形成"超体素"。这一过分割步骤在简化图像的同时保留了局部边界信息。
3. **生境聚类**: 在整个患者队列中对超体素进行聚类，以识别共通的、反复出现的模式，从而形成最终的"生境"。
4. **特征工程**: 从这些生境中提取高阶特征，如它们的大小、形状、空间关系（MSI特征）和异质性（ITH分数）。
5. **机器学习**: 使用工程化的生境特征来训练预测模型，用于如患者生存期、治疗反应或疾病诊断等临床终点的预测。

### 三种聚类策略

HABIT 支持三种不同的聚类策略，适用于不同的研究场景：

#### 1️⃣ 一步法 (One-Step)
- **流程**：体素 → 生境（直接聚类）
- **特点**：每个患者独立确定最优聚类数，生境标签独立
- **适用场景**：个体异质性分析、小样本研究、每个患者需要个性化分析

#### 2️⃣ 二步法 (Two-Step) ⭐ 默认方法
- **流程**：体素 → 超体素 → 生境
  - **第一步**：对每个患者的体素进行聚类，生成超体素（如每个患者50个超体素）
  - **第二步**：将所有患者的超体素合并，进行群体级聚类，识别统一的生境模式
- **特点**：先个体聚类，再群体聚类，所有患者共享统一的生境标签
- **适用场景**：队列研究、跨患者生境模式识别、需要统一标签进行比较

#### 3️⃣ 直接拼接法 (Direct Pooling)
- **流程**：拼接所有患者的所有体素 → 直接群体聚类
- **特点**：跳过超体素步骤，直接对所有体素进行群体级聚类，所有患者共享统一标签
- **适用场景**：数据量适中、需要统一标签但不需要超体素中间步骤

### 🔍 三种聚类策略的可视化对比

#### 1. 一步法 (One-Step) - 个性化分析
*每个患者独立进行聚类，适合分析个体异质性。*

```
      Patient 1 (P1)              Patient 2 (P2)
   +------------------+        +------------------+
   |  P1 Tumor Image  |        |  P2 Tumor Image  |
   +--------+---------+        +--------+---------+
            |                           |
            v  (提取体素)                v
   +--------+---------+        +--------+---------+
   | Voxels: . . . .  |        | Voxels: . . . .  |
   +--------+---------+        +--------+---------+
            |                           |
            v  (独立聚类)                v
   +--------+---------+        +--------+---------+
   | Habitats: # * @  |        | Habitats: & % $  |
   +------------------+        +------------------+
      P1 独有生境                  P2 独有生境
    (标签互不通用)               (标签互不通用)
```

#### 2. 二步法 (Two-Step) - 队列研究 (⭐ 推荐)
*先生成超体素(Supervoxels)，再进行群体聚类。平衡了局部细节和群体一致性。*

```
      Patient 1 (P1)              Patient 2 (P2)
   +------------------+        +------------------+
   |  P1 Tumor Image  |        |  P2 Tumor Image  |
   +--------+---------+        +--------+---------+
            |                           |
            v                           v
   +--------+---------+        +--------+---------+
   | Voxels: . . . .  |        | Voxels: . . . .  |
   +--------+---------+        +--------+---------+
            |  (局部聚类)                |
            v                           v
   +--------+---------+        +--------+---------+
   | Supervoxels:     |        | Supervoxels:     |
   | AA BB CC DD      |        | EE FF GG HH      |
   +--------+---------+        +--------+---------+
            \                         /
             \   (汇聚所有超体素)    /
              \                     /
               v                   v
           +---------------------------+
           |   Population Clustering   |
           |    (群体级生境聚类)        |
           +-------------+-------------+
                         |
                         v
           +---------------------------+
           |  Unified Habitats (统一)  |
           |  Type 1: # (e.g. Necrosis)|
           |  Type 2: * (e.g. Active)  |
           |  Type 3: @ (e.g. Edema)   |
           +---------------------------+
             (所有患者共享相同的标签体系)
```

#### 3. 直接拼接法 (Direct Pooling)
*跳过超体素，直接对所有体素进行群体聚类。*

```
      Patient 1 (P1)              Patient 2 (P2)
   +------------------+        +------------------+
   |  P1 Tumor Image  |        |  P2 Tumor Image  |
   +--------+---------+        +--------+---------+
            |                           |
            v                           v
   +--------+---------+        +--------+---------+
   | Voxels: . . . .  |        | Voxels: . . . .  |
   +--------+---------+        +--------+---------+
             \                         /
              \    (直接拼接所有体素)   /
               \                     /
                v                   v
           +---------------------------+
           |   Population Clustering   |
           |    (群体级体素聚类)        |
           +-------------+-------------+
                         |
                         v
           +---------------------------+
           |  Unified Habitats (统一)  |
           |     Type 1: #, 2: *, 3: @ |
           +---------------------------+
```


### 📊 策略选择指南

**选择一步法如果：**
- 想要逐个分析每个肿瘤
- 患者间样本大小差异很大
- 对个性化生境模式感兴趣
- 计算资源有限

**选择二步法如果：**
- 正在进行队列研究
- 需要跨患者可比较的生境 ⭐ **大多数研究**
- 想要平衡计算效率与生物学相关性
- 需要可解释的中间结果（超体素）

**选择直接拼接法如果：**
- 拥有适中的计算资源
- 想要统一生境但不需要超体素中间步骤
- 处理的数据集适合体素级聚类

**三种方法对比表**：

| 特性 | 一步法 | 二步法 | 直接拼接法 |
|------|--------|--------|------------|
| **聚类流程** | 体素→生境 | 体素→超体素→生境 | 拼接所有体素→生境 |
| **聚类层级** | 单层级（个体） | 双层级（个体+群体） | 单层级（群体） |
| **生境标签** | 每个患者独立 | 所有患者统一 | 所有患者统一 |
| **计算复杂度** | 低 | 中等 | 高（取决于总体素数） |
| **适用场景** | 个体异质性分析 | 队列研究（推荐） | 中等规模数据 |

---

## 🧪 快速测试（使用示例数据）

**🎯 重要提示**：HABIT 提供了完整的示例数据，您无需准备自己的数据即可快速体验所有功能！

### 使用示例数据快速运行

项目中的 `demo_data/` 目录包含了：
- ✅ 示例 DICOM 影像数据（2个受试者）
- ✅ 预处理后的影像和掩膜
- ✅ 完整的配置文件示例
- ✅ 示例分析结果

### 三步快速体验

```bash
# 1. 确保已安装 HABIT（见下方安装指南）
# 2. 激活环境
conda activate habit

# 3. 使用示例数据运行 Habitat 分析
habit get-habitat --config demo_data/config_habitat.yaml
```

**预期结果**：
- 分析完成后，结果将保存在 `demo_data/results/habitat/` 目录下
- 您将看到：
  - `habitats.csv` - 生境标签结果
  - `subj001_habitats.nrrd` 和 `subj002_habitats.nrrd` - 生境地图（可用 ITK-SNAP 或 3D Slicer 查看）
  - `visualizations/` - 自动生成的可视化图表
  - `supervoxel2habitat_clustering_strategy_bundle.pkl` - 训练好的模型

### 参考示例配置文件

所有示例配置文件都在 `demo_data/` 目录下：
- `config_habitat.yaml` - Habitat 分析配置（推荐从这里开始）
- `config_preprocessing.yaml` - 影像预处理配置
- `config_icc.yaml` - ICC 分析配置

**💡 提示**：您可以复制这些配置文件并根据自己的数据修改路径和参数。

---

## 🛠️ 安装

详细指南请参见 [**INSTALL.md**](INSTALL.md)。

### 快速安装步骤

```bash
# 1. 克隆仓库
git clone <repository_url>
cd habit_project

# 2. 创建并激活Conda环境
conda create -n habit python=3.8
# 如果使用autogluon，则需要创建py310或以上的环境
# conda create -n habit python=3.10
conda activate habit

# 3. 安装依赖
pip install -r requirements.txt

# 4. 以可编辑模式安装HABIT包
pip install -e .
```

### 验证安装

```bash
# 检查命令是否可用
habit --help

# 如果看到命令列表，说明安装成功！
```

---

## 📖 快速入门

### 🎯 统一命令行界面 (CLI) - **推荐使用方式**

**HABIT 提供了统一、简洁的命令行界面！** ✨ 

使用基于 **Click** 构建的 CLI 系统，您只需使用 `habit` 命令即可访问所有功能，无需记住复杂的脚本路径。

#### 安装后立即使用

完成 `pip install -e .` 后，`habit` 命令将在您的环境中全局可用：

```bash
# 查看所有可用命令
habit --help

# 查看特定命令的帮助信息
habit get-habitat --help
```

#### 核心命令示例

```bash
# 1️⃣ 图像预处理 - 重采样、配准、标准化
habit preprocess --config config/config_image_preprocessing.yaml

# 2️⃣ 生成 Habitat 地图 - 识别肿瘤亚区
# 支持一步法、二步法或直接拼接法
habit get-habitat --config demo_data/config_habitat.yaml

# 3️⃣ 提取 Habitat 特征 - MSI, ITH等高级特征
habit extract --config config/config_extract_features.yaml

# 4️⃣ 机器学习 - 训练预测模型
habit model --config config/config_machine_learning.yaml --mode train

# 5️⃣ 模型预测 - 使用训练好的模型
habit model --mode predict \
  --model ./ml_data/model_package.pkl \
  --data ./new_data.csv \
  --output ./predictions/

# 6️⃣ K折交叉验证 - 更稳健的模型评估
habit cv --config config/config_machine_learning_kfold.yaml

# 7️⃣ 模型比较 - ROC, DCA, 校准曲线等可视化
habit compare --config config/config_model_comparison.yaml

# 8️⃣ ICC分析 - 特征可重复性评估
habit icc --config config/config_icc_analysis.yaml
```

#### 快速参考表

| 命令 | 功能 | 配置文件示例 | 文档 |
|------|------|-------------|:---:|
| `habit preprocess` | 图像预处理 | `config_image_preprocessing.yaml` | [📖](doc/app_image_preprocessing.md) |
| `habit get-habitat` | 生成Habitat地图 | `demo_data/config_habitat.yaml` ⭐ | [📖](doc/app_habitat_analysis.md) |
| `habit extract` | 提取Habitat特征 | `config_extract_features.yaml` | [📖](doc/app_extracting_habitat_features.md) |
| `habit model` | 机器学习训练/预测 | `config_machine_learning.yaml` | [📖](doc/app_of_machine_learning.md) |
| `habit cv` | K折交叉验证 | `config_machine_learning_kfold.yaml` | [📖](doc/app_kfold_cross_validation.md) |
| `habit compare` | 模型比较与可视化 | `config_model_comparison.yaml` | [📖](doc/app_model_comparison_plots.md) |
| `habit icc` | ICC可重复性分析 | `config_icc_analysis.yaml` | [📖](doc/app_icc_analysis.md) |

---

## 🔬 完整研究流程

一个典型的基于HABIT的影像组学研究项目包含以下步骤。HABIT工具包为其中标记为 `[HABIT]` 的步骤提供了强大支持。

1. **数据采集与下载**: 从医院PACS系统或公开数据集中获取原始影像数据（通常为DICOM格式）。
2. **数据整理与匿名化**: 将数据按 `患者/序列/文件` 的结构进行整理，对患者隐私信息进行匿名化处理。
3. **格式转换 (DICOM to NIfTI)**: `[HABIT]` 使用 `habit preprocess` 命令将DICOM序列转换为NIfTI格式。
4. **感兴趣区域 (ROI) 分割**: 由放射科医生或研究人员使用ITK-SNAP, 3D Slicer等专业软件手动勾画肿瘤区域（ROI），并保存为mask文件。
5. **影像预处理**: `[HABIT]` 使用 `habit preprocess` 命令进行配准、重采样、强度标准化、N4偏置场校正等预处理。
6. **生境分析与特征提取**: 
   - `[HABIT]` 运行 `habit get-habitat` 命令来识别肿瘤生境（支持一步法、二步法、直接拼接法）
   - `[HABIT]` 运行 `habit extract` 命令从生境中提取高级特征（如MSI, ITH分数等）
7. **构建与评估预测模型**: 
   - `[HABIT]` 使用 `habit model` 命令进行特征选择、模型训练和内部验证
   - `[HABIT]` 使用 `habit compare` 命令对不同模型进行性能比较和可视化
8. **结果分析与论文撰写**: 解释模型的发现，并撰写研究论文。

---

## 🚀 主要功能

| 类别 | 功能 | 描述 | 文档 |
| :--- | :--- | :--- | :---: |
| 🖼️ **影像处理** | **预处理流水线** | 提供DICOM转换、重采样、配准和标准化的端到端工具。 | [📖](doc/app_image_preprocessing.md) |
| | **N4偏置场校正** | 校正MRI扫描中的信号强度不均匀性。 | [📖](doc/app_image_preprocessing.md) |
| 🧬 **生境分析** | **一步法聚类** | 直接聚类到生境，每个肿瘤独立确定聚类数，生境标签不统一。 | [📖](doc/app_habitat_analysis.md) |
| | **二步法聚类** | 两阶段聚类（个体supervoxels → 群体habitats），统一生境标签体系。 | [📖](doc/app_habitat_analysis.md) |
| | **直接拼接法** | 拼接所有体素直接聚类，跳过超体素步骤。 | [📖](doc/app_habitat_analysis.md) |
| | **🎨 自动可视化** | 自动生成2D/3D聚类散点图、最优聚类数曲线等高质量可视化结果。 | [📖](doc/app_habitat_analysis.md) |
| 🔬 **特征提取** | **高级特征集** | 提取传统影像组学、多区域空间交互（MSI）和肿瘤内异质性（ITH）等特征。 | [📖](doc/app_extracting_habitat_features.md) |
| 🤖 **机器学习** | **完整工作流** | 包括数据分割、特征选择、模型训练和评估。 | [📖](doc/app_of_machine_learning.md) |
| | **K折交叉验证** | 完善的K折交叉验证流程，支持多模型评估和可视化。 | [📖](doc/app_kfold_cross_validation.md) |
| | **模型比较** | 提供生成ROC曲线、决策曲线分析（DCA）和执行DeLong检验的工具。 | [📖](doc/app_model_comparison_plots.md) |
| 📊 **验证与工具** | **可复现性分析** | 包括测试-重测（Test-Retest）和组内相关系数（ICC）分析工具。 | [📖](doc/app_icc_analysis.md) |

---

## ❓ 常见问题

### Q1: 如何开始使用 HABIT？

**推荐方式**：使用 `demo_data` 中的示例数据快速体验！

```bash
# 1. 确保已安装（见安装章节）
conda activate habit

# 2. 运行示例
habit get-habitat --config demo_data/config_habitat.yaml

# 3. 查看结果
# 结果在 demo_data/results/habitat/ 目录下
```

### Q2: `habit` 命令找不到怎么办？

**解决方案**：
```bash
# 确保已激活正确的环境
conda activate habit

# 重新安装
pip install -e .

# 验证安装
habit --help
```

### Q3: 如何修改配置文件？

**推荐方式**：
1. 复制 `demo_data/config_habitat.yaml` 作为模板
2. 修改其中的路径和参数
3. 主要需要修改的参数：
   - `data_dir`: 您的数据路径
   - `out_dir`: 输出结果路径
   - `FeatureConstruction.voxel_level.method`: 特征提取方法
   - `HabitatsSegmention.clustering_mode`: 选择聚类策略（one_step/two_step/direct_pooling）

### Q4: 如何查看分析结果？

**结果位置**：
- CSV文件：`{out_dir}/habitats.csv` - 可用Excel打开查看
- 图像文件：`{out_dir}/*_habitats.nrrd` - 可用 ITK-SNAP 或 3D Slicer 查看
- 可视化图表：`{out_dir}/visualizations/` - PNG格式，可直接查看

### Q5: 三种聚类策略如何选择？

- **一步法**：适合每个患者需要个性化分析，样本差异大的情况
- **二步法**：适合队列研究，需要统一标签进行比较（**推荐用于大多数研究**）
- **直接拼接法**：适合数据量适中，需要统一标签但不需要超体素中间步骤

### Q6: 如何理解输出结果？

- **habitats.csv**：包含每个超体素（或体素）的生境标签
- **habitat地图**：3D图像，不同颜色代表不同的生境
- **可视化图表**：帮助理解聚类效果和最优聚类数

---

## 🤝 贡献

欢迎各种形式的贡献！请参考贡献指南（待添加）或开启一个Issue来讨论您的想法。

## 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 🔬 引用

如果您在研究中使用了 HABIT，请考虑引用：
> [引用信息待添加]

## 🙋‍♀️ 支持

如果您遇到任何问题或有改进建议，请：
1. 阅读 `doc/` 文件夹中的详细文档
2. 在 GitHub 上提交一个 [Issue](https://github.com/your-repo/habit_project/issues)

### 📖 多语言文档

HABIT提供完整的中英文双语文档：
- **中文文档**: 位于 `doc/` 目录
- **English Documentation**: 位于 `doc_en/` 目录

💡 **语言切换**: 点击页面顶部的 "🇬🇧 English" 或 "🇨🇳 简体中文" 链接即可快速切换语言。

---

**祝使用愉快！** 🎉
