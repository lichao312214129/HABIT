# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

<p align="center">
  <img src="https://www.imagilys.com/wp-content/uploads/2018/09/radiomics-illustration-500x500.png" alt="Radiomics" width="200"/>
</p>

<p align="center">
  <strong>📖 语言 / Language</strong><br>
  <a href="README.md">🇨🇳 简体中文</a> | <a href="README_en.md">🇬🇧 English</a>
</p>

<p align="center">
    <a href="https://github.com/lichao312214129/HABIT/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Active-green.svg"></a>
</p>

> **专为临床医生和研究人员设计的肿瘤"生境"（Habitat）分析与异质性评估工具箱。**
>
> *无需复杂的编程知识，即可开展顶刊级别的影像组学与生境分析研究。*

---

## 🌟 为什么选择 HABIT？

在精准医疗时代，**肿瘤异质性**是影响患者预后和治疗反应的关键因素。传统的影像组学往往提取整个肿瘤的平均特征，掩盖了肿瘤内部的空间异质性。

**HABIT 致力于解决这一痛点：**

1.  **自动划分生境**：自动识别肿瘤内部的坏死区、活性肿瘤区、水肿区等具有不同生物学行为的亚区域。
2.  **无缝对接临床**：完美支持由 **ITK-SNAP** 或 **3D Slicer** 勾画的 ROI，生成的生境地图可直接拖回软件中叠加显示。
3.  **零代码门槛**：通过修改简单的配置文件（类似填表）即可运行，无需编写复杂的代码。
4.  **一站式流程**：从 DICOM 预处理 -> 生境聚类 -> 特征提取 -> 机器学习建模 -> 绘制图表，全流程覆盖。

---

## 👨‍⚕️ 临床工作流：从影像到发表

HABIT 将复杂的计算过程封装为简单的四个步骤：

1.  **准备数据 (Prepare)**
    *   使用您熟悉的 **ITK-SNAP** 或 **3D Slicer** 勾画肿瘤 ROI，保存为 NIfTI (`.nii.gz`) 格式。
2.  **填写配置 (Config)**
    *   修改 `.yaml` 配置文件，指定您的影像文件夹路径（就像填写 Excel 表格一样简单）。
3.  **一键运行 (Run)**
    *   在终端运行一行命令，HABIT 自动完成所有计算。
4.  **获取结果 (Result)**
    *   获得 **彩色生境地图**（用于直观展示）和 **量化特征表格**（用于统计分析）。

---

## ⚡ 5分钟快速开始

### Step 0: 准备基础环境 (仅需一次)

> 如果您是第一次使用 Python 软件，请先安装 **Miniconda**。
> *如果您已经安装了 Anaconda 或 Miniconda，请跳过此步骤。*

#### 💻 Windows 用户
1.  **下载**：👉 [点击下载 Windows 版 Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
2.  **安装**：双击下载的 `.exe` 文件，**一路点击 "Next"**，直到安装完成。
    *   *建议勾选 "Add Miniconda3 to my PATH environment variable"（如果出现该选项）。*
3.  **打开终端**：
    *   点击屏幕左下角的 **"开始"** 菜单。
    *   搜索 **"Anaconda Prompt"**（或 **"Miniconda Prompt"**）。
    *   点击出现的黑色图标。

#### 🍎 macOS (苹果电脑) 用户
1.  **下载**：
    *   **M1/M2/M3 芯片** (2020年后的新款): 👉 [点击下载 Apple Silicon 版](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)
    *   **Intel 芯片** (旧款): 👉 [点击下载 Intel 版](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)
    *   *如果不确定，点击屏幕左上角苹果图标 -> "关于本机" 查看。*
2.  **安装**：双击下载的 `.pkg` 文件，按照提示一路点击 "继续" 或 "安装"。
3.  **打开终端**：
    *   按键盘上的 `Command ⌘` + `空格`。
    *   输入 **"Terminal"** (或 **"终端"**) 并回车。
    *   在终端中输入 `conda init` 并回车，然后**重启终端**。

---

*✅ 验证成功示例：*
> 您应该看到类似下面的黑色窗口，最前面有 **`(base)`** 字样：
> ```text
> (base) C:\Users\YourName> _
> ```

### Step 1: 下载与安装 HABIT

**1. 下载工具包**
👉 [**点击下载最新版 HABIT 压缩包**](https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip)

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
> 命令行前缀会从 `(base)` 变成 **`(habit)`**：
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
> ```text
> Successfully installed habit-1.0.0
> (habit) D:\HABIT> _
> ```

**❌ 如果安装失败怎么办？**
如果运行 `pip install` 时出现红色报错，请尝试以下步骤：
1.  **逐个排查**：打开项目文件夹下的 `requirements.txt`，手动逐行运行 `pip install <包名>`（例如 `pip install SimpleITK`），找出具体是哪个包安装失败。
2.  **寻求帮助**：如果无法解决，请在 GitHub 上 [提交 Issue](https://github.com/lichao312214129/HABIT/issues) 或者直接发送邮件联系作者：**lichao19870617@163.com**。

### Step 2: 下载演示数据 (Download Demo Data)

演示数据已上传至百度网盘，请先下载并解压：

**📦 下载演示数据**

链接: https://pan.baidu.com/s/1cBw6WtLtOXNE7vpF8429NA  
提取码: `xypk`

**⚠️ 重要说明**：
- 所有隐私信息已完全去除
- **严禁商业用途，仅供学术研究和Demo演示使用**
- 下载后请将文件解压到 `demo_data` 目录下

解压完成后，运行以下命令：

```bash
# 运行生境分析示例（一步法）
habit get-habitat --config demo_data/config_habitat_one_step.yaml
```

### Step 3: 查看结果 (Output)

运行完成后，打开 `demo_data/results/habitat/` 文件夹，您将看到类似下面的文件结构：

```text
demo_data/results/habitat/
├── 🖼️ sub001_habitats.nrrd      <-- 3D 生境地图 (拖入 ITK-SNAP 查看)
├── 📊 habitats.csv              <-- 详细特征表格 (Excel 打开)
└── 📈 visualizations/           <-- 自动生成的统计图表
    ├── cluster_centroids.png    <-- 聚类中心图
    └── feature_heatmap.png      <-- 特征热图
```

*   🖼️ **`*_habitats.nrrd`**：**3D 生境地图**。
    *   *如何查看？* 直接拖入 ITK-SNAP，叠加在原始影像上，您将看到五颜六色的分区，直观展示肿瘤内部异质性。
*   📊 **`habitats.csv`**：包含每个体素的详细分类结果。
*   📈 **`visualizations/`**：自动生成的聚类分析图表，可直接用于论文插图。

---

## 📚 核心功能模块

| 功能模块 | 临床应用价值 |
| :--- | :--- |
| **影像预处理** | 自动进行重采样、配准、N4 偏置场校正，**省去手动处理的繁琐**，保证数据一致性。 |
| **生境聚类** | 提供一步法、二步法等策略，**自动发现**肉眼难以区分的肿瘤亚区（生境）。 |
| **特征提取** | 提取 **多区域空间交互 (MSI)** 和 **肿瘤内异质性 (ITH)** 特征，挖掘深层生物学信息。 |
| **机器学习** | 内置全流程建模工具，从特征筛选到 AUC 曲线绘制，**助力高分文章发表**。 |

> **💡 术语小贴士**：
> *   **体素 (Voxel)**：3D 影像中的像素点，影像分析的基本单位。
> *   **超体素 (Supervoxel)**：将性质相似的相邻体素打包成的小方块。相比传统体素，它能更稳定地反映纹理特征，抗噪性更强，计算效率更高。

---

## 📖 文档与支持

详细的使用指南、参数说明和 API 文档请访问：

👉 **[HABIT 在线文档](https://lichao312214129.github.io/HABIT)**

*   [如何准备数据？](https://lichao312214129.github.io/HABIT/user_guide/image_preprocessing_zh.html)
*   [如何解读生境图？](https://lichao312214129.github.io/HABIT/user_guide/habitat_segmentation_zh.html)
*   [常见问题 (FAQ)](https://lichao312214129.github.io/HABIT/index.html#faq)

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

<p align="center">由 HABIT 团队维护 | 欢迎提交 Issue 或 Pull Request</p>
