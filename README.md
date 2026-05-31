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

## ⚡ 5 分钟快速开始

**学习路径**：装环境 → 装 HABIT → 跑 demo。更详细的分步说明见 [在线安装指南](https://lichao312214129.github.io/HABIT/getting_started/installation_zh.html) 与 [完整 Demo 教程](https://lichao312214129.github.io/HABIT/getting_started/quickstart_zh.html)。

### Step 0：安装 Miniconda 或 Anaconda（仅需一次）

> 若已安装 **Miniconda** 或 **Anaconda**，可跳过本节，直接从 Step 1 开始。

**二选一即可**（推荐 Miniconda，体积更小）：

| 选项 | 下载 |
|------|------|
| **Miniconda**（推荐） | [官方下载页](https://docs.anaconda.com/miniconda) 或 [Windows 直链](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) |
| **Anaconda** | [Anaconda 下载页](https://www.anaconda.com/download) |

**（1）下载并安装**

- Windows：下载 `.exe`，双击后一路 **Next** 完成安装。
- macOS：下载 `.pkg`，一路 **继续 / 安装**；首次使用请在 Terminal 中执行 `conda init` 后重启终端。

**（2）打开终端（模拟界面）**

Windows 用户请从开始菜单打开 **Anaconda Prompt** 或 **Anaconda Powershell Prompt**：

```text
+--------------------------------------------------+
|  [开始]  搜索: Anaconda Prompt                    |
+--------------------------------------------------+
|  > Anaconda Prompt                               |
|  > Anaconda Powershell Prompt   <-- 任选其一     |
+--------------------------------------------------+
```

**（3）验证：命令行前缀应出现 `(base)`**

```text
+--------------------------------------------------+
| Anaconda Prompt                          - x     |
+--------------------------------------------------+
| (base) C:\Users\YourName>_                       |
|       ^^^^                                       |
|       出现 (base) 表示 conda 已就绪               |
+--------------------------------------------------+
```

---

### Step 1：创建环境并安装 HABIT

**（1）下载源码**

👉 **[点击下载 HABIT 压缩包](https://github.com/lichao312214129/HABIT/archive/refs/heads/main.zip)**，**直接解压**到任意位置（Windows 如 `D:\`，macOS 如「下载」文件夹），无需新建或重命名文件夹；解压后会自动出现 **`HABIT-main`** 目录。

**（2）在终端中依次执行**

先创建 Python 3.10 环境（兼容 PyTorch、AutoGluon 等深度学习依赖）：

```bash
conda create -n habit python=3.10 -y
conda activate habit
```

激活成功后，前缀由 `(base)` 变为 `(habit)`：

```text
(base) C:\Users\YourName> conda activate habit
(habit) C:\Users\YourName>_
       ^^^^^^
```

**（3）配置 pip 镜像（中国大陆用户强烈推荐）**

```bash
pip config set global.timeout 6000
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.extra-index-url https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
```

**（4）进入项目目录并安装**

`cd` 到解压后的 **`HABIT-main`** 目录（Git 克隆时目录名为 `HABIT`，路径按本机实际位置修改）：

```bash
# Windows（ZIP 解压到 D:\ 后）
cd D:\HABIT-main
# Windows 跨盘符时先输入 D: 再 cd

# macOS / Linux（例如解压到「下载」/ Downloads）
cd ~/Downloads/HABIT-main
# 若在桌面: cd ~/Desktop/HABIT-main
```

确认当前目录（ZIP 解压后典型结构）：

```text
  HABIT-main/
  ├── config/
  ├── habit/
  ├── requirements.txt   <-- 必须能看到此文件
  └── setup.py
```

```bash
# Windows: dir requirements.txt
# macOS / Linux: ls requirements.txt
pip install -r requirements.txt
pip install -e .
habit --version
```

> ``requirements.txt`` 含 ``numpy==1.26.1`` 与 GPU 版 ``torch==2.4.0+cu121``（CUDA 12.1）。无 NVIDIA GPU 时请先注释文件末尾 torch 相关行，再单独安装 CPU 版 torch。

安装成功时终端会出现 ``Successfully installed habit-...``，且 ``habit --version`` 能输出版本号。

**❌ 安装失败？**

1. 打开 `requirements.txt`，逐行执行 `pip install <包名>`（如 `pip install SimpleITK==2.2.1`）定位问题包。
2. 仍无法解决：[提交 Issue](https://github.com/lichao312214129/HABIT/issues) 或邮件 **[lichao19870617@163.com](mailto:lichao19870617@163.com)**。

---

### Step 2：下载 demo 并跑通第一条命令

**📦 演示数据**

- **文件**：`demo_data.rar`
- **链接**：[百度网盘](https://pan.baidu.com/s/1bHTLvVMHnfiApArmZf8wrQ)
- **提取码**：`kbmd`
- 解压到项目根目录下的 `demo_data/`（与 `config/` 同级）

> 所有隐私信息已去除；**仅供学术研究**，严禁商业用途。

**第一条命令**（影像预处理，约 2–5 分钟）：

```bash
conda activate habit
cd D:\HABIT-main          # Windows
# cd ~/Downloads/HABIT-main   # macOS / Linux
habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml
```

> demo 包内已含 `demo_data/preprocessed/`，也可跳过预处理，直接从生境分析开始。

---

### 下一步：完整研究流程（5 步）

| 步骤 | 命令概要 | 在线文档 |
|------|----------|----------|
| 1 预处理 | `habit preprocess --config ...` | [快速入门](https://lichao312214129.github.io/HABIT/getting_started/quickstart_zh.html) |
| 2 生境分析 | `habit get-habitat --config ...` | [生境分割](https://lichao312214129.github.io/HABIT/user_guide/habitat_segmentation_zh.html) |
| 3 特征提取 | `habit extract --config ...` | [特征提取](https://lichao312214129.github.io/HABIT/user_guide/habitat_feature_extraction_zh.html) |
| 4 机器学习 | `habit model --config ... --mode train` | [机器学习建模](https://lichao312214129.github.io/HABIT/user_guide/machine_learning_modeling_zh.html) |
| 5 模型对比 | `habit compare --config ...` | [模型比较](https://lichao312214129.github.io/HABIT/app_model_comparison_zh.html) |

完整命令、结果目录树与 ITK-SNAP 查看方法见 **[完整 Demo 教程](https://lichao312214129.github.io/HABIT/getting_started/quickstart_zh.html)**。

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
- **汤君杰博士** - 南方医科大学附属珠江医院

他们在临床应用、算法改进和项目优化方面提供了宝贵的意见和反馈，为 HABIT 的发展做出了重要贡献。

---

由 HABIT 团队维护 | 欢迎提交 Issue 或 Pull Request