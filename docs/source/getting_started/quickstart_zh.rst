完整 Demo 教程
==============

本教程带您走完典型的影像组学研究全流程：从原始影像到预测模型与模型对比。

一、前提条件
------------

1. 已完成 :doc:`installation_zh` 中的 HABIT 安装
2. 已准备 **工作根目录**：该目录下同时存在 ``config/`` 与 ``demo_data/`` （Demo YAML 使用 ``../../demo_data/...`` 等相对路径，二者须与 **运行 ``habit`` 时的当前目录** 匹配）。

**Windows 便携包用户**（推荐）

1. 便携包解压并完成 ``setup_habit.bat`` （见 :doc:`installation_zh` 方式一）
2. 从网盘下载 ``config.zip`` 、``demo_data.rar`` ，解压到 **pack 根目录**（与 ``python.exe`` 同级，如 ``D:\habit-cpu\``）
3. 每次跑 Demo 前 **无需** ``conda activate``；新开终端后：

   .. code-block:: bash

      cd /d D:\habit-cpu
      habit --version

**源码 / GitHub ZIP 用户**（方式二）

1. ``conda activate habit`` ，``cd`` 到含 ``config/`` 、``habit/`` 的项目根（ZIP 解压后多为 ``HABIT-main``）
2. 从网盘下载 ``demo_data.rar`` 解压到项目根下的 ``demo_data/``

   .. code-block:: bash

      conda activate habit
      cd "D:\HABIT-main"

若解压造成 ``HABIT-main/HABIT-main`` 嵌套，须 ``cd`` 到 **最内层** 且含 ``config/`` 、``habit/`` 的那一级（详见 :doc:`installation_zh`）。

二、下载配置与演示数据
----------------------

便携包 **不含** ``config/`` 、``demo_data/`` ；请从百度网盘单独下载（链接与提取码见 :doc:`installation_zh` 「配置、演示数据与测试」表格）。

+------------------+---------------------+
| 网盘文件         | 解压目标            |
+==================+=====================+
| ``config.zip``   | 工作根目录 → ``config/`` |
+------------------+---------------------+
| ``demo_data.rar``| 工作根目录 → ``demo_data/`` |
+------------------+---------------------+

**演示数据**

- **链接**: |demo_data_link|
- **提取码**: |demo_data_code|

**重要说明**

- 所有隐私信息已完全去除
- **严禁商业用途，仅供学术研究和 Demo 演示使用**

**便携包解压后目录示意**（``D:\habit-cpu\`` 为例）：

.. code-block:: text

   D:\habit-cpu\
   ├── python.exe
   ├── setup_habit.bat
   ├── Scripts\
   ├── config\               <-- config.zip
   └── demo_data\            <-- demo_data.rar
       ├── dicom/
       ├── preprocessed/processed_images/   <-- Demo 影像与 mask（可跳过步骤 1）
       └── ...

**源码安装目录示意**：

.. code-block:: text

   HABIT-main/
   ├── config/
   ├── demo_data/
   └── habit/

三、完整研究流程（5 步）
------------------------

demo 已含预处理结果，**首次可跳过步骤 1** ，从步骤 2 开始。改参数见 :doc:`../configuration_zh`。
下文 YAML 均在 **`config/`** 下；完整场景索引见 ``config/README_CONFIG.md`` 。

**请先** ``cd`` **到工作根目录**（便携包为 pack 根，如 ``D:\habit-cpu`` ；源码为 ``HABIT-main`` ）。

**步骤 1 — 预处理** （约 2–5 分钟：重采样 → SimpleITK 配准 → Z-score）→ ``demo_data/results/preprocessed/processed_images/`` （输入可仍用 ``demo_data/preprocessed/processed_images/``）

.. code-block:: bash

   habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml

**步骤 2 — 生境分割** （约 5–10 分钟）→ ``demo_data/results/habitat_two_step/``

.. code-block:: bash

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

将 ``demo_data/results/habitat_two_step/subj001_habitats.nrrd`` 拖入 ITK-SNAP，叠加在 ``demo_data/preprocessed/processed_images/images/subj001/delay2/`` 下的参考序列上查看。步骤 2–4 的输入影像均来自 ``demo_data/preprocessed/processed_images/``。

**步骤 3 — 特征提取** （约 3–8 分钟）→ ``demo_data/results/features/``

.. code-block:: bash

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

**步骤 4 — 机器学习** （约 2–5 分钟）→ ``demo_data/results/ml/radiomics/``

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

**步骤 5 — 模型对比** （约 1–3 分钟，需先训练 radiomics 与 clinical 两个模型）→ ``demo_data/results/model_comparison/``

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

四、主要输出目录
----------------

- ``demo_data/preprocessed/processed_images/`` — 预处理影像与 mask
- ``demo_data/results/habitat_two_step/`` — 生境地图（ITK-SNAP 叠加查看）
- ``demo_data/results/features/`` — 特征 CSV（Excel / SPSS）
- ``demo_data/results/ml/`` — 模型训练输出（含 ROC 等 PDF）
- ``demo_data/results/model_comparison/`` — 多模型对比图与指标

下一步
------

- :doc:`../user_guide/index_zh` — 各步命令速查
- :doc:`../configuration_zh` — 修改 YAML 参数
