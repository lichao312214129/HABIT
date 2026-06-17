完整 Demo 教程
==============

跑通典型流程：预处理 → 生境分割 → 特征提取 → 机器学习 → 模型对比。

一、准备
--------

.. note::

   下文 ``D:\habit-cpu`` 等路径仅为 **文档示例**，请改为您本机 **便携包或项目根目录的实际路径**。

1. 已完成 :doc:`installation_zh` 中的安装。
2. 从百度网盘下载下列文件，**解压到工作文件夹**（与 ``python.exe`` 同级；便携包用户）或 **项目根目录**（源码用户）。解压时选 **解压到当前文件夹**。

**config.rar**（便携包用户需要；源码用户 ``config\`` 已随仓库自带，可跳过）

- `百度网盘 <https://pan.baidu.com/s/1k1AVXRU6N0V8ggG1cZVtnQ?pwd=ziex>`_ ，提取码 **ziex**
- 解压后应出现 ``config\`` 文件夹

**demo_data.rar**（所有用户都需要）

- `百度网盘 <https://pan.baidu.com/s/1vDx6JZeM4Ay7VR1GAt7a-g?pwd=hkvq>`_ ，提取码 **hkvq**
- 解压后应出现 ``demo_data\`` 文件夹

演示数据已去标识，**仅供学术研究与 Demo，禁止商业用途**。

3. 打开命令提示符，进入工作文件夹并确认 ``habit`` 可用：

.. code-block:: bash

   cd /d D:\habit-cpu    # 示例路径，请改为本机工作文件夹
   habit --version

二、运行（5 步）
----------------

Demo 已含预处理结果，**首次建议从步骤 2 开始**。每步前请先 ``cd`` 到 **本机工作文件夹**（便携包根目录或源码项目根；示例见上文）。

**步骤 1 — 预处理**（可选，首次可跳过）

.. code-block:: bash

   habit preprocess --config config/preprocessing/config_preprocessing_demo.yaml

**步骤 2 — 生境分割**

.. code-block:: bash

   habit get-habitat --config config/habitat/config_habitat_two_step.yaml

**步骤 3 — 特征提取**

.. code-block:: bash

   habit extract --config config/feature_extraction/config_extract_features_demo.yaml

**步骤 4 — 机器学习（影像组学模型）**

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_radiomics.yaml --mode train

**步骤 5 — 临床模型 + 模型对比**

.. code-block:: bash

   habit model --config config/machine_learning/config_machine_learning_clinical.yaml --mode train
   habit compare --config config/model_comparison/config_model_comparison_demo.yaml

结果在 ``demo_data/results/`` 下（特征 CSV、ROC 图、对比图等）。修改参数见 :doc:`../configuration_zh` 。

下一步
------

- :doc:`../user_guide/index_zh` — 命令速查
- :doc:`../configuration_zh` — 配置说明
