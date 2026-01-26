Dice 系数计算器使用指南
========================

``habit dice`` 模块用于计算两批图像（通常是 ROI 或 Mask）之间的 Dice 系数。这对于评估分割算法的准确性或比较不同专家的标注一致性非常有用。

功能特点
--------

* **批量计算**：自动匹配两个目录或配置文件中的受试者和 Mask 类型。
* **灵活输入**：支持直接指定目录路径或使用 YAML 配置文件。
* **自动匹配**：基于受试者 ID 和 Mask 子文件夹名称自动对齐文件。
* **详细报告**：生成包含每个样本详细 Dice 值的 CSV 文件，并输出统计摘要（均值、标准差等）。

命令行用法
----------

基本语法：

.. code-block:: bash

   habit dice [OPTIONS]

选项说明
--------

* ``--input1``: **(必需)** 第一批数据的路径。可以是包含 ``masks`` 子文件夹的根目录，也可以是 YAML 配置文件。
* ``--input2``: **(必需)** 第二批数据的路径。可以是包含 ``masks`` 子文件夹的根目录，也可以是 YAML 配置文件。
* ``--output``: 结果 CSV 文件的保存路径。默认为 ``dice_results.csv``。
* ``--mask-keyword``: Mask 文件夹的关键字（仅当输入为目录时使用）。默认为 ``masks``。
* ``--label-id``: 计算 Dice 时使用的标签 ID。默认为 ``1``。

使用示例
--------

1. 比较两个文件夹
^^^^^^^^^^^^^^^^^

假设你有两批标注数据，分别存储在 ``data/batch1`` 和 ``data/batch2`` 中。每个目录下都有 ``masks`` 子文件夹，结构如下：

.. code-block:: text

   data/batch1/masks/
     ├── subject001/
     │   └── tumor/
     │       └── mask.nii.gz
     └── ...

运行以下命令进行比较：

.. code-block:: bash

   habit dice --input1 data/batch1 --input2 data/batch2 --output comparison_results.csv

2. 使用配置文件
^^^^^^^^^^^^^^^^

如果你已经为数据创建了 YAML 配置文件（例如用于其他 HABIT 模块），也可以直接使用：

.. code-block:: bash

   habit dice --input1 config/dataset_A.yaml --input2 config/dataset_B.yaml

**YAML 配置文件格式示例：**

.. code-block:: yaml

   masks:
     subject001:
       t1: /path/to/dataset_A/masks/subject001/t1/mask.nii.gz
     subject002:
       t1: /path/to/dataset_A/masks/subject002/t1/mask.nii.gz
   # images 字段对于 dice 计算是可选的，但如果复用其他模块配置可能会包含
   # 注意：masks 下的键名（如 t1）应与 images 下的序列名称对应
   images:
     subject001:
       t1: /path/to/dataset_A/images/subject001/t1/image.nii.gz

3. 指定 Label ID
^^^^^^^^^^^^^^^^

如果你的 Mask 文件中肿瘤的标签值不是 1（例如是 255），可以指定：

.. code-block:: bash

   habit dice --input1 dir1 --input2 dir2 --label-id 255

输出结果
--------

程序将在控制台打印进度和统计摘要，并生成 CSV 文件。

**控制台输出示例：**

.. code-block:: text

   Loading paths from dir1...
   Loading paths from dir2...
   Found 10 subjects in batch 1.
   Found 12 subjects in batch 2.
   Found 10 common subjects to compare.
   Calculating Dice  [####################################]  100%

   Results saved to dice_results.csv
   Mean Dice: 0.8543
   Std Dice: 0.0521
   Min Dice: 0.7632
   Max Dice: 0.9123

**CSV 文件内容示例：**

+-----------+-----------+-------+---------------------------------------------+---------------------------------------------+
| Subject   | MaskType  | Dice  | Path1                                       | Path2                                       |
+===========+===========+=======+=============================================+=============================================+
| subject001| tumor     | 0.88  | dir1/masks/subject001/tumor/mask.nii.gz      | dir2/masks/subject001/tumor/mask.nii.gz      |
+-----------+-----------+-------+---------------------------------------------+---------------------------------------------+
| subject002| tumor     | 0.85  | dir1/masks/subject002/tumor/mask.nii.gz      | dir2/masks/subject002/tumor/mask.nii.gz      |
+-----------+-----------+-------+---------------------------------------------+---------------------------------------------+
| ...       | ...       | ...   | ...                                         | ...                                         |
+-----------+-----------+-------+---------------------------------------------+---------------------------------------------+