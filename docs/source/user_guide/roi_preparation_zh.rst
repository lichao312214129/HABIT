ROI 文件准备
============

本节说明如何准备肿瘤 ROI（Region of Interest）掩码文件。

概述
----

HABIT 需要肿瘤 ROI 掩码来进行生境分析和特征提取。ROI 掩码指定了肿瘤的位置和范围，是生境分析的重要输入。

**重要说明：**

- **勾画不是 HABIT 的重点**: HABIT 专注于生境分析和特征提取，不提供 ROI 勾画功能
- **使用专业工具**: 推荐使用 ITK-SNAP、3D Slicer 等专业医学图像勾画工具
- **预处理和勾画可以互换**: 可以先进行预处理，也可以先勾画 ROI

ROI 文件格式要求
----------------

**支持的文件格式：**

- `.nii.gz`: 压缩的 NIfTI 格式（推荐）
- `.nii`: 未压缩的 NIfTI 格式
- `.nrrd`: Nearly Raw Raster Data 格式
- `.mha`: MetaImage 格式

**文件命名规则：**

- 推荐使用有意义的文件名，如 `mask_T1.nii.gz`、`mask_T2.nii.gz`
- 避免使用空格和特殊字符
- 建议使用英文命名，避免编码问题

**掩码值要求：**

- **二值掩码**: 推荐使用二值掩码（0 表示背景，1 表示肿瘤）
- **多标签掩码**: 也支持多标签掩码，但需要指定要处理的标签

**空间匹配：**

- 掩码必须与对应的图像在空间上匹配
- 掩码和图像应该具有相同的维度、间距和方向
- 如果不匹配，HABIT 会发出警告

勾画工具推荐
------------

**ITK-SNAP**

ITK-SNAP 是一个流行的医学图像分割工具，具有以下特点：

- **优点**:
  - 界面友好，易于使用
  - 支持多种图像格式
  - 提供自动分割算法
  - 支持手动编辑

- **下载地址**: http://www.itksnap.org/

- **基本使用步骤**:
  1. 加载医学图像
  2. 使用自动分割或手动勾画 ROI
  3. 编辑和优化 ROI
  4. 保存为 NIfTI 格式

**3D Slicer**

3D Slicer 是一个功能强大的医学图像处理平台，具有以下特点：

- **优点**:
  - 功能丰富，支持多种分析
  - 插件系统强大
  - 支持批量处理
  - 开源免费

- **下载地址**: https://www.slicer.org/

- **基本使用步骤**:
  1. 加载医学图像
  2. 使用 Segment Editor 模块勾画 ROI
  3. 编辑和优化 ROI
  4. 导出为 NIfTI 格式

**其他工具**

- **MITK**: http://www.mitk.org/
- **Horos**: https://www.horosproject.org/
- **RadiAnt DICOM Viewer**: https://www.radiantviewer.com/

ROI 勾画指南
------------

**勾画原则：**

1. **准确性**: 确保勾画的 ROI 准确反映肿瘤范围
2. **一致性**: 对所有受试者使用一致的勾画标准
3. **可重复性**: 确保勾画结果可重复
4. **排除伪影**: 避免包含伪影和噪声区域

**勾画步骤：**

1. **加载图像**: 在勾画工具中加载预处理后的图像
2. **确定范围**: 确定肿瘤的边界范围
3. **勾画 ROI**: 使用工具提供的勾画功能勾画 ROI
4. **编辑优化**: 编辑和优化 ROI，确保准确性
5. **保存文件**: 保存为支持的格式（推荐 NIfTI 格式）

**注意事项：**

- **避免包含正常组织**: 只勾画肿瘤区域，避免包含正常组织
- **处理边界**: 对于边界模糊的肿瘤，使用一致的标准
- **多时相图像**: 对于多时相图像，可以为每个时相勾画 ROI
- **质量检查**: 勾画完成后，检查 ROI 的质量

ROI 验证方法
------------

**视觉检查：**

1. 在医学图像查看器中同时加载图像和 ROI
2. 检查 ROI 是否准确覆盖肿瘤区域
3. 检查 ROI 是否包含不必要的区域
4. 检查 ROI 是否遗漏肿瘤区域

**统计检查：**

计算 ROI 的统计信息：

- **体积**: ROI 的总体积
- **形状**: ROI 的形状特征
- **强度分布**: ROI 内的图像强度分布

**一致性检查：**

- **受试者间一致性**: 检查不同受试者的 ROI 是否一致
- **勾画者间一致性**: 如果有多个勾画者，检查一致性
- **时间一致性**: 如果有多次扫描，检查 ROI 的时间一致性

**自动验证工具：**

HABIT 提供了一些验证工具：

- **Dice 系数**: 计算两个 ROI 之间的相似度
- **重叠率**: 计算 ROI 之间的重叠程度
- **边界检查**: 检查 ROI 是否超出图像边界

**使用 Dice 系数验证：**

.. code-block:: bash

   habit dice --input1 ./batch1 --input2 ./batch2 --output dice_results.csv

ROI 文件组织
------------

**文件夹结构：**

推荐按照以下结构组织 ROI 文件：

.. code-block:: text

   data_root/
   ├── images/
   │   ├── subject1/
   │   │   ├── T1/
   │   │   │   └── T1.nii.gz
   │   │   └── T2/
   │   │       └── T2.nii.gz
   └── masks/
       ├── subject1/
       │   ├── T1/
       │   │   └── mask_T1.nii.gz
       │   └── T2/
       │       └── mask_T2.nii.gz
       └── subject2/
           ├── T1/
           │   └── mask_T1.nii.gz
           └── T2/
               └── mask_T2.nii.gz

**YAML 配置：**

在 YAML 配置文件中指定 ROI 路径：

.. code-block:: yaml

   auto_select_first_file: true

   images:
     subject1:
       T1: ./data_root/images/subject1/T1/
       T2: ./data_root/images/subject1/T2/

   masks:
     subject1:
       T1: ./data_root/masks/subject1/T1/
       T2: ./data_root/masks/subject1/T2/

常见问题
--------

**Q1: ROI 和图像不匹配怎么办？**

A: 确保使用相同的预处理步骤处理图像和 ROI，或者使用配准工具将 ROI 配准到图像空间。

**Q2: 如何处理多标签 ROI？**

A: HABIT 支持多标签 ROI，但需要在配置文件中指定要处理的标签 ID。

**Q3: ROI 勾画太慢怎么办？**

A: 可以使用自动分割算法（如 ITK-SNAP 的自动分割）来加速勾画过程。

**Q4: 如何保证 ROI 的一致性？**

A: 制定详细的勾画指南，培训勾画者，定期检查勾画质量。

**Q5: ROI 文件太大怎么办？**

A: 可以使用压缩格式（如 `.nii.gz`）来减小文件大小。

下一步
-------

ROI 文件准备完成后，您可以：

- :doc:`image_preprocessing_zh`: 进行图像预处理
- :doc:`habitat_segmentation_zh`: 进行生境分割
- :doc:`../data_structure_zh`: 了解数据结构的详细说明
