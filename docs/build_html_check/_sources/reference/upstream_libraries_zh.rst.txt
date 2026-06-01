上游依赖与文档链接
====================

本页汇总 HABIT 各模块依赖的 **第三方库官方文档** 与 **概念说明**。实现细节、全部超参数与数学定义以上游为准；HABIT 用户手册侧重 **配置键、CLI、数据契约**（参见 :doc:`../configuration_zh`）。

图像预处理（``habit preprocess``）
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 用途
     - 库 / 工具
     - 文档
   * - 配准（``backend: ants``）
     - ANTsPy（``ants.registration``）
     - `Registration  <https://antspy.readthedocs.io/en/stable/registration.html>`_ · `ants.registration API  <https://antspy.readthedocs.io/en/stable/api/ants.registration.html>`_
   * - 配准（``backend: elastix``）
     - 官方 **elastix** / **transformix** 可执行文件（命令行 ``-f`` / ``-m`` / ``-out`` / ``-p`` / ``-tp`` 等，见 `elastix 命令行说明  <https://elastix.dev/doxygen/commandlinearg.html>`_）
     - `官网 / 下载  <https://elastix.dev/>`_ · `Model Zoo（LKEB，按数据/任务说明与模板）  <https://lkeb.ml/modelzoo/>`_ · `Model Zoo（elastix.dev）  <https://elastix.dev/modelzoo.html>`_
   * - 配准（``backend: simpleitk``）
     - SimpleITK（``ImageRegistrationMethod``）
     - `ImageRegistrationMethod  <https://simpleitk.readthedocs.io/en/master/registrationOverview.html>`_ · `Fundamental Concepts  <https://simpleitk.readthedocs.io/en/master/FundamentalConcepts.html>`_
   * - ANTs 核心
     - ANTs
     - `ANTsX/ANTs  <https://github.com/ANTsX/ANTs>`_
   * - 重采样 / N4 / 多数滤波
     - SimpleITK
     - `User Guide  <https://simpleitk.readthedocs.io/>`_ · `Fundamental Concepts  <https://simpleitk.readthedocs.io/en/master/FundamentalConcepts.html>`_ · `ResampleImageFilter  <https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html>`_
   * - DICOM 转 NIfTI
     - dcm2niix
     - `rordenlab/dcm2niix  <https://github.com/rordenlab/dcm2niix>`_

生境分割（``habit get-habitat``）
----------------------------------

聚类与降维等步骤大量使用 **scikit-learn**；超像素 **SLIC** 等与实现所在类一致。

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 用途
     - 库
     - 文档
   * - K-Means / GMM / 流程与指标
     - scikit-learn
     - `Clustering  <https://scikit-learn.org/stable/modules/clustering.html>`_ · `Metrics  <https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation>`_
   * - 特征缩放、无监督预处理（与 sklearn 管线概念对应）
     - scikit-learn
     - `Preprocessing  <https://scikit-learn.org/stable/modules/preprocessing.html>`_

生境特征与传统影像组学（``habit extract`` / ``habit radiomics``）
----------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 用途
     - 库
     - 文档
   * - 影像组学特征
     - PyRadiomics
     - `PyRadiomics  <https://pyradiomics.readthedocs.io/>`_
   * - 图像读写 / 部分几何
     - SimpleITK
     - 同预处理节

机器学习（``habit model`` / ``habit cv``）
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 用途
     - 库
     - 文档
   * - 预处理、分类器、流水线、度量
     - scikit-learn
     - `User Guide  <https://scikit-learn.org/stable/user_guide.html>`_ · `API Reference  <https://scikit-learn.org/stable/modules/classes.html>`_
   * - 类别不平衡重采样（若启用）
     - imbalanced-learn（可选）
     - `imbalanced-learn  <https://imbalanced-learn.org/stable/>`_
   * - 梯度提升等模型（若配置）
     - XGBoost / LightGBM 等
     - `XGBoost Python  <https://xgboost.readthedocs.io/>`_（按需）
   * - AutoML（若配置）
     - AutoGluon
     - `AutoGluon Tabular  <https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html>`_
   * - SHAP 图（若启用）
     - shap
     - `SHAP  <https://shap.readthedocs.io/>`_

模型对比（``habit compare``）
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 用途
     - 库 / 方法
     - 文档
   * - ROC / PR / 校准等可视化
     - matplotlib / 自研绘图（见源码）
     - 医学论文常引用 **TRIPOD** / **CONSORT-AI** 等报告规范（非库文档）
   * - DeLong 检验（若启用）
     - 见实现与文献
     - 以 HABIT 输出说明与引用的统计学文献为准

ICC / Test–retest / 其它
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 用途
     - 库
     - 文档
   * - 统计与矩阵运算
     - NumPy / SciPy / pandas
     - `NumPy  <https://numpy.org/doc/>`_ · `SciPy  <https://docs.scipy.org/doc/scipy/>`_ · `pandas  <https://pandas.pydata.org/docs/>`_

ROI 勾画（外部工具）
--------------------

HABIT 不内置勾画功能，常见工具：

- `ITK-SNAP  <http://www.itksnap.org/>`_
- `3D Slicer  <https://www.slicer.org/>`_
