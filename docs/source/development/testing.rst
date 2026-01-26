测试指南
========

HABIT 使用 pytest 进行测试。

运行测试
----------

运行所有测试：

.. code-block:: bash

   pytest tests/

运行特定测试文件：

.. code-block:: bash

   pytest tests/test_habitat.py

运行特定测试函数：

.. code-block:: bash

   pytest tests/test_habitat.py::test_feature_extraction

查看测试覆盖率：

.. code-block:: bash

   pytest --cov=habit --cov-report=html tests/

编写测试
----------

测试文件应该放在 ``tests/`` 目录下，并以 ``test_`` 开头。

示例：

.. code-block:: python

   import pytest
   import numpy as np
   from habit.core.habitat_analysis.extractors import MeanVoxelFeaturesExtractor
   
   def test_mean_voxel_extractor():
       # 准备测试数据
       image = np.random.rand(10, 10, 10)
       mask = np.ones((10, 10, 10))
       
       # 创建提取器
       extractor = MeanVoxelFeaturesExtractor()
       
       # 提取特征
       features = extractor.extract(image, mask)
       
       # 验证结果
       assert features.shape[0] == 1000  # 10*10*10 个体素
       assert features.shape[1] == 1  # 1 个特征

使用 Fixtures
--------------

.. code-block:: python

   @pytest.fixture
   def sample_image():
       return np.random.rand(10, 10, 10)
   
   @pytest.fixture
   def sample_mask():
       return np.ones((10, 10, 10))
   
   def test_with_fixtures(sample_image, sample_mask):
       extractor = MeanVoxelFeaturesExtractor()
       features = extractor.extract(sample_image, sample_mask)
       assert features is not None

测试最佳实践
------------

* 每个测试应该独立运行
* 使用描述性的测试名称
* 测试应该快速（< 1 秒）
* 使用 fixtures 来共享测试数据
* 测试应该覆盖正常情况和边界情况

持续集成
----------

HABIT 使用 GitHub Actions 进行持续集成。每次提交都会自动运行测试。

查看 CI 状态：

.. code-block:: bash

   # 在你的 PR 页面查看 Actions 标签
