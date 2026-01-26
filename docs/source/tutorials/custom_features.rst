自定义特征提取
================

本教程将教你如何创建自定义特征提取器。

创建自定义提取器
----------------

继承 `BaseFeatureExtractor` 类：

.. code-block:: python

   from habit.core.habitat_analysis.extractors import BaseFeatureExtractor
   import numpy as np
   
   class MyCustomExtractor(BaseFeatureExtractor):
       def __init__(self, params=None):
           super().__init__(params)
           self.param1 = params.get('param1', 1.0)
       
       def extract(self, image, mask):
           """
           提取自定义特征
           
           Args:
               image: numpy array, shape (H, W, D) or (H, W, D, T)
               mask: numpy array, shape (H, W, D)
           
           Returns:
               features: numpy array, shape (N_voxels, N_features)
           """
           # 只在 mask 区域提取特征
           masked_image = image[mask > 0]
           
           # 示例：计算简单的统计特征
           features = np.column_stack([
               masked_image,  # 原始强度
               np.mean(masked_image) * np.ones_like(masked_image),  # 均值
               np.std(masked_image) * np.ones_like(masked_image),   # 标准差
           ])
           
           return features

注册自定义提取器
----------------

在配置文件中使用你的自定义提取器：

.. code-block:: yaml

   FeatureConstruction:
     voxel_level:
       method: my_custom
       params:
         param1: 2.5

高级示例：纹理特征
------------------

使用 skimage 计算纹理特征：

.. code-block:: python

   from skimage.feature import graycomatrix, graycoprops
   
   class TextureExtractor(BaseFeatureExtractor):
       def extract(self, image, mask):
           features = []
           
           # 对每个体素计算纹理特征
           for i in range(image.shape[0]):
               for j in range(image.shape[1]):
                   for k in range(image.shape[2]):
                       if mask[i, j, k] > 0:
                           # 提取局部纹理
                           patch = self._get_patch(image, i, j, k, size=5)
                           glcm = graycomatrix(patch, distances=[1], angles=[0])
                           props = graycoprops(glcm)
                           
                           # 收集特征
                           feat = [
                               props['contrast'],
                               props['dissimilarity'],
                               props['homogeneity'],
                               props['ASM'],
                               props['energy'],
                           ]
                           features.append(feat)
           
           return np.array(features)
       
       def _get_patch(self, image, i, j, k, size):
           # 提取局部 patch
           pad = size // 2
           padded = np.pad(image, pad, mode='reflect')
           return padded[i:i+size, j:j+size, k:k+size]

下一步
--------

* 查看 `基础教程 <basic_habitat_analysis.html>`_
* 了解 `机器学习集成 <ml_integration.html>`_
