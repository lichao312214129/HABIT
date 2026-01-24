示例
========

示例 1: 基础 Habitat 分析
----------------------------

.. code-block:: python

   from habit.core.habitat_analysis import HabitatAnalysis
   
   config = {
       'data_dir': './data',
       'out_dir': './output',
       'FeatureConstruction': {
           'voxel_level': {
               'method': 'mean_voxel_features',
               'params': {}
           }
       },
       'HabitatsSegmention': {
           'clustering_mode': 'two_step',
           'supervoxel': {
               'algorithm': 'kmeans',
               'n_clusters': 50
           },
           'habitat': {
               'algorithm': 'kmeans',
               'max_clusters': 5
           }
       }
   }
   
   analysis = HabitatAnalysis(config)
   results = analysis.run()

示例 2: 自定义特征提取
------------------------

.. code-block:: python

   from habit.core.habitat_analysis.extractors import BaseFeatureExtractor
   
   class MyCustomExtractor(BaseFeatureExtractor):
       def extract(self, image, mask):
           # 自定义特征提取逻辑
           return features
   
   # 使用自定义提取器
   config['FeatureConstruction']['voxel_level']['method'] = 'my_custom'

示例 3: 使用 Pipeline
----------------------

.. code-block:: python

   from habit.core.habitat_analysis.pipeline import HabitatPipeline
   
   pipeline = HabitatPipeline([
       ('feature', FeatureExtractionStep(config)),
       ('preprocess', PreprocessingStep(config)),
       ('supervoxel', SupervoxelClusteringStep(n_clusters=50)),
       ('habitat', HabitatClusteringStep(n_clusters=3)),
   ])
   
   # 训练
   pipeline.fit(train_data)
   
   # 预测
   habitat_maps = pipeline.predict(test_data)
