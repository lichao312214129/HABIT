告诉用户怎么训练融合模型
什么是radscore habitat score

docs自定义特征提取器：FeatureConstruction:
  voxel_level:
    method: my_feature_extractor(T1, T2)
    params:
      param1: value1
method有误

特征选择器是否也模块化，现在是函数

docs数据结构：每个文件夹中只能有一个文件，有错误，如果是dcm当然有多个，但dcm2nii后每个文件夹一个。如果多个habit只会选择第一个！

docs数据结构中：两种方式对比的表格显示有问题。还有一些换行等旧md格式问题，一并解决。