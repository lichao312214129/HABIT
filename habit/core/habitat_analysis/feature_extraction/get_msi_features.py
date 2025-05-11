# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
此代码用来计算subregion的特征
"""

import os
import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk
# 多进程
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor



class GetMsiFeatures:

    def __init__(self, subregion_dir, n_region, out_dir):
        self.subregion_dir = subregion_dir
        self.out_dir = out_dir
        self.n_region = n_region
        self.voxel_cutoff = 10

    def calculate_MSI_matrix(self, habitat_array, unique_class=5):
        """计算生境矩阵的对应的MSI矩阵
        REF1:Intratumoral Spatial Heterogeneity at Perfusion MR Imaging Predicts Recurrence-free Survival in Locally Advanced Breast CancerTreated with Neoadjuvant Chemotherapy
        REF2:Habitat Imaging Biomarkers for Diagnosis and Prognosis in Cancer Patients Infected with COVID-19 
        REF3：Enhancing NSCLC recurrence prediction with PET/CT habitat imaging, ctDNA, and integrative radiogenomics-blood insights
        Args:
            habitat_array: 生境的array
            unique_class: 生境的类别数量
        Returns:
            msi_matrix: 计算出来的MSI矩阵

        # Example:
                # 假设box_of_VOI是一个numpy数组，代表3D肿瘤图像。其是从superpixel_array的肿瘤ROI中提取的最小外接矩形box
                np.random.seed(0)
                box_of_VOI = np.random.randint(1, 4, (3, 3, 3))
                # 周围加一圈0，保证器官背景和器官之间的跨区域信息，代表某个亚区（habitats和器官的边界长度）
                box_of_VOI = np.pad(box_of_VOI, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)

                # 忽略角连接的邻居，只考虑立方体面连接的邻居
                neighborhood_3d_cube_only = [
                (-1, 0, 0), (1, 0, 0),  # 上下邻居
                (0, -1, 0), (0, 1, 0),  # 左右邻居
                (0, 0, -1), (0, 0, 1)   # 前后邻居
                ]
        """

        # 把团块内体素小于10的团块去掉,图像中有多个label，每个label代表一个生境
        # 假设 habitat_array 是已标记好的二维或三维图像数组，其中每个连通区域有不同的正整数标签
        # 遍历每一个标签，获取每一个标签的形状属性
        # size_json = {k : 0 for k in np.arange(1, unique_class)}
        # for label in np.arange(1, unique_class):
        #     # 获取其中一个标签的mask
        #     labeled_data = (habitat_array == label).astype(np.uint16)
        #     labeled_img = sitk.GetImageFromArray(labeled_data)

        #     # ConnectedComponentImageFilter类，用于连通域划分
        #     cc_filter = sitk.ConnectedComponentImageFilter()
        #     cc_filter.SetFullyConnected(True)
        #     output_mask = cc_filter.Execute(labeled_img)  # 执行连通域划分，输出为sitk.Image格式

        #     mask_array = sitk.GetArrayFromImage(output_mask)
        #     np.unique(habitat_array[mask_array!=0])
        #     np.unique(mask_array)

        #     # 给output_mask中的不相邻的label重新标记，比如有2个不为0的子区域，但是他们不相邻，那么他们的label就是1和2

        #     # LabelShapeStatisticsImageFilter类，用于获取各标签形状属性
        #     lss_filter = sitk.LabelShapeStatisticsImageFilter()
        #     lss_filter.Execute(output_mask)
        #     num_regions = lss_filter.GetNumberOfLabels()
        #     num_connected_label = cc_filter.GetObjectCount()
        #     size = [lss_filter.GetNumberOfPixels(i) for i in range(1, num_connected_label + 1)]
        #     size_json_ = {label: size}
        #     size_json.update(size_json_)
        
        # # 去掉体素小于10的团块


        # 找到superpixel_array中不为0的区域的最小外接矩形
        roi_z, roi_y, roi_x = np.where(habitat_array != 0)
        z_min, z_max = np.min(roi_z), np.max(roi_z)
        y_min, y_max = np.min(roi_y), np.max(roi_y)
        x_min, x_max = np.min(roi_x), np.max(roi_x)

        # 取bounding_box的数据
        box_of_VOI = habitat_array[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        
        # 周围加一圈0，保证器官背景和器官之间的跨区域信息，代表某个亚区（habitats和器官的边界长度）
        box_of_VOI = np.pad(box_of_VOI, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)

        neighborhood_3d_cube_only = [
        (-1, 0, 0), (1, 0, 0),  # 上下邻居
        (0, -1, 0), (0, 1, 0),  # 左右邻居
        (0, 0, -1), (0, 0, 1)   # 前后邻居
        ]

        # 在遍历3D图像时，使用这些偏移量访问当前体素的邻居
        msi_matrix = np.zeros((unique_class, unique_class), dtype=np.int64)
        for z in range(box_of_VOI.shape[0]):  
            for y in range(box_of_VOI.shape[1]):  
                for x in range(box_of_VOI.shape[2]): 

                    # 获取当前体素的值
                    current_voxel_value = box_of_VOI[z, y, x]

                    # 遍历当前体素的邻居
                    for dz, dy, dx in neighborhood_3d_cube_only:
                        neighbor_z = z + dz
                        neighbor_y = y + dy
                        neighbor_x = x + dx
                        
                        # 检查邻居是否在图像范围内
                        if 0 <= neighbor_z < box_of_VOI.shape[0] and \
                        0 <= neighbor_y < box_of_VOI.shape[1] and \
                        0 <= neighbor_x < box_of_VOI.shape[2]:
                            
                            neighbor_voxel_value = box_of_VOI[neighbor_z, neighbor_y, neighbor_x]
                            
                            # 更新MSI矩阵
                            msi_matrix[current_voxel_value, neighbor_voxel_value] += 1

        # 返回计算出来的矩阵
        return msi_matrix

    def calculate_MSI_feature(self, msi_matrix, name):
        """根据提供的msi_matrix计算对应的msi_feature  1861315"""
        # 断言msi_matrix是一个方阵且没有任何的负数
        assert msi_matrix.shape[0] == msi_matrix.shape[1], f'msi_matrix of {name} is not a square matrix'
        assert np.all(msi_matrix >= 0), f'msi_matrix of {name} has negative value'
        
        # 一阶特征： Volume of each subregion (diagonal) and borders of two differing subregions (off-diagonal)
        firstorder_feature = {}
        for i in range(0, msi_matrix.shape[0]):
            for j in range(i+1, msi_matrix.shape[0]):
                firstorder_feature['firstorder_{}_and_{}'.format(i, j)] = msi_matrix[i, j]

        # 计算对角线上的元素，但不计算背景的
        for i in range(1, msi_matrix.shape[0]):
            firstorder_feature['firstorder_{}_and_{}'.format(i, i)] = msi_matrix[i, i]

        # 归一化的一阶特征，分母应该只算纳入到计算的部分,既下三角部分，但不包括第一个元素
        denominator_mat = np.tril(msi_matrix, k=0)
        denominator_mat[0] = 0
        denominator = np.sum(denominator_mat)
        normal_msi_matrix = msi_matrix / denominator
        firstorder_feature_normalized = {}
        for i in range(0, normal_msi_matrix.shape[0]):
            for j in range(i+1, normal_msi_matrix.shape[1]):
                firstorder_feature_normalized['firstorder_normalized_{}_and_{}'.format(i, j)] = normal_msi_matrix[i, j]

        for i in range(1, normal_msi_matrix.shape[0]):
            firstorder_feature_normalized['firstorder_normalized_{}_and_{}'.format(i, i)] = normal_msi_matrix[i, i]
        
        # 第三类特征：二阶特征，基于归一化的MSI矩阵进行计算
        # 使用归一化后的MSI矩阵计算二阶特征
        p = normal_msi_matrix.copy()
        
        # 计算contrast (对比度)
        # contrast = sum_{i,j} |i-j|^2 * p(i,j)
        i_indices, j_indices = np.indices(p.shape)
        contrast = np.sum((i_indices - j_indices)**2 * p)
        
        # 计算homogeneity (同质性)
        # homogeneity = sum_{i,j} p(i,j) / (1 + |i-j|^2)
        homogeneity = np.sum(p / (1.0 + (i_indices - j_indices)**2))
        
        # 计算correlation (相关性)：correlation = (sum_{i,j} p(i,j)*i*j - ux*uy) / (sigmax*sigmay)
        # 首先计算边缘分布
        px = np.sum(p, axis=1)
        py = np.sum(p, axis=0)
        
        # 计算均值
        ux = np.sum(px * np.arange(len(px)))
        uy = np.sum(py * np.arange(len(py)))
        
        # 计算标准差
        sigmax = np.sqrt(np.sum(px * (np.arange(len(px)) - ux)**2))
        sigmay = np.sqrt(np.sum(py * (np.arange(len(py)) - uy)**2))
        
        # 计算相关性
        if sigmax > 0 and sigmay > 0:
            # 计算 sum_{i,j} p(i,j)*i*j
            sum_p_ij = np.sum(p * i_indices * j_indices)
            correlation = (sum_p_ij - ux * uy) / (sigmax * sigmay)
        else:
            # 如果是flat region (只有一个灰度值)，则返回1
            correlation = 1.0
        
        # 计算energy (能量)
        # energy = sum_{i,j} p(i,j)^2
        energy = np.sum(p**2)
        
        secondorder_feature = { 
            'contrast': contrast,
            'homogeneity': homogeneity,
            'correlation': correlation,
            'energy': energy
        }

        # 将三类特征拼接起来
        msi_feature = {**firstorder_feature, **firstorder_feature_normalized, **secondorder_feature}

        # 将结果返回出去
        return msi_feature

    def main(self):
        # 获取所有的subregion文件
        subregion_files = os.listdir(self.subregion_dir)
        subregion_files = [os.path.join(self.subregion_dir, file) for file in subregion_files]

        # 使用多进程进行处理
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.process_subregion_file, subregion_file) for subregion_file in subregion_files]

            # 使用as_completed来处理每个文件的结果
            features_dict = {}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc='Processing subregion files'):
                msi_feature = future.result()
                features_dict.update(msi_feature)
        
        # 将结果保存成pd.DataFrame
        subregion_features_df = pd.DataFrame(features_dict).T

        # 保存结果
        out_file = os.path.join(self.out_dir, 'msi_features_df.csv')
        subregion_features_df.to_csv(out_file, index=True)

    def process_subregion_file(self, subregion_file):
        img = sitk.ReadImage(subregion_file)
        array = sitk.GetArrayFromImage(img)
        name = os.path.basename(subregion_file).split('.')[0]
        calcu_MSI_matrix = self.calculate_MSI_matrix(array, unique_class=self.n_region)
        msi_feature = self.calculate_MSI_feature(calcu_MSI_matrix, name)
        
        return {name: msi_feature}

if __name__ == '__main__':
    subregion_dir = r'F:\work\research\radiomics_TLSs\data\results_test'
    out_dir = r'F:\work\workstation_b\dingHuYingXiang\the_third_training_202504\demo_data\results'
    n_region = 5 + 1
    gsf = GetMsiFeatures(subregion_dir, n_region, out_dir)
    gsf.main()



