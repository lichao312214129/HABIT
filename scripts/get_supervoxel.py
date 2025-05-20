"""
用于对每个肿瘤进行聚类，得到超像素，然后计算超像素的动态学特征

动态增强MRI的动态学特征包括如下：
1、wash_in_slope
    解释：动脉期的增强速率
2、wash_out_slope_of_lap_and_pvp
    解释：LAP和PVP的洗出速率
3、wash_out_slope_of_pvp_and_dp
    解释：PVP和DP的洗出速率
4、signal_enhancement_ratio_of_lap_and_pvp
    解释：LAP和PVP的信号增强比
5、signal_enhancement_ratio_of_pvp_and_dp
    解释：PVP和DP的信号增强比
6、signal_enhancement_ratio_of_lap_and_dp
    解释：LAP和DP的信号增强比
7、percentage_enhancement_of_lap
    解释：LAP的百分比增强
8、percentage_enhancement_of_pvp
    解释：PVP的百分比增强
9、percentage_enhancement_of_dp
    解释：DP的百分比增强
"""

import numpy as np
import tqdm
import SimpleITK as sitk
import time
import re
import os
import pandas as pd
import json
from datetime import datetime
import logging
from sklearn.cluster import KMeans
import concurrent.futures
import concurrent
# get mask and image
from habit.utils.io_utils import get_image_and_mask_paths

import warnings

from image2array import Image2Array

warnings.filterwarnings("ignore")


# load the configuration file
# config_file = r'E:\work\postdoc\Research\hccHeterogeneity\Version2\projectConfigurationFile.json'
# with open(config_file, 'r', encoding='utf-8') as file:
#     data = json.load(file)
NUMBEROFSUPERVOXEL = 50


class GetSuperVoxel:
    def __init__(self, image_root_path, n_clusters, out_folder=None):
        """
        Args:
            image_timestamp_file (str): The file path of the image timestamps of each phase.
            phase_name_file (str): The file path of the phase names.
            image_root_path (str): The root path of the images.
            n_clusters (int): The number of clusters for KMeans.

        Returns:
            None
        """

        self.image_timestamp_file = '../data/scan_time_of_phases.xlsx'  # 里面有Name，Pre_contrast，LAP，PVP，DP的扫描时间
        self.phase_name_file = '../data/final_patient_list.xlsx'  # phase 的名字，用于找原始图像
        self.image_root_path = image_root_path # 原始图像的根目录
        self.n_clusters = n_clusters  # 超像素的数量
        self.out_folder = out_folder  # 输出文件夹

        # log保持到本地
        logfile = "../data/kinetic_" + f"features{self.n_clusters}.log"
        logging.basicConfig(filename=logfile, level=logging.INFO)

    
    def load_timestamp(self):
        """
        Load the image timestamps from the given file.

        Returns:
            list: A list of image timestamps.
        """
        df = pd.read_excel(self.image_timestamp_file)
        
        # 把df变成dict，key是Name，value是一个列表，里面是Pre-contrast，LAP，PVP，DP的时间
        # 注意用re把Name中的第一个连续的数字提取出来

        df_dict = df.set_index('Name').T.to_dict('list')
        
        # 把key中的第一个连续的数字提取出来
        df_dict = {str(re.findall(r'\d+', k)[0]): v for k, v in df_dict.items()}

        return df_dict
   
    def calculation(self, image_array_list, image_timestamp_list):
        """
        Calculate kinetic features based on the given image array list and image timestamp list.

        Args:
            image_array_list (list): A list of image arrays.
            image_timestamp_list (list): A list of image timestamps.

        Returns:
            list: A list of calculated kinetic features, including wash in slope, wash out slope of LAP and PVP,
            wash out slope of PVP and DP, signal enhancement ratio of LAP and PVP, signal enhancement ratio of PVP and DP,
            signal enhancement ratio of LAP and DP, percentage enhancement of LAP, percentage enhancement of PVP,
            and percentage enhancement of DP.
        """

        # 断言，如果image_array_list和image_timestamp_list的长度不是4，就报错
        assert len(image_array_list) == 4, "The length of image array list should be 4."
        assert len(image_timestamp_list) == 4, "The length of image timestamp list should be 4."

        # NOTE: 这个函数我做了修改
        # 防止除0
        epsilon = 1e-4
        # 计算不同的时间
        time_format = "%H-%M-%S"

        # FIXME:修改1 ======================================================
        # FIXME: 我建议把只有EAP的数据删除，以简化工作量,那么下面的代码适用于LAP的情况
        # FIXME: 这里的pre-contrast和动脉期的时间差不应该是二者的插值，应该是固定值
        # FIXME 例如：pre-contrast和EAP是15秒，和LAP是25秒
        # image_timestamp_list = [datetime.strptime(time, time_format) for time in image_timestamp_list]
        image_timestamp_list = [datetime.strptime(time, time_format) for time in image_timestamp_list]
        # 把时间的第一个时间减去25秒
        image_timestamp_list[0] = image_timestamp_list[1] - pd.Timedelta(seconds=25)

        delta_t1 = (image_timestamp_list[1] - image_timestamp_list[0]).total_seconds()
        delta_t2 = (image_timestamp_list[2] - image_timestamp_list[1]).total_seconds()
        delta_t3 = (image_timestamp_list[3] - image_timestamp_list[2]).total_seconds()

        # 注射造影剂后，如果信号轻度小于基线，就把它设置为0，表示没有增强
        p1_p0 = (image_array_list[1] - image_array_list[0])
        p2_p0 = (image_array_list[2] - image_array_list[0])
        p3_p0 = (image_array_list[3] - image_array_list[0])
        p1_p0 = np.array([p1_p0_ if p1_p0_ > 0 else 0 for p1_p0_ in p1_p0])  # 保证p1_p0大于0
        p2_p0 = np.array([p2_p0_ if p2_p0_ > 0 else 0 for p2_p0_ in p2_p0])  # 保证p2_p0大于0
        p3_p0 = np.array([p3_p0_ if p3_p0_ > 0 else 0 for p3_p0_ in p3_p0])  # 保证p3_p0大于0

        # 第1个参数图：wash in slope
        wash_in_slope = p1_p0 / delta_t1

        # 第2个参数图：wash out slope of LAP and PVP
        # wash_out_slope = (image_array_list[2] - image_array_list[1]) / (delta_t3 - delta_t1)
        wash_out_slope_of_lap_and_pvp = (image_array_list[1] - image_array_list[2]) / (delta_t2)

        # wash out slope of PVP and DP
        wash_out_slope_of_pvp_and_dp = (image_array_list[2] - image_array_list[3]) / (delta_t3)

        # 第3个参数图：
        # signal_enhancement_ratio = (image_array_list[1] - image_array_list[0]) / (image_array_list[3] - image_array_list[0] + epsilon)
        signal_enhancement_ratio_of_lap_and_pvp = p1_p0 / (p2_p0 + epsilon)
        signal_enhancement_ratio_of_pvp_and_dp = p2_p0 / (p3_p0 + epsilon)
        signal_enhancement_ratio_of_lap_and_dp = p1_p0 / (p3_p0 + epsilon)

        # # check
        # ((p1_p0>=-0.1) & (p1_p0<=0.1)).sum()
        # ((p2_p0)==0).sum()
        # ((p3_p0)<0).sum()
        # ((image_array_list[2] - image_array_list[1])>0).sum()
        # ((image_array_list[3] - image_array_list[2])>0).sum()
        # # ((image_array_list[3] - image_array_list[1])>0).sum()
        # import matplotlib.pyplot as plt
        # plt.hist(signal_enhancement_ratio_of_lap_and_pvp)
        # plt.show(block=True)    


        # 第4个参数图：
        percentage_enhancement_of_lap = p1_p0 / (image_array_list[0] + epsilon)
        percentage_enhancement_of_pvp = (p2_p0) / (image_array_list[0] + epsilon)
        percentage_enhancement_of_dp = (p3_p0) / (image_array_list[0] + epsilon)

        # 返回这9个参数图
        return [wash_in_slope,
                wash_out_slope_of_lap_and_pvp,
                wash_out_slope_of_pvp_and_dp,
                signal_enhancement_ratio_of_lap_and_pvp,
                signal_enhancement_ratio_of_pvp_and_dp,
                signal_enhancement_ratio_of_lap_and_dp,
                percentage_enhancement_of_lap,
                percentage_enhancement_of_pvp,
                percentage_enhancement_of_dp
        ]

    def get_super_voxel(self, features):
        """
        用聚类的方法把图像分成超像素

        Args:
            features (array): A 2D array of features. x轴是特征，y轴是体素

        Returns:
            super voxel: A dictionary of super voxels, with the key as the label and the value as the mean feature.
            label: A list of labels for each feature.
        """
        # s = time.time()
        # 用kmeans把图像分成50个超像素
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, init='k-means++')
        kmeans.fit(features)
        label = kmeans.labels_

        # 每个label的数量
        unique, counts = np.unique(label, return_counts=True)

        # 将用一个label的feature取平均值
        super_voxel = np.zeros([self.n_clusters, features.shape[1]])
        for i in range(self.n_clusters):
            super_voxel[i,:] = np.mean(features[label == i], axis=0)
        # e = time.time()
        # print(f"Time of kmeans: {e-s}")
        return super_voxel, label

    def main(self):
        image2array = Image2Array(self.phase_name_file, self.image_root_path)
        # image2array.change_folder_and_file_name()
        image2array.get_file_path()
        file_paths = image2array.file_paths
        mask_paths = image2array.mask_paths
        time_df = self.load_timestamp()
        # 找到3者都有的key
        keys = list(set(file_paths.keys()) & set(mask_paths.keys()) & set(time_df.keys()))
        # 按照顺序排序keys
        keys.sort()
        
        # NOTE
        # keys = keys[:10]
        # print(keys)

        # calculate the kinetic features for each image
        super_voxel_dict = {}
        label_dict = {}
        for name in tqdm.tqdm(keys, desc="Calculating kinetic features"):
            print(name)

            try:
                # 如果file_paths或者mask_paths中没有这个key，就跳过
                if name not in mask_paths.keys():
                    continue
                if name not in file_paths.keys():
                    continue

                mask_path = mask_paths.get(name)[0]
                file_path = file_paths.get(name)

                # 检查file_path是不是都有文件
                for file in file_path:
                    if not os.path.exists(file):
                        print(f"{file}不存在")

                image_array, mask_array = image2array.img2array_(mask_path, file_path)

                features = self.calculation(image_array, time_df.get(name))

                features = np.array(features).T

                super_voxel, label = self.get_super_voxel(features)
                super_voxel_dict[name] = super_voxel
                label_dict[name] = label

            except Exception as e:
                # 保持错误信息到日志
                logging.error(f"Error: {name}")
                continue

        # save the super voxels and labels
        np.save(os.path.join(self.out_folder, f"super_voxel_{self.n_clusters}_supervoxels.npy"), super_voxel_dict)
        np.save(os.path.join(self.out_folder, f"label_{self.n_clusters}_supervoxels.npy"), label_dict)
        print("Done")

    def main_multi(self):
        file_paths, mask_paths = get_image_and_mask_paths(self.image_root_path)
        time_df = pd.read_excel(self.image_timestamp_file)
        time_df = time_df.set_index('Name').T.to_dict('list')

        
        # 找到3者都有的key
        keys = list(set(file_paths.keys()) & set(mask_paths.keys()) & set(time_df.keys()))
        keys.sort()


        super_voxel_dict = {}
        label_dict = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor: 
            s = time.time() 
            future_to_phase_name = {
                executor.submit(self.calculate_kinetic_features, mask_paths.get(name), file_paths.get(name), time_df.get(name), name) 
                                                        for name in keys
            }
            e = time.time()
            print(f"Time of submit: {e-s}")
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_phase_name), total=len(future_to_phase_name), desc="Calculating kinetic features"):
                try:
                    # s = time.time()
                    super_voxel, label, name = future.result()
                    super_voxel_dict[name] = super_voxel
                    label_dict[name] = label
                    # e = time.time()
                    # print(f"Time of 1 result: {e-s}")
                except Exception as e:
                    logging.error(f"Error calculating kinetic features for phase: {name}. Exception: {e}")

        # save the super voxels and labels
        np.save(os.path.join(self.out_folder, f"super_voxel_{self.n_clusters}_supervoxels.npy"), super_voxel_dict)
        np.save(os.path.join(self.out_folder, f"label_{self.n_clusters}_supervoxels.npy"), label_dict)
        print("Done")

    def calculate_kinetic_features(self, mask_path, file_paths, time, name):
        image_array = []
        # use sitk to read the image
        keys = ['pre_contrast', 'LAP', 'PVP', 'delay_3min']
        for file in keys:
            print(file)
            mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path[file]))
            array = sitk.GetArrayFromImage(sitk.ReadImage(file_paths[file]))
            array = array[mask_array == 1]
            image_array.append(array)

        features = self.calculation(image_array, time)
        features = np.array(features).T
        super_voxel, label = self.get_super_voxel(features)

        ia = np.array(image_array)
        ((ia[1] - ia[0])<0).sum()
        ((ia[2] - ia[1])<0).sum()
        ((ia[3] - ia[2])<0).sum()

        
        return super_voxel, label, name

if __name__ == "__main__":
    s = time.time()
    kf = GetSuperVoxel(image_root_path=r'H:\Registration_ICC_structured_test',
        n_clusters=NUMBEROFSUPERVOXEL, 
        out_folder=r'F:\work\research\radiomics_TLSs\data\results_365_test'
    )
    kf.main_multi()
    print(time.time() - s)