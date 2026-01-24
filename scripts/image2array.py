"""
将患者图像转为数组
把ROI区域的图像转为数组

第一步：读取患者的excel表格，得到患者Name还有需要纳入研究的序列名称
第二步：根据Name和序列名称，找到对应的图像文件
第三步：读取图像文件，得到图像数组
第四步：找到该患者的ROI文件，读取ROI文件，得到ROI数组
"""


import os
import re
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import nrrd
import time

class Image2Array:
    def __init__(self, file, root_path):
        self.file = file
        self.root_path = root_path
        self.suffix = '.nrrd'
    
    def change_folder_and_file_name(self):
        # 定义一个函数，把root_path下的所有文件夹以及文件夹下面的文件名中含有2个连续以上下划线的地方变成一个下划线
        for subj in os.listdir(self.root_path):
            subj_new = re.sub(r'_{2,}', '_', subj)
            files_new = []
            for file in os.listdir(os.path.join(self.root_path, subj)):
                fn = re.sub(r'_{2,}', '_', file)
                fn = os.path.join(self.root_path, subj, fn)
                files_new.append(fn)
            files_old = os.listdir(os.path.join(self.root_path, subj))
            for file, file_new in zip(files_old, files_new):
                old_name = os.path.join(self.root_path, subj, file)
                new_name = file_new
                os.rename(old_name, new_name)
            os.rename(os.path.join(self.root_path, subj), os.path.join(self.root_path, subj_new))


        # 把文件名中带.nii.gz.nrrd的改为.nrrd
        for subj in os.listdir(self.root_path):
            for file in os.listdir(os.path.join(self.root_path, subj)):
                if '.nii.gz.nrrd' in file:
                    old_name = os.path.join(self.root_path, subj, file)
                    new_name = os.path.join(self.root_path, subj, file.replace('.nii.gz.nrrd', '.nrrd'))
                    os.rename(old_name, new_name)
        return self

    def get_file_path(self):
        """
        Reads an Excel file and extracts file paths and mask paths based on the data in the file.

        Returns:
            self: The current instance of the class.
        """
        data = pd.read_excel(self.file)
        name = data['Name']
        series = data[['Pre_contrast','LAP','PVP','DP']]
        mask_path = {}
        file_path = {}
        for i in range(len(name)):
            #判断文件夹是否存在
            if not os.path.exists(os.path.join(self.root_path, name[i])):
                # print(f'Folder not exist: {name[i]}')
                continue
            # Find mask, 文件名中含有mask的文件
            files_ = os.listdir(os.path.join(self.root_path, name[i]))
            mask_ = [file for file in files_ if 'mask' in file]
            mask_path[name[i]] = [os.path.join(self.root_path, name[i], mask) for mask in mask_]

            file_path_ = []
            for j in range(len(series.columns)):
                file_path_.append(os.path.join(self.root_path, name[i], series.iloc[i,j]+self.suffix))
            file_path[name[i]] = file_path_

        # 把file_path中的字符串中所有连着2个以上的下划线变成一个下划线
        # 用re的正则表达式
        for key in file_path.keys():
            file_path[key] = [re.sub(r'_{2,}', '_', file) for file in file_path[key]]

        self.file_paths = file_path
        self.mask_paths = mask_path

        # 用re把key的第一个连续数字提取并替换旧的key
        
        self.file_paths = {str(re.findall(r'\d+', k)[0]): v for k, v in self.file_paths.items()}
        self.mask_paths = {str(re.findall(r'\d+', k)[0]): v for k, v in self.mask_paths.items()}

        return self

    def check_if_file_exist(self):
        # 先检查每个file_path是否有4个文件
        for i in self.file_paths.keys():
            if len(self.file_paths[i]) != 4:
                print(f'File number is not 4: {i}')

        # 再检查mask_path是否有1个文件
        for i in self.mask_paths.keys():
            if len(self.mask_paths[i]) != 1:
                print(f'Mask number is not 1: {i}')

        file_not_exist = []
        for i in self.file_paths.keys():
            for j in self.file_paths[i]:
                if not os.path.exists(j):
                    file_not_exist.append(j)

        mask_not_exist = []
        for i in self.mask_paths.keys():
            for j in self.mask_paths[i]:
                if not os.path.exists(j):
                    mask_not_exist.append(j)

        self.file_not_exist = file_not_exist
        self.mask_not_exist = mask_not_exist
        return self
    

    def img2array_(self, mask_path, file_paths):
        """Reads image files within the ROI and returns the image arrays.

        Args:
            mask_path (str): The path to the mask file.
            file_paths (list): A list of paths to the image files.

        Returns:
            tuple: A tuple containing the image arrays and the mask array.
        """
        # s = time.time()
        mask_array, header = nrrd.read(mask_path)

        image_array = []
        for path in file_paths:
            img_array_, header = nrrd.read(path)

            # 检查mask和image的维度是否一致，如不一致则跳过
            if mask_array.shape != img_array_.shape:
                print(f'Image and mask dimensions do not match: {path}')
                continue

            img_array_in_mask = img_array_[mask_array != 0]  # Only take images within the mask region
            image_array.append(img_array_in_mask) 
        # print(f'Reading image array takes {time.time()-s} seconds')
        return image_array, mask_array

    def img2array(self):
        
        image_array = {}
        mask_array = {}
        for key in self.file_paths.keys():
            image_array[key] = self.img2array_(self.mask_paths[key][0], self.file_paths[key])
            mask_array[key], header = nrrd.read(self.mask_paths[key][0])

        # 用re把image_array和mask_array的key的第一个连续数字提取并替换旧的key
        for key in image_array.keys():
            new_key = re.sub(r'\d{1,}', '', key)
            image_array[new_key] = image_array.pop(key)
            mask_array[new_key] = mask_array.pop(key)
            
        return image_array, mask_array
    
    def _test_():
        dd = np.array(image_array[key])
        dd = dd.T
        # 导入kmeans
        from sklearn.cluster import KMeans
        # 聚类, kmeans++自动选择初始点
        kmeans = KMeans(n_clusters=10, random_state=0, init='k-means++')
        kmeans.fit(dd)
        label = kmeans.predict(dd)


if __name__ == '__main__':
    file = '../data/final_patient_list.xlsx'
    root_path = r'G:\DMS\肿瘤内异质性\Registration_ZSSY'
    image2array = Image2Array(file, root_path)
    # image2array.change_folder_and_file_name()
    image2array.get_file_path()
    image2array.check_if_file_exist()
    # image2array.img2array()