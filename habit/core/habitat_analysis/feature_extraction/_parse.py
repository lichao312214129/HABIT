"""
This code is used to parse habitat features to CSV files.
Currently supports:
1. Conventional radiomics features of the entire ROI
2. Non-radiomics features of habitats, such as number of habitats, volume proportion of each habitat, number of habitat clusters
3. Radiomics features of the entire habitat map (ROI=whole tumor, image=entire habitat map)
4. Radiomics features of each tumor subregion (ROI=each tumor subregion, image=original image [e.g., CT, MRI])
"""

import numpy as np
import pandas as pd
import os
import argparse
import sys
import logging
from typing import Dict, List, Optional, Union
from utils import flatten_dict, save_to_excel_sheet

# 日志配置会在类初始化时设置，因为需要使用输入文件的路径
logger = logging.getLogger(__name__)


class HabitatFeatureParse:
    """
    Extract features from habitat maps
    """
    
    def __init__(self, file: str, out_dir: str) -> None:
        """
        Initialize the habitat feature extractor.

        Args:
            file (str): The data to be processed, npy format.
            out_dir (str): Output directory for saving results.
        """
        self.file = file
        self.out_dir = out_dir

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        # 设置日志 - 保存到输入文件的同级目录
        file_dir = os.path.dirname(os.path.abspath(file))
        log_file = os.path.join(file_dir, 'habitat_features_parse.log')
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        
        logger.info(f"Log file will be saved to: {log_file}")
        logger.info(f"Created output directory: {self.out_dir}")

        self.read_data()

    def read_data(self) -> 'HabitatFeatureParse':
        """
        Read the npy data file.

        Returns:
            HabitatFeatureParse: Self instance for method chaining.
        """
        try:
            self.data = np.load(self.file, allow_pickle=True).item()
            logger.info(f"Successfully loaded data from {self.file}")
        except Exception as e:
            logger.error(f"Error loading data from {self.file}: {str(e)}")
            raise
        return self

    def extract_traditional_radiomics(self, key: str = 'tranditional_radiomics_features') -> pd.DataFrame:
        """
        Extract traditional radiomics features from the habitat map.

        Args:
            key (str): Key for accessing traditional radiomics features.

        Returns:
            pd.DataFrame: Extracted traditional radiomics features.
        """
        logger.info("Starting extraction of traditional radiomics features")
        subjs = list(self.data.keys())
        tranditional_radiomics = []
        total = len(subjs)
        
        for i, subj in enumerate(subjs):
            progress = int((i + 1) / total * 50)
            bar = "█" * progress + "-" * (50 - progress)
            percent = (i + 1) / total * 100
            print(f"\r处理进度: [{bar}] {percent:.2f}% ({i+1}/{total})", end="")

            imgs = list(self.data.get(subj).get(key).keys())
            dfs = pd.DataFrame([self.data.get(subj).get(key).get(img) for img in imgs], index=imgs)
            dfs = dfs.loc[:, ~dfs.columns.str.contains('diagnostic')]
            new_columns = [f"{col}_of_{idx}" for idx in dfs.index for col in dfs.columns]
            dfs_reshaped = pd.DataFrame([dfs.values.flatten()], columns=new_columns)
            tranditional_radiomics.append(dfs_reshaped)
        
        print()  # 完成后换行
        tranditional_radiomics = pd.concat(tranditional_radiomics)
        tranditional_radiomics.index = subjs
        
        out_file = os.path.join(self.out_dir, "tranditional_radiomics.csv")
        tranditional_radiomics.to_csv(out_file, index=True)
        logger.info(f"Saved traditional radiomics features to {out_file}")
        return tranditional_radiomics

    def extract_non_radiomics_features(self, key: str = 'non_radiomics_features', n_habitats: int = 5) -> pd.DataFrame:
        """
        Extract non-radiomics features from the habitat map.

        Args:
            key (str): Key for accessing non-radiomics features.
            n_habitats (int): Number of habitats to process.

        Returns:
            pd.DataFrame: Extracted non-radiomics features.
        """
        logger.info("Starting extraction of non-radiomics features")
        subjs = list(self.data.keys())
        n1 = [f"num_regions_{i}" for i in range(1, n_habitats+1)]
        n2 = [f"volume_ratio_{i}" for i in range(1, n_habitats+1)]
        non_radiomics_features = pd.DataFrame(index=subjs, columns=['num_habitats']+n1+n2)
        total = len(subjs)
        
        for i, subj in enumerate(subjs):
            progress = int((i + 1) / total * 50)
            bar = "█" * progress + "-" * (50 - progress)
            percent = (i + 1) / total * 100
            print(f"\r处理进度: [{bar}] {percent:.2f}% ({i+1}/{total})", end="")
            
            for fn in non_radiomics_features.columns:
                fd = flatten_dict(self.data.get(subj).get(key))
                non_radiomics_features.loc[subj, fn] = fd.get(fn)

        print()  # 完成后换行
        out_file = os.path.join(self.out_dir, "non_radiomics_features.csv")
        non_radiomics_features.to_csv(out_file, index=True)
        logger.info(f"Saved non-radiomics features to {out_file}")
        return non_radiomics_features

    def extract_radiomics_features_for_whole_habitat_map(self, key: str = 'radiomics_features_of_whole_habitat_map') -> pd.DataFrame:
        """
        Extract radiomics features for the whole habitat map.

        Args:
            key (str): Key for accessing whole habitat map features.

        Returns:
            pd.DataFrame: Extracted whole habitat map features.
        """
        logger.info("Starting extraction of whole habitat map features")
        subjs = list(self.data.keys())
        radiomics_of_whole_habitat = []
        total = len(subjs)
        
        for i, subj in enumerate(subjs):
            progress = int((i + 1) / total * 50)
            bar = "█" * progress + "-" * (50 - progress)
            percent = (i + 1) / total * 100
            print(f"\r处理进度: [{bar}] {percent:.2f}% ({i+1}/{total})", end="")

            features = self.data.get(subj).get(key)
            features_df = pd.DataFrame.from_dict(features, orient='index').T
            features_df = features_df.loc[:, ~features_df.columns.str.contains('diagnostic')]
            radiomics_of_whole_habitat.append(features_df)
        
        print()  # 完成后换行
        radiomics_of_whole_habitat = pd.concat(radiomics_of_whole_habitat)
        radiomics_of_whole_habitat.index = subjs

        out_file = os.path.join(self.out_dir, "radiomics_of_whole_habitat.csv")
        radiomics_of_whole_habitat.to_csv(out_file, index=True)
        logger.info(f"Saved whole habitat map features to {out_file}")
        return radiomics_of_whole_habitat

    def extract_radiomics_features_from_each_habitat(self, key: str = 'radiomics_features_from_each_habitat', n_habitats: int = 5) -> Dict[int, pd.DataFrame]:
        """
        Extract radiomics features from each tumor subregion.

        Args:
            key (str): Key for accessing individual tumor subregion features.
            n_habitats (int): Number of tumor subregions to process.

        Returns:
            Dict[int, pd.DataFrame]: Dictionary of extracted features for each tumor subregion.
        """
        logger.info("Starting extraction of tumor subregion features")
        subjs = list(self.data.keys())
        radiomics_of_each_subregion = {i+1: [] for i in range(n_habitats)}
        subregion_count = np.zeros((len(subjs), n_habitats))
        subregion_count = pd.DataFrame(subregion_count, index=subjs, columns=[np.arange(1, n_habitats+1)])

        total = len(radiomics_of_each_subregion)
        
        for subregion_id in radiomics_of_each_subregion.keys():
            progress = int((subregion_id) / total * 50)
            bar = "█" * progress + "-" * (50 - progress)
            percent = (subregion_id) / total * 100
            print(f"\r处理子区域: [{bar}] {percent:.2f}% ({subregion_id}/{total})", end="")

            for i, subj in enumerate(subjs):
                if i == 0: 
                    imgs = list(self.data.get(subj).get(key).keys())
                radiomics_of_subregion = [] 
                for iimg, img in enumerate(imgs):
                    if (subregion_id == 1) & (iimg == 0): 
                        col = list(self.data.get(subj).get(key).get(img).keys())
                        subregion_count.loc[subj, col] = 1

                    feature = self.data.get(subj).get(key).get(img).get(subregion_id)
                    if feature is not None: 
                        df = pd.DataFrame.from_dict(feature, orient='index').T
                        radiomics_of_subregion.append(df)  

                if feature is not None:   
                    radiomics_of_subregion = pd.concat(radiomics_of_subregion)
                    radiomics_of_subregion.index = imgs
                    radiomics_of_subregion = radiomics_of_subregion.loc[:, ~radiomics_of_subregion.columns.str.contains('diagnostic')]
                    new_columns = [f"{col}_of_{idx}" for idx in radiomics_of_subregion.index for col in radiomics_of_subregion.columns]
                    radiomics_of_subregion = pd.DataFrame([radiomics_of_subregion.values.flatten()], columns=new_columns, index=[subj])
                    radiomics_of_each_subregion[subregion_id].append(radiomics_of_subregion)
            
            radiomics_of_each_subregion[subregion_id] = pd.concat(radiomics_of_each_subregion[subregion_id])

            out_file = os.path.join(self.out_dir, f"radiomics_features_from_subregion_{subregion_id}.csv")
            radiomics_of_each_subregion[subregion_id].to_csv(out_file, index=True)
            logger.info(f"Saved features for tumor subregion {subregion_id} to {out_file}")
        
        print()  # 完成后换行
        # subregion count的header修改一下
        subregion_count.columns = [f"is_have_habitat_{i}" for i in range(1, n_habitats+1)]
        subregion_count.to_csv(os.path.join(self.out_dir, "subregion_count.csv"), index=True)
        logger.info("Saved tumor subregion count information")
        return radiomics_of_each_subregion

    def extract_features(self, feature_types: List[str], n_habitats: int = 5) -> Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]:
        """
        Extract specified types of features.

        Args:
            feature_types (List[str]): List of feature types to extract. Options:
                - 'traditional': Traditional radiomics features
                - 'non_radiomics': Non-radiomics features
                - 'whole_habitat': Whole habitat map features
                - 'subregion_features': Tumor subregion features
            n_habitats (int): Number of tumor subregions to process.

        Returns:
            Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]: Dictionary of extracted features.
        """
        results = {}
        feature_mapping = {
            'traditional': self.extract_traditional_radiomics,
            'non_radiomics': lambda: self.extract_non_radiomics_features(n_habitats=n_habitats),
            'whole_habitat': self.extract_radiomics_features_for_whole_habitat_map,
            'subregion_features': lambda: self.extract_radiomics_features_from_each_habitat(n_habitats=n_habitats)
        }
        
        for feature_type in feature_types:
            if feature_type in feature_mapping:
                logger.info(f"Extracting {feature_type} features")
                results[feature_type] = feature_mapping[feature_type]()
            else:
                logger.warning(f"Unknown feature type: {feature_type}")
        
        return results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Parse habitat features from radiomics results')
    
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the npy file containing habitat features')
    
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for parsed features')
    
    parser.add_argument('--n_habitats', type=int, default=5,
                        help='Number of tumor subregions to process (default: 5)')
    
    parser.add_argument('--feature_types', nargs='+', type=str,
                        choices=['traditional', 'non_radiomics', 'whole_habitat', 'subregion_features'],
                        default=['traditional', 'non_radiomics', 'whole_habitat', 'subregion_features'],
                        help='Types of features to extract')
    
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with default parameters')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Set default command line arguments if none provided
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--file', r'F:\work\research\radiomics_TLSs\data\results\habitats_radiomics_features.npy',
            '--out_dir', r'F:\work\research\radiomics_TLSs\data\results\parsed_habitat_features',
            '--n_habitats', '5',
            '--feature_types', 'traditional', 'non_radiomics', 'whole_habitat', 'subregion_features',
            '--debug'
        ])
    
    args = parse_arguments()
    
    # Initialize feature parser and extract features
    hfe = HabitatFeatureParse(args.file, args.out_dir)
    results = hfe.extract_features(args.feature_types, args.n_habitats)
    
    logger.info("Feature extraction completed successfully")