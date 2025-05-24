"""
A Complete Modeling Pipeline

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from .models.factory import ModelFactory
from .feature_selectors import run_selector, get_available_selectors
from .evaluation.model_evaluation import ModelEvaluator, calculate_metrics
from .visualization.plotting import Plotter 
from habit.utils.log_utils import setup_output_logger, setup_logger

# Ignore warnings
warnings.filterwarnings("ignore")

class Modeling:
    """
    A class for radiomics modeling pipeline
    
    The pipeline includes the following steps:
    1. Reading data from one or multiple files
    2. Data preprocessing and cleaning
    3. Splitting data into training and testing sets
    4. Feature selection before normalization (for variance-sensitive methods)
    5. Z-score normalization
    6. Feature selection after normalization
    7. Model training
    8. Model evaluation and visualization
    
    Feature selection can be performed in two phases:
    - Before normalization: Useful for methods sensitive to feature variance 
      (e.g., variance threshold filter, as Z-score makes all features have unit variance)
    - After normalization: For methods that benefit from normalized data
      
    To control when a feature selection method runs, use the 'before_z_score' 
    parameter in the configuration file.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the modeling class
        
        Args:
            config: Configuration dictionary containing all modeling parameters
        """
        self.config = config
        self.data_file = config['input']
        self.output_dir = config['output']
        self.test_size = config.get('test_size', 0.3)
        self.SEED = config.get('random_state', 42)
        self.feature_selection_methods = config.get('feature_selection_methods')
        
        # Normalization configuration
        self.normalization_config = config.get('normalization', {'method': 'z_score'})
        
        # Setup logger
        self.logger = setup_output_logger(self.output_dir, name="modeling", level=config.get('log_level', 20))
        self.logger.info("Initializing modeling with config: %s", config)
        
        # Data split configuration
        self.split_method = config.get('split_method', 'stratified')
        self.train_ids_file = config.get('train_ids_file', None)
        self.test_ids_file = config.get('test_ids_file', None)

        # visualization and save configuration
        self.is_visualize = config.get('is_visualize', False)
        self.is_save_model = config.get('is_save_model', False)
        
        # Get available feature selectors
        try:
            self.available_selectors = get_available_selectors()
            print(f"Available feature selectors: {self.available_selectors}")
            self.logger.info("Available feature selectors: %s", self.available_selectors)
        except Exception as e:
            self.available_selectors = []
            error_msg = f"Warning: Failed to get available feature selectors: {e}"
            print(error_msg)
            self.logger.warning(error_msg)
    
        self.plotter = Plotter(self.output_dir)
        self.logger.info("Modeling initialization completed")
        
    def read_data(self) -> 'Modeling':
        """
        Read data from file(s)
        
        Returns:
            Modeling: Self instance for method chaining
        """
            
        if isinstance(self.data_file, list):
            # Multiple files case
            print(f"Reading data from multiple files: {len(self.data_file)} files")
            self.logger.info("Reading data from multiple files: %d files", len(self.data_file))
            
            merged_df = None
            all_labels = {}  # 用于存储所有样本的标签值
            first_subject_id_col = None  # 记录第一个subject_id_col
            first_label_col = None  # 记录第一个标签列名
            
            for ithf, file_config in enumerate(self.data_file):
                file_path = file_config['path']
                name = file_config.get('name', f"model{ithf+1}")
                subject_id_col = file_config.get('subject_id_col')
                label_col = file_config.get('label_col')
                
                # Check if subject_id_col or label_col is None
                if subject_id_col is None:
                    error_msg = f"Subject ID column must be specified for file {file_path}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                if label_col is None:
                    error_msg = f"Label column must be specified for file {file_path}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                features = file_config.get('features', [])
                
                print(f"  Reading file: {file_path}")
                print(f"  Dataset name: {name}")
                print(f"  Subject ID column: {subject_id_col}")
                print(f"  Label column: {label_col}")
                print(f"  Features: {features}")
                
                self.logger.info("Reading file: %s, Dataset name: %s", file_path, name)
                self.logger.info("Subject ID column: %s, Label column: %s", subject_id_col, label_col)
                self.logger.debug("Features: %s", features)
                
                # Read the file
                df = pd.read_csv(file_path)
                self.logger.info("File loaded with shape: %s", str(df.shape))

                # check if subject_id_col and label_col exist in df
                if subject_id_col not in df.columns:
                    error_msg = f"Subject ID column '{subject_id_col}' not found in {file_path}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                if label_col not in df.columns:
                    error_msg = f"Label column '{label_col}' not found in {file_path}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Convert subject IDs to string for consistent merging
                df[subject_id_col] = df[subject_id_col].astype(str)
                
                # 记录第一个文件的标签列名作为最终标签列名
                if first_label_col is None:
                    first_label_col = label_col
                    self.label_col = first_label_col
                    label_values = df[label_col]
                    self.logger.info("Using label column: %s", self.label_col)
                
                # 记录第一个文件的subject_id_col作为最终subject_id_col
                if first_subject_id_col is None:
                    first_subject_id_col = subject_id_col
                    self.subject_id_col = first_subject_id_col
                    self.logger.info("Using subject ID column: %s", self.subject_id_col)
                
                # 提取并保存标签数据,同时保留subjID
                all_labels[file_path] = df.loc[:, [subject_id_col, label_col]]
                all_labels[file_path].set_index(subject_id_col, inplace=True)
                
                # 选择要保留的列，不包括标签列
                columns_to_keep = [subject_id_col]
                
                # 添加特征列，添加前缀避免冲突
                feature_columns = {}
                if features:
                    for feature in features:
                        if feature in df.columns:
                            columns_to_keep.append(feature)
                            # 为特征列名添加前缀
                            feature_columns[feature] = f"{name}{feature}"
                        else:
                            warning_msg = f"Warning: Feature '{feature}' not found in {file_path}"
                            print(warning_msg)
                            self.logger.warning(warning_msg)
                else:
                    # 如果未指定特征，使用除ID和标签外的所有列
                    for col in df.columns:
                        if col != subject_id_col and col != label_col:
                            columns_to_keep.append(col)
                            if name is not None:
                                feature_columns[col] = f"{name}{col}"
                            else:
                                feature_columns[col] = col
                
                # 只保留需要的列
                df = df[columns_to_keep]
                
                # 重命名特征列，添加前缀
                rename_dict = {col: feature_columns[col] for col in feature_columns if col in df.columns}
                df = df.rename(columns=rename_dict)
                self.logger.debug("Renamed features: %s", rename_dict)
                
                # 首次处理初始化合并数据框
                if merged_df is None:
                    merged_df = df
                    # 设置索引用于后续合并
                    merged_df.set_index(subject_id_col, inplace=True)
                    # 索引转为字符串
                    merged_df.index = merged_df.index.astype(str)
                    self.logger.info("Created initial merged dataframe with shape: %s", str(merged_df.shape))
                else:
                    # 为合并设置临时索引
                    df.set_index(subject_id_col, inplace=True)
                    # 索引转为字符串
                    df.index = df.index.astype(str)
                    
                    # 与现有数据合并
                    merged_df = merged_df.join(df, how='outer')
                    self.logger.info("Merged dataframe updated, new shape: %s", str(merged_df.shape))

                    # 更新索引名称
                    merged_df.index.name = self.subject_id_col

            # 将标签数据添加回合并后的数据框
            print(f"Adding unified label column: {self.label_col}")
            self.logger.info("Adding unified label column: %s", self.label_col)
            merged_df[self.label_col] = label_values.values
            # 把label放到第一列
            merged_df = merged_df[[self.label_col] + [col for col in merged_df.columns if col != self.subject_id_col and col != self.label_col]]
            self.data = merged_df
            self.logger.info("Final merged dataframe shape: %s with %d features", 
                          str(self.data.shape), self.data.shape[1]-1)

        else:
            error_msg = f"Expected list for data_file, got {type(self.data_file)}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        
        return self
    
    def preprocess_data(self) -> 'Modeling':
        """
        Preprocess the data
        
        Returns:
            Modeling: Self instance for method chaining
        """
        # Convert to numeric
        self.logger.info("Converting data to numeric types")
        self.data = self.data.apply(pd.to_numeric)
        self.logger.info("Data converted to numeric types")
       
        # Check for missing values
        missing_values = self.data.isnull().sum()
        self.logger.info(f"Missing values: {missing_values}")
        
        # Remove missing values
        self.logger.info("Starting data preprocessing")
        self.logger.info(f"Sample size before removing missing values: {self.data.shape[0]}")
        self.data = self.data.dropna()
        self.logger.info(f"Sample size after removing missing values: {self.data.shape[0]}")

        # Data exploration
        self.logger.info("Data preprocessing completed")
        return self

    def _split_data(self) -> 'Modeling':
        """
        Split data into training and test sets based on the specified method
        
        Returns:
            Modeling: Self instance for method chaining
        """
        self.logger.info(f"Splitting data using method: {self.split_method}")
        
        # Save original data to allow access to subject IDs later
        self.original_data = self.data.copy()
        
        # Different split methods
        if self.split_method == 'random':
            # Random split without stratification
            self.logger.info(f"Performing random split with test size: {self.test_size}")
            self.data_train, self.data_test = train_test_split(
                self.data,
                test_size=self.test_size, 
                random_state=self.SEED
            )
            print(f"Data split using random method (test size: {self.test_size})")
            self.logger.info(f"Random split completed: train={len(self.data_train)}, test={len(self.data_test)}")
            
        elif self.split_method == 'stratified':
            # Stratified split based on label column (default method)
            self.logger.info(f"Performing stratified split with test size: {self.test_size}")
            self.data_train, self.data_test = train_test_split(
                self.data,
                test_size=self.test_size, 
                random_state=self.SEED, 
                stratify=self.data[self.label_col]
            )
            print(f"Data split using stratified method (test size: {self.test_size})")
            self.logger.info(f"Stratified split completed: train={len(self.data_train)}, test={len(self.data_test)}")
            
        elif self.split_method == 'custom':
            # Use custom subject IDs for train and test sets
            if not self.train_ids_file or not self.test_ids_file:
                error_msg = "For custom split method, both train_ids_file and test_ids_file must be specified"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Read subject IDs from files
            try:
                self.logger.info(f"Reading custom train IDs from {self.train_ids_file}")
                train_ids = self._read_subject_ids(self.train_ids_file)
                self.logger.info(f"Read {len(train_ids)} train subject IDs")
                
                self.logger.info(f"Reading custom test IDs from {self.test_ids_file}")
                test_ids = self._read_subject_ids(self.test_ids_file)
                self.logger.info(f"Read {len(test_ids)} test subject IDs")
                
                # Ensure DataFrame index is converted to string type
                if not all(isinstance(idx, str) for idx in self.data.index):
                    print("Converting DataFrame index to string type")
                    self.logger.info("Converting DataFrame index to string type")
                    self.data.index = self.data.index.astype(str)
                
                # Verify no overlap between train and test
                overlap = set(train_ids).intersection(set(test_ids))
                if overlap:
                    warning_msg = f"Warning: Found {len(overlap)} overlapping subject IDs between train and test sets"
                    print(warning_msg)
                    self.logger.warning(warning_msg)
                
                # Create train and test datasets
                self.data_train = self.data.loc[self.data.index.isin(train_ids)]
                self.data_test = self.data.loc[self.data.index.isin(test_ids)]
                
                # Verify all IDs were found
                missing_train = set(train_ids) - set(self.data_train.index)
                missing_test = set(test_ids) - set(self.data_test.index)
                
                if missing_train:
                    warning_msg = f"Warning: {len(missing_train)} train subject IDs not found in data"
                    print(warning_msg)
                    self.logger.warning(warning_msg)
                    self.logger.debug(f"Missing train IDs: {list(missing_train)}")
                    
                if missing_test:
                    warning_msg = f"Warning: {len(missing_test)} test subject IDs not found in data"
                    print(warning_msg)
                    self.logger.warning(warning_msg)
                    self.logger.debug(f"Missing test IDs: {list(missing_test)}")
                
                print(f"Data split using custom subject IDs (train: {len(self.data_train)}, test: {len(self.data_test)})")
                self.logger.info(f"Custom split completed: train={len(self.data_train)}, test={len(self.data_test)}")
                
            except Exception as e:
                error_msg = f"Failed to split data using custom subject IDs: {e}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"Unsupported split method: {self.split_method}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract features and labels
        self.x_train = self.data_train.drop(self.label_col, axis=1)
        self.y_train = self.data_train[self.label_col]
        self.x_test = self.data_test.drop(self.label_col, axis=1)
        self.y_test = self.data_test[self.label_col]
        self.header = self.x_train.columns
        self.logger.info(f"Extracted features and labels: {len(self.x_train.columns)} features")
        
        # Save train/test split indices for later reference
        self._save_split_info()
        self.logger.info("Split information saved")
        
        return self
    
    def _read_subject_ids(self, file_path: str) -> List[str]:
        """
        Read subject IDs from a file
        
        Args:
            file_path: Path to file containing subject IDs
            
        Returns:
            List of subject IDs (all converted to string type)
        """
        self.logger.info(f"Reading subject IDs from file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                # Support different formats: one ID per line, CSV, or JSON
                content = f.read().strip()
                
                # Try to parse as JSON
                if content.startswith('[') and content.endswith(']'):
                    try:
                        ids = json.loads(content)
                        # Ensure all IDs are string type
                        result = [str(id) for id in ids]
                        self.logger.info(f"Parsed {len(result)} subject IDs from JSON format")
                        return result
                    except:
                        self.logger.debug("Failed to parse as JSON, trying other formats")
                        
                # Try to parse as CSV
                if ',' in content:
                    ids = [id.strip() for id in content.split(',')]
                    # Ensure all IDs are string type
                    result = [str(id) for id in ids]
                    self.logger.info(f"Parsed {len(result)} subject IDs from CSV format")
                    return result
                    
                # Default: one ID per line
                ids = [line.strip() for line in content.split('\n') if line.strip()]
                # Ensure all IDs are string type
                result = [str(id) for id in ids]
                self.logger.info(f"Parsed {len(result)} subject IDs from line-by-line format")
                return result
        except Exception as e:
            error_msg = f"Error reading subject IDs from file {file_path}: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _save_split_info(self) -> None:
        """
        Save train/test split information
        """
        self.logger.info("Saving train/test split information")
        split_info = {
            'split_method': self.split_method,
            'train_size': len(self.data_train),
            'test_size': len(self.data_test),
            'train_subjects': list(self.data_train.index),
            'test_subjects': list(self.data_test.index)
        }
        
        # Save to JSON
        split_info_path = os.path.join(self.output_dir, 'split_info.json')
        try:
            with open(split_info_path, 'w') as f:
                json.dump(split_info, f, indent=4)
            self.logger.info(f"Split information saved to: {split_info_path}")
        except Exception as e:
            self.logger.error(f"Failed to save split information: {e}")

    def feature_selection_before_normalization(self) -> 'Modeling':
        """
        Perform feature selection before normalization for methods with before_z_score=True
        
        This method executes feature selection methods that should run before Z-score normalization,
        which is particularly important for variance-based feature selection methods. After Z-score
        normalization, all features will have a variance of 1, making variance-based selection ineffective.
        
        The method works by:
        1. Filtering feature selection methods based on the 'before_z_score' parameter
        2. Saving the remaining methods for later execution after normalization
        3. Running the selected pre-normalization methods in sequence
        4. Updating the feature set in x_train and x_test
        
        Returns:
            Modeling: Self instance for method chaining
        """
        self.logger.info("Checking for feature selection methods to run before normalization")
        
        # Check if feature selection methods are provided
        if self.feature_selection_methods is None:
            self.logger.info("No feature selection methods provided")
            return self
        
        # Filter methods that should run before normalization
        before_zscore_methods = []
        remaining_methods = []
        
        for selection_config in self.feature_selection_methods:
            if selection_config.get('params', {}).get('before_z_score', False):
                before_zscore_methods.append(selection_config)
            else:
                remaining_methods.append(selection_config)
        
        if not before_zscore_methods:
            self.logger.info("No feature selection methods need to run before normalization")
            return self
        
        # Save remaining methods for later
        self.feature_selection_methods = remaining_methods
        
        # Execute feature selection methods that should run before normalization
        self.logger.info(f"Running {len(before_zscore_methods)} feature selection methods before normalization")
        
        # Store current feature set
        selected_features = list(self.x_train.columns)
        self.logger.info(f"Starting with {len(selected_features)} features")
        
        # Execute the methods
        for selection_config in before_zscore_methods:
            method_name = selection_config['method']
            params = selection_config.get('params', {})
            
            print(f"\nExecuting {method_name} feature selection BEFORE normalization...")
            self.logger.info(f"Executing {method_name} feature selection before normalization with params: {params}")
            
            try:
                # Add common parameters
                params['outdir'] = params.get('outdir', self.output_dir)
                
                # Run feature selector
                selected = run_selector(
                    method_name,
                    self.x_train,
                    self.y_train,
                    selected_features,
                    **params
                )
                
                # Update feature set
                selected_features = selected
                
                # Print selected features and removed features
                print(f"Selected features: {selected_features}")
                removed_count = len(self.x_train.columns) - len(selected_features)
                self.logger.info(f"{method_name} selected {len(selected_features)} features, removed {removed_count} features")
                
                # Warn if no features selected
                if not selected_features:
                    error_msg = f"No features remaining after {method_name} selection, please check settings"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
            except Exception as e:
                warning_msg = f"Warning: {method_name} selection failed: {e}"
                print(warning_msg)
                self.logger.warning(warning_msg)
                print(f"Skipping this method, continuing with current feature set: {len(selected_features)} features")
        
        # Update x_train and x_test to only include selected features
        self.x_train = self.x_train[selected_features]
        self.x_test = self.x_test[selected_features]
        
        self.logger.info(f"Pre-normalization feature selection completed. Selected {len(selected_features)} features")
        print(f"Pre-normalization feature selection completed. Selected {len(selected_features)} features")
        
        return self

    def normalization(self) -> 'Modeling':
        """
        Perform data normalization/standardization based on configuration
        
        Supported methods:
        - z_score (StandardScaler): Z-score standardization (zero mean, unit variance)
        - min_max (MinMaxScaler): Scale features to a given range, default [0, 1]
        - robust (RobustScaler): Scale features using statistics that are robust to outliers
        - max_abs (MaxAbsScaler): Scale features by their maximum absolute value
        - normalizer (Normalizer): Scale samples to have unit norm
        - quantile (QuantileTransformer): Transform features to follow a uniform or normal distribution
        - power (PowerTransformer): Apply a power transformation to make data more Gaussian-like
        
        Returns:
            Modeling: Self instance for method chaining
        """
        method = self.normalization_config.get('method', 'z_score')
        self.logger.info(f"Starting data normalization using method: {method}")
        
        # Get additional parameters
        params = self.normalization_config.get('params', {})
        
        # Initialize the appropriate scaler based on method
        if method == 'z_score':
            scaler = StandardScaler()
            self.logger.info("Using StandardScaler (Z-score normalization)")
        elif method == 'min_max':
            feature_range = params.get('feature_range', (0, 1))
            scaler = MinMaxScaler(feature_range=feature_range)
            self.logger.info(f"Using MinMaxScaler with range {feature_range}")
        elif method == 'robust':
            quantile_range = params.get('quantile_range', (25.0, 75.0))
            with_centering = params.get('with_centering', True)
            with_scaling = params.get('with_scaling', True)
            scaler = RobustScaler(
                quantile_range=quantile_range,
                with_centering=with_centering,
                with_scaling=with_scaling
            )
            self.logger.info(f"Using RobustScaler with quantile range {quantile_range}")
        elif method == 'max_abs':
            scaler = MaxAbsScaler()
            self.logger.info("Using MaxAbsScaler")
        elif method == 'normalizer':
            norm = params.get('norm', 'l2')
            scaler = Normalizer(norm=norm)
            self.logger.info(f"Using Normalizer with {norm} norm")
        elif method == 'quantile':
            n_quantiles = params.get('n_quantiles', 1000)
            output_distribution = params.get('output_distribution', 'uniform')
            scaler = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution=output_distribution,
                random_state=self.SEED
            )
            self.logger.info(f"Using QuantileTransformer with {output_distribution} distribution")
        elif method == 'power':
            method = params.get('method', 'yeo-johnson')
            standardize = params.get('standardize', True)
            scaler = PowerTransformer(method=method, standardize=standardize)
            self.logger.info(f"Using PowerTransformer with {method} method")
        else:
            self.logger.warning(f"Unknown normalization method '{method}', falling back to StandardScaler")
            scaler = StandardScaler()
        
        # Fit and transform the data
        self.logger.info(f"Fitting scaler on training data with shape: {self.x_train.shape}")
        x_train = scaler.fit_transform(self.x_train)
        self.logger.info(f"Transforming test data with shape: {self.x_test.shape}")
        x_test = scaler.transform(self.x_test)
        self.x_train.values[:,:] = x_train
        self.x_test.values[:,:] = x_test
        
        # Save scaler for future use
        self.scaler = scaler
        # Save feature names used during fit
        self.scaler_feature_names = list(self.x_train.columns)
        self.logger.info(f"Data normalization completed using {method} method")
        self.logger.debug(f"Feature names saved for scaler: {len(self.scaler_feature_names)} features")
        
        return self

    def feature_selection(self) -> 'Modeling':
        """
        Perform feature selection after normalization
        
        Executes feature selection methods that remain after pre-normalization feature selection.
        This method runs after Z-score normalization and processes all methods that do not have
        the 'before_z_score' parameter set to True.
        
        The feature selection methods are applied in sequence according to the configuration,
        with each method working on the features selected by the previous method.
        
        Returns:
            Modeling: Self instance for method chaining
        """
        self.logger.info("Starting feature selection")
        
        # Check if feature selection methods are provided
        if self.feature_selection_methods is None:
            print("Warning: No feature selection methods provided, using original feature set")
            self.logger.warning("No feature selection methods provided, using original feature set")
            self.selected_features = list(self.x_train.columns)
            self.logger.info(f"Using all {len(self.selected_features)} features")
            return self
        
        # Store current feature set
        selected_features = list(self.x_train.columns)
        self.logger.info(f"Starting with {len(selected_features)} features")
        
        # Execute feature selection methods in sequence
        selected_features_of_each_method = {}
        for selection_config in self.feature_selection_methods:
            method_name = selection_config['method']
            params = selection_config.get('params', {})
            
            print(f"\nExecuting {method_name} feature selection...")
            self.logger.info(f"Executing {method_name} feature selection with params: {params}")
            
            try:
                # Add common parameters
                params['outdir'] = params.get('outdir', self.output_dir)
                
                # Run feature selector
                selected = run_selector(
                    method_name,
                    self.x_train,
                    self.y_train,
                    selected_features,
                    **params
                )

                # selected_features和selected交集  TODO
                # 因为如果设置方差筛选和如ICC筛选时，用于方差筛选在zscore前
                # 导致方差筛选后，特征剩下的1，2，3，而ICC后剩下的可能是1，2，3，4
                # 这样x_train以及没有了4，但是selected还有4
                # 在后续x_train[selected]时，会报错,所以需要取交集
                selected = list(set(selected_features) & set(selected))
                
                # Update feature set
                selected_features = selected
                selected_features_of_each_method[method_name] = selected_features

                # print selected features and removed features
                print(f"Selected features: {selected_features}")
                removed_count = len(self.x_train.columns) - len(selected_features)
                self.logger.info(f"{method_name} selected {len(selected_features)} features, removed {removed_count} features")
                self.logger.debug(f"Selected features: {selected_features}")
                
                # Warn if no features selected
                if not selected_features:
                    error_msg = f"No features remaining after {method_name} selection, please check settings"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                
            except Exception as e:
                warning_msg = f"Warning: {method_name} selection failed: {e}"
                print(warning_msg)
                self.logger.warning(warning_msg)
                self.logger.warning(f"Skipping this method, continuing with current feature set: {len(selected_features)} features")
                print(f"Skipping this method, continuing with current feature set: {len(selected_features)} features")
        
        # Final selected features
        self.selected_features = selected_features
        print(f"\nFinal number of selected features: {len(self.selected_features)}")
        self.logger.info(f"Feature selection completed. Final number of selected features: {len(self.selected_features)}")
        print(f"Selected features: {self.selected_features}")
        self.logger.debug(f"Final selected features: {self.selected_features}")
        
        # Save feature selection results
        results_dict = {
            'selected_features': self.selected_features,
            'feature_selection_methods': self.feature_selection_methods,
            'selected_features_of_each_method': selected_features_of_each_method
        }
        
        results_path = os.path.join(self.output_dir, 'feature_selection_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Feature selection results saved to: {results_path}")
        
        return self

    def modeling(self) -> 'Modeling':
        """
        Train machine learning models
        
        Returns:
            Modeling: Self instance for method chaining
        """
        self.logger.info("Starting model training")
        
        # Use direct model import instead of factory pattern
        results = {
            'train': {},
            'test': {}
        }
        models = {}
        
        # Iterate through models in configuration
        models_config = self.config.get('models', {})
        self.logger.info(f"Training {len(models_config)} models: {list(models_config.keys())}")
        
        for model_name, model_config in models_config.items():
            print(f"\nTraining model: {model_name}")
            self.logger.info(f"Training model: {model_name} with config: {model_config}")
            
            # Create model
            try:
                model = ModelFactory.create_model(model_name, model_config)
                self.logger.debug(f"Model {model_name} created successfully")
            except Exception as e:
                error_msg = f"Failed to create model {model_name}: {e}"
                self.logger.error(error_msg)
                print(error_msg)
                continue
            
            X_train = self.x_train[self.selected_features]
            X_test = self.x_test[self.selected_features]
            self.logger.info(f"Training model with {len(self.selected_features)} features")
            
            # Train model
            try:
                model.fit(X_train, self.y_train)
                self.logger.info(f"Model {model_name} trained successfully")
            except Exception as e:
                error_msg = f"Failed to train model {model_name}: {e}"
                self.logger.error(error_msg)
                print(error_msg)
                continue
            
            # Save model object
            models[model_name] = model
        
        # Save results and models
        self.results = results
        self.models = models
        self.logger.info(f"Model training completed. Trained {len(models)} models successfully")
        
        return self
        
    def evaluate_models(self) -> 'Modeling':
        """
        Evaluate models, visualize results, and save evaluation results
        
        Returns:
            Modeling: Self instance for method chaining
        """
        self.logger.info("Starting model evaluation")
        
        # Create evaluator
        evaluator = ModelEvaluator(self.output_dir)
        
        # Get feature data
        X_train = self.x_train[self.selected_features]
        X_test = self.x_test[self.selected_features]
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\nEvaluating model: {model_name}")
            self.logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Evaluate model - training set
                self.logger.info(f"Evaluating {model_name} on training set")
                train_results = evaluator.evaluate(model, X_train, self.y_train, "train")
                self.results['train'][model_name] = train_results
                self.logger.info(f"Training set metrics: {train_results['metrics']}")
                
                # Evaluate model - test set
                self.logger.info(f"Evaluating {model_name} on test set")
                test_results = evaluator.evaluate(model, X_test, self.y_test, "test")
                self.results['test'][model_name] = test_results
                self.logger.info(f"Test set metrics: {test_results['metrics']}")
                
                # Print feature importance (if available)
                if hasattr(model, 'get_feature_importance'):
                    feature_importance = model.get_feature_importance()
                    print(f"Feature importance: {feature_importance}")
                    self.logger.info(f"Feature importance calculated for {model_name}")
                    self.logger.debug(f"Feature importance: {feature_importance}")
                
                # Plot SHAP values for each model
                try:
                    self.logger.info(f"Plotting SHAP values for {model_name}")
                    self.plotter.plot_shap(model, X_test, self.selected_features, save_name=f'{model_name}_SHAP.pdf')
                    self.logger.info(f"SHAP plot saved for {model_name}")
                except Exception as e:
                    warning_msg = f"Failed to plot SHAP values for {model_name}: {e}"
                    self.logger.warning(warning_msg)
            except Exception as e:
                error_msg = f"Error evaluating model {model_name}: {e}"
                self.logger.error(error_msg)
                print(error_msg)
        
        # Print performance table
        try:
            evaluator._print_performance_table(self.results)
            self.logger.info("Performance table generated")
        except Exception as e:
            warning_msg = f"Failed to print performance table: {e}"
            self.logger.warning(warning_msg)
        
        # Save performance table
        try:
            evaluator._save_performance_table(self.results)
            self.logger.info("Performance table saved")
        except Exception as e:
            warning_msg = f"Failed to save performance table: {e}"
            self.logger.warning(warning_msg)
        
        # Plot evaluation results
        if self.is_visualize:
            try:
                self.logger.info("Generating visualization plots")
                evaluator.plot_curves(self.results)
                self.logger.info("Visualization plots saved")
            except Exception as e:
                warning_msg = f"Failed to generate plots: {e}"
                self.logger.warning(warning_msg)
        
        # Save detailed prediction results
        try:
            self.logger.info("Saving detailed prediction results")
            self._save_prediction_results()
            self.logger.info("Prediction results saved")
        except Exception as e:
            warning_msg = f"Failed to save prediction results: {e}"
            self.logger.warning(warning_msg)
        
        # Save trained models and all necessary preprocessing information
        if self.is_save_model:
            try:
                self.logger.info("Saving trained models and preprocessing information")
                self._save_complete_models()
                self.logger.info("Models and preprocessing information saved")
            except Exception as e:
                warning_msg = f"Failed to save models: {e}"
                self.logger.warning(warning_msg)
        
        self.logger.info("Model evaluation completed")
        return self

    def _save_prediction_results(self) -> None:
        """
        Save detailed prediction results to CSV files
        """
        self.logger.info("Saving detailed prediction results")
        
        # Create clean dataframes for results with only essential columns
        train_df = pd.DataFrame({self.subject_id_col: self.data_train.index})
        test_df = pd.DataFrame({self.subject_id_col: self.data_test.index})
        
        # Add true labels
        train_df['true_label'] = self.y_train.values
        test_df['true_label'] = self.y_test.values
        
        # Add predictions and probabilities for each model
        for model_name in self.results['train'].keys():
            # Training set results
            train_df[f'{model_name}_pred'] = self.results['train'][model_name]['y_pred']
            train_df[f'{model_name}_prob'] = self.results['train'][model_name]['y_pred_proba']
            
            # Testing set results
            test_df[f'{model_name}_pred'] = self.results['test'][model_name]['y_pred']
            test_df[f'{model_name}_prob'] = self.results['test'][model_name]['y_pred_proba']
        
        # Save all-in-one results with both training and testing data
        all_df = pd.concat([train_df, test_df])
        # Add a column to distinguish train and test samples
        all_df['split'] = ['train'] * len(train_df) + ['test'] * len(test_df)
        
        all_results_path = os.path.join(self.output_dir, 'all_prediction_results.csv')
        all_df.to_csv(all_results_path, index=False)
        print(f"Saved complete prediction results to: {all_results_path}")
        self.logger.info(f"Saved complete prediction results to: {all_results_path}")
    
    def _save_complete_models(self) -> None:
        """
        Save trained models and all necessary preprocessing information
        """
        self.logger.info("Saving trained models and preprocessing information")
        import pickle
        
        # Create a dictionary with all information needed for future predictions
        model_package = {
            'models': self.models,
            'scaler': self.scaler,
            'scaler_feature_names': self.scaler_feature_names,  # Save feature names used during scaling
            'selected_features': self.selected_features,
            'config': self.config,
            'preprocessing_info': {
                'subject_id_col': self.subject_id_col,
                'label_col': self.label_col,
                'feature_selections': self.feature_selection_methods,
                'normalization': self.scaler
            }
        }
        
        # Save the complete model package
        model_package_path = os.path.join(self.output_dir, 'model_package.pkl')
        try:
            with open(model_package_path, 'wb') as f:
                pickle.dump(model_package, f)
            
            print(f"Saved complete model package to: {model_package_path}")
            self.logger.info(f"Saved complete model package to: {model_package_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model package: {e}")
            print(f"Error saving model package: {e}")
            return
        
        # Prepare model names list for usage instructions
        model_names_list = '\n'.join([f"- {name}" for name in self.models.keys()])
        
        # Save usage instructions as a text file
        usage_instructions = f"""
        Model Usage Instructions
        =======================

        This directory contains trained models and all necessary information for making new predictions.
        Here's how to use the saved models:

        1. Load the model package:
        ```python
        import pickle
        with open('model_package.pkl', 'rb') as f:
            model_package = pickle.load(f)
        ```

        2. Extract components:
        ```python
        models = model_package['models']
        scaler = model_package['scaler']
        scaler_feature_names = model_package['scaler_feature_names']
        selected_features = model_package['selected_features']
        preprocessing_info = model_package['preprocessing_info']
        ```

        3. Preprocess new data:
        - Ensure your data includes all features used during training
        - Apply the same preprocessing (scaling) using the safe_transform function:
        ```python
        def safe_transform(scaler, data, scaler_feature_names):
            '''
            Apply scaling only to columns that were present during fit
            
            Args:
                scaler: Trained scaler object
                data: DataFrame to transform
                scaler_feature_names: Feature names used during scaler fit
            '''
            # Get common columns between data and scaler features
            common_cols = [col for col in scaler_feature_names if col in data.columns]
            
            # Get missing columns
            missing_cols = [col for col in scaler_feature_names if col not in data.columns]
            if missing_cols:
                print(f"Warning: Missing columns in new data: {{missing_cols}}")
            
            # Transform only the columns that were present during fit
            if common_cols:
                data_to_transform = data[common_cols].copy()
                data_transformed = scaler.transform(data_to_transform)
                result = data.copy()
                result[common_cols] = data_transformed
                return result
            else:
                print("No common columns found for transformation")
                return data

        # Transform the new data safely
        X_new_scaled = safe_transform(scaler, your_data, scaler_feature_names)

        # Filter to only include selected features for prediction
        X_new_selected = X_new_scaled[selected_features]
        ```

        4. Make predictions with one or more models:
        ```python
        # Choose a model (e.g., LogisticRegression)
        model = models['LogisticRegression']

        # Make predictions
        predictions = model.predict(X_new_selected)
        probabilities = model.predict_proba(X_new_selected)[:, 1]  # Probability of positive class
        ```

        5. Available models:
        {model_names_list}

        6. Command-line prediction:
        You can also use the command-line interface for predictions:
        ```
        python day_10_whole_code.py --mode predict --model {model_package_path} --data your_data.csv
        ```
        """

        # Save usage instructions
        try:
            instructions_path = os.path.join(self.output_dir, 'model_usage_instructions.txt')
            with open(instructions_path, 'w') as f:
                f.write(usage_instructions)
            self.logger.info(f"Saved model usage instructions to: {instructions_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save usage instructions: {e}")

    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """
        Apply trained model to new data
        
        Args:
            new_data (pd.DataFrame): New data for prediction
            model_name (str, optional): Name of model to use. If None, use all models.
            
        Returns:
            pd.DataFrame: Dataframe with predictions (only essential columns)
        """
        self.logger.info(f"Predicting on new data with shape: {new_data.shape}")
        if model_name:
            self.logger.info(f"Using specific model: {model_name}")
        else:
            self.logger.info(f"Using all available models: {list(self.models.keys())}")
            
        # Check if the models are available
        if not hasattr(self, 'models'):
            error_msg = "No trained models available. Run the modeling pipeline first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create a new DataFrame with only necessary columns
        if self.subject_id_col in new_data.columns:
            # ID is a column
            result_df = pd.DataFrame({self.subject_id_col: new_data[self.subject_id_col]})
            self.logger.info(f"Found subject ID column: {self.subject_id_col}")
        elif isinstance(new_data.index, pd.Index) and new_data.index.name == self.subject_id_col:
            # ID is the index
            result_df = pd.DataFrame({self.subject_id_col: new_data.index})
            self.logger.info(f"Using index as subject ID column: {self.subject_id_col}")
        else:
            # Create numerical IDs
            warning_msg = f"Warning: Subject ID column '{self.subject_id_col}' not found. Using row numbers."
            print(warning_msg)
            self.logger.warning(warning_msg)
            result_df = pd.DataFrame({self.subject_id_col: range(len(new_data))})
        
        # Add true label if available
        if self.label_col in new_data.columns:
            self.logger.info(f"Found label column: {self.label_col}")
            result_df[self.label_col] = new_data[self.label_col]
        
        # Remove label column for processing
        X_new = new_data.drop([self.label_col], axis=1) if self.label_col in new_data.columns else new_data.copy()
        
        # Apply scaling safely - only to columns that were present during fit
        def safe_transform(data):
            # Get common columns between data and scaler features
            common_cols = [col for col in self.scaler_feature_names if col in data.columns]
            
            # Get missing columns
            missing_cols = [col for col in self.scaler_feature_names if col not in data.columns]
            if missing_cols:
                warning_msg = f"Warning: Missing columns in new data: {missing_cols}"
                print(warning_msg)
                self.logger.warning(warning_msg)
            
            # Transform only the columns that were present during fit
            if common_cols:
                self.logger.info(f"Applying scaling to {len(common_cols)} common columns")
                data_to_transform = data[common_cols].copy()
                data_transformed = self.scaler.transform(data_to_transform)
                result = data.copy()
                result[common_cols] = data_transformed
                return result
            else:
                warning_msg = "No common columns found for transformation"
                print(warning_msg)
                self.logger.warning(warning_msg)
                return data
        
        # Scale the data
        X_new_scaled = safe_transform(X_new)
        
        # Check for missing selected features
        missing_features = [f for f in self.selected_features if f not in X_new_scaled.columns]
        if missing_features:
            error_msg = f"Missing features in new data: {missing_features}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Filter to selected features only
        X_new_selected = X_new_scaled[self.selected_features]
        self.logger.info(f"Selected {len(self.selected_features)} features for prediction")
        
        # Get models to use
        models_to_use = {model_name: self.models[model_name]} if model_name else self.models
        
        # Make predictions with each model
        for name, model in models_to_use.items():
            self.logger.info(f"Making predictions with model: {name}")
            
            try:
                # Predict class
                y_pred = model.predict(X_new_selected)
                result_df[f'{name}_pred'] = y_pred
                
                # Predict probability
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_new_selected)[:, 1]
                else:
                    y_pred_proba = model.decision_function(X_new_selected)
                    # Convert decision function values to probabilities
                    y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
                    
                result_df[f'{name}_prob'] = y_pred_proba
                self.logger.info(f"Predictions made with model {name} successfully")
            except Exception as e:
                error_msg = f"Error making predictions with model {name}: {e}"
                self.logger.error(error_msg)
                print(error_msg)
        
        self.logger.info(f"Prediction completed for {len(new_data)} samples")
        return result_df

    @staticmethod
    def load_and_predict(model_file_path: str, new_data: pd.DataFrame, model_name: str = None, 
                        output_dir: str = None, evaluate: bool = False) -> pd.DataFrame:
        """
        Load a saved model package and make predictions on new data
        
        Args:
            model_file_path (str): Path to the saved model package (.pkl file)
            new_data (pd.DataFrame): New data for prediction
            model_name (str, optional): Name of model to use. If None, use all models.
            output_dir (str, optional): Directory to save evaluation results and plots
            evaluate (bool, optional): Whether to evaluate model performance and generate plots
            
        Returns:
            pd.DataFrame: DataFrame with predictions (only essential columns)
        """
        # Setup logger
        logger = setup_logger("model_prediction", output_dir)
        logger.info(f"Loading model from {model_file_path} for prediction")
        
        # Load the model package
        import pickle
        try:
            with open(model_file_path, 'rb') as f:
                model_package = pickle.load(f)
                
            print(f"Successfully loaded model package from {model_file_path}")
            logger.info(f"Successfully loaded model package from {model_file_path}")
        except Exception as e:
            error_msg = f"Failed to load model package: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Extract components
        models = model_package.get('models', {})
        scaler = model_package.get('scaler')
        scaler_feature_names = model_package.get('scaler_feature_names', [])
        selected_features = model_package.get('selected_features', [])
        preprocessing_info = model_package.get('preprocessing_info', {})
        
        logger.info(f"Loaded model package with {len(models)} models")
        logger.info(f"Selected features: {len(selected_features)} features")
        
        # Check if required components are available
        if not models:
            error_msg = "No models found in the model package"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if model_name and model_name not in models:
            error_msg = f"Model '{model_name}' not found in the model package. Available models: {list(models.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get models to use
        models_to_use = {model_name: models[model_name]} if model_name else models
        logger.info(f"Using models: {list(models_to_use.keys())}")
        
        # Get subject ID and label column names
        subject_id_col = preprocessing_info.get('subject_id_col', 'subjID')
        label_col = preprocessing_info.get('label_col', 'label')
        logger.info(f"Subject ID column: {subject_id_col}, Label column: {label_col}")
        
        # Create a new DataFrame with only necessary columns
        if subject_id_col in new_data.columns:
            # ID is a column
            result_df = pd.DataFrame({subject_id_col: new_data[subject_id_col]})
            logger.info(f"Found subject ID column: {subject_id_col}")
        elif isinstance(new_data.index, pd.Index) and new_data.index.name == subject_id_col:
            # ID is the index
            result_df = pd.DataFrame({subject_id_col: new_data.index})
            logger.info(f"Using index as subject ID column: {subject_id_col}")
        else:
            # Create numerical IDs
            warning_msg = f"Warning: Subject ID column '{subject_id_col}' not found. Using row numbers."
            print(warning_msg)
            logger.warning(warning_msg)
            result_df = pd.DataFrame({subject_id_col: range(len(new_data))})
        
        # Add true label if available
        if label_col in new_data.columns:
            logger.info(f"Found label column: {label_col}")
            result_df[label_col] = new_data[label_col]
        
        # Remove label column if present for processing
        X_new = new_data.drop([label_col], axis=1) if label_col in new_data.columns else new_data.copy()
        
        # Apply scaling safely - only to columns that were present during fit
        def safe_transform(data, scaler, feature_names):
            if scaler is None:
                warning_msg = "Warning: No scaler found in model package. Skipping scaling step."
                print(warning_msg)
                logger.warning(warning_msg)
                return data
                
            # Get common columns between data and scaler features
            common_cols = [col for col in feature_names if col in data.columns]
            
            # Get missing columns
            missing_cols = [col for col in feature_names if col not in data.columns]
            if missing_cols:
                warning_msg = f"Warning: Missing columns in new data: {missing_cols}"
                print(warning_msg)
                logger.warning(warning_msg)
            
            # Transform only the columns that were present during fit
            if common_cols:
                logger.info(f"Applying scaling to {len(common_cols)} common columns")
                data_to_transform = data[common_cols].copy()
                data_transformed = scaler.transform(data_to_transform)
                result = data.copy()
                result[common_cols] = data_transformed
                return result
            else:
                warning_msg = "No common columns found for transformation"
                print(warning_msg)
                logger.warning(warning_msg)
                return data
        
        # Scale the data
        X_new_scaled = safe_transform(X_new, scaler, scaler_feature_names)
        
        # Check for missing selected features
        missing_features = [f for f in selected_features if f not in X_new_scaled.columns]
        if missing_features:
            error_msg = f"Missing required features in new data: {missing_features}. These features are needed for prediction."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Filter to selected features only
        X_new_selected = X_new_scaled[selected_features]
        logger.info(f"Selected {len(selected_features)} features for prediction")
        
        # Make predictions with each model
        for name, model in models_to_use.items():
            logger.info(f"Making predictions with model: {name}")
            
            try:
                # Predict class
                y_pred = model.predict(X_new_selected)
                result_df[f'{name}_pred'] = y_pred
                
                # Predict probability
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_new_selected)[:, 1]
                else:
                    y_pred_proba = model.decision_function(X_new_selected)
                    # Convert decision function values to probabilities
                    y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
                    
                result_df[f'{name}_prob'] = y_pred_proba
                logger.info(f"Predictions made with model {name} successfully")
            except Exception as e:
                error_msg = f"Error making predictions with model {name}: {e}"
                logger.error(error_msg)
                print(error_msg)
        
        # If true labels are available and evaluation is requested
        if label_col in new_data.columns and evaluate:
            logger.info("Performing model evaluation on prediction results")
            
            # Create output directory if not exists
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Using output directory: {output_dir}")
            else:
                output_dir = os.path.dirname(model_file_path)
                logger.info(f"Using default output directory: {output_dir}")
            
            # Create ModelEvaluator instance
            evaluator = ModelEvaluator(output_dir)
            
            # Prepare data for evaluation
            y_true = new_data[label_col]
            
            # Create results dictionary in the format expected by ModelEvaluator
            results = {
                'test': {},
                'train': {}  # Empty for prediction mode
            }
            
            for name in models_to_use:
                results['test'][name] = {
                    'metrics': calculate_metrics(y_true, result_df[f'{name}_pred'], result_df[f'{name}_prob']),
                    'y_true': y_true.tolist(),
                    'y_pred': result_df[f'{name}_pred'].tolist(),
                    'y_pred_proba': result_df[f'{name}_prob'].tolist()
                }
                logger.info(f"Evaluation metrics for {name}: {results['test'][name]['metrics']}")
            
            # Print performance table
            try:
                evaluator._print_performance_table(results)
                logger.info("Performance table generated")
            except Exception as e:
                logger.warning(f"Failed to print performance table: {e}")
            
            # Save performance table
            try:
                evaluator._save_performance_table(results)
                logger.info("Performance table saved")
            except Exception as e:
                logger.warning(f"Failed to save performance table: {e}")
            
            # Plot evaluation results
            try:
                evaluator.plot_curves(results)
                logger.info("Evaluation plots generated")
            except Exception as e:
                logger.warning(f"Failed to generate evaluation plots: {e}")
        
        logger.info(f"Prediction completed for {len(new_data)} samples")
        return result_df

