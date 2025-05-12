"""
A Complete Modeling Pipeline

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from .models.factory import ModelFactory
from .feature_selectors import run_selector, get_available_selectors
from .evaluation.model_evaluation import ModelEvaluator, calculate_metrics
from .visualization.plotting import Plotter 

# Ignore warnings
warnings.filterwarnings("ignore")

class Modeling:
    """
    A class for radiomics modeling pipeline
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
        except Exception as e:
            self.available_selectors = []
            print(f"Warning: Failed to get available feature selectors: {e}")
    
        self.plotter = Plotter(self.output_dir)
        
    def read_data(self) -> 'Modeling':
        """
        Read data from file(s)
        
        Returns:
            Modeling: Self instance for method chaining
        """
            
        if isinstance(self.data_file, list):
            # Multiple files case
            print(f"Reading data from multiple files: {len(self.data_file)} files")
            
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
                    raise ValueError(f"Subject ID column must be specified for file {file_path}")
                if label_col is None:
                    raise ValueError(f"Label column must be specified for file {file_path}")
                
                features = file_config.get('features', [])
                
                print(f"  Reading file: {file_path}")
                print(f"  Dataset name: {name}")
                print(f"  Subject ID column: {subject_id_col}")
                print(f"  Label column: {label_col}")
                print(f"  Features: {features}")
                
                # Read the file
                df = pd.read_csv(file_path)

                # check if subject_id_col and label_col exist in df
                if subject_id_col not in df.columns:
                    raise ValueError(f"Subject ID column '{subject_id_col}' not found in {file_path}")
                if label_col not in df.columns:
                    raise ValueError(f"Label column '{label_col}' not found in {file_path}")
                
                # Convert subject IDs to string for consistent merging
                df[subject_id_col] = df[subject_id_col].astype(str)
                
                # 记录第一个文件的标签列名作为最终标签列名
                if first_label_col is None:
                    first_label_col = label_col
                    self.label_col = first_label_col
                    label_values = df[label_col]
                
                # 记录第一个文件的subject_id_col作为最终subject_id_col
                if first_subject_id_col is None:
                    first_subject_id_col = subject_id_col
                    self.subject_id_col = first_subject_id_col
                
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
                            print(f"  Warning: Feature '{feature}' not found in {file_path}")
                else:
                    # 如果未指定特征，使用除ID和标签外的所有列
                    for col in df.columns:
                        if col != subject_id_col and col != label_col:
                            columns_to_keep.append(col)
                            feature_columns[col] = f"{name}{col}"
                
                # 只保留需要的列
                df = df[columns_to_keep]
                
                # 重命名特征列，添加前缀
                rename_dict = {col: feature_columns[col] for col in feature_columns if col in df.columns}
                df = df.rename(columns=rename_dict)
                
                # 首次处理初始化合并数据框
                if merged_df is None:
                    merged_df = df
                    # 设置索引用于后续合并
                    merged_df.set_index(subject_id_col, inplace=True)
                    # 索引转为字符串
                    merged_df.index = merged_df.index.astype(str)
                else:
                    # 为合并设置临时索引
                    df.set_index(subject_id_col, inplace=True)
                    # 索引转为字符串
                    df.index = df.index.astype(str)
                    
                    # 与现有数据合并
                    merged_df = merged_df.join(df, how='outer')

                    # 更新索引名称
                    merged_df.index.name = self.subject_id_col

            # 将标签数据添加回合并后的数据框
            print(f"Adding unified label column: {self.label_col}")
            merged_df[self.label_col] = label_values.values
            # 把label放到第一列
            merged_df = merged_df[[self.label_col] + [col for col in merged_df.columns if col != self.subject_id_col and col != self.label_col]]
            self.data = merged_df

        else:
            raise TypeError(f"Expected list for data_file, got {type(self.data_file)}")
        
        return self
    
    def preprocess_data(self) -> 'Modeling':
        """
        Preprocess the data
        
        Returns:
            Modeling: Self instance for method chaining
        """
        # Remove missing values
        print(f"Sample size before removing missing values: {self.data.shape[0]}")
        # self.data = self.data.dropna()
        print(f"Sample size after removing missing values: {self.data.shape[0]}")

        # Check for missing values
        print(f"Missing values: {self.data.isnull().sum()}")

        # Convert to numeric
        # self.data = self.data.apply(pd.to_numeric)

        # Perform Shapiro-Wilk normality test on all clinical data

        # Data exploration

        # etc.

        return self

    def _split_data(self) -> 'Modeling':
        """
        Split data into training and test sets based on the specified method
        
        Returns:
            Modeling: Self instance for method chaining
        """
        # Save original data to allow access to subject IDs later
        self.original_data = self.data.copy()
        
        # Different split methods
        if self.split_method == 'random':
            # Random split without stratification
            self.data_train, self.data_test = train_test_split(
                self.data,
                test_size=self.test_size, 
                random_state=self.SEED
            )
            print(f"Data split using random method (test size: {self.test_size})")
            
        elif self.split_method == 'stratified':
            # Stratified split based on label column (default method)
            self.data_train, self.data_test = train_test_split(
                self.data,
                test_size=self.test_size, 
                random_state=self.SEED, 
                stratify=self.data[self.label_col]
            )
            print(f"Data split using stratified method (test size: {self.test_size})")
            
        elif self.split_method == 'custom':
            # Use custom subject IDs for train and test sets
            if not self.train_ids_file or not self.test_ids_file:
                raise ValueError("For custom split method, both train_ids_file and test_ids_file must be specified")
            
            # Read subject IDs from files
            try:
                train_ids = self._read_subject_ids(self.train_ids_file)
                test_ids = self._read_subject_ids(self.test_ids_file)
                
                # Ensure DataFrame index is converted to string type
                if not all(isinstance(idx, str) for idx in self.data.index):
                    print("Converting DataFrame index to string type")
                    self.data.index = self.data.index.astype(str)
                
                # Verify no overlap between train and test
                overlap = set(train_ids).intersection(set(test_ids))
                if overlap:
                    print(f"Warning: Found {len(overlap)} overlapping subject IDs between train and test sets")
                
                # Create train and test datasets
                self.data_train = self.data.loc[self.data.index.isin(train_ids)]
                self.data_test = self.data.loc[self.data.index.isin(test_ids)]
                
                # Verify all IDs were found
                missing_train = set(train_ids) - set(self.data_train.index)
                missing_test = set(test_ids) - set(self.data_test.index)
                
                if missing_train:
                    print(f"Warning: {len(missing_train)} train subject IDs not found in data")
                if missing_test:
                    print(f"Warning: {len(missing_test)} test subject IDs not found in data")
                
                print(f"Data split using custom subject IDs (train: {len(self.data_train)}, test: {len(self.data_test)})")
                
            except Exception as e:
                raise ValueError(f"Failed to split data using custom subject IDs: {e}")
        else:
            raise ValueError(f"Unsupported split method: {self.split_method}")
        
        # Extract features and labels
        self.x_train = self.data_train.drop(self.label_col, axis=1)
        self.y_train = self.data_train[self.label_col]
        self.x_test = self.data_test.drop(self.label_col, axis=1)
        self.y_test = self.data_test[self.label_col]
        self.header = self.x_train.columns
        
        # Save train/test split indices for later reference
        self._save_split_info()
        
        return self
    
    def _read_subject_ids(self, file_path: str) -> List[str]:
        """
        Read subject IDs from a file
        
        Args:
            file_path: Path to file containing subject IDs
            
        Returns:
            List of subject IDs (all converted to string type)
        """
        with open(file_path, 'r') as f:
            # Support different formats: one ID per line, CSV, or JSON
            content = f.read().strip()
            
            # Try to parse as JSON
            if content.startswith('[') and content.endswith(']'):
                try:
                    ids = json.loads(content)
                    # Ensure all IDs are string type
                    return [str(id) for id in ids]
                except:
                    pass
                    
            # Try to parse as CSV
            if ',' in content:
                ids = [id.strip() for id in content.split(',')]
                # Ensure all IDs are string type
                return [str(id) for id in ids]
                
            # Default: one ID per line
            ids = [line.strip() for line in content.split('\n') if line.strip()]
            # Ensure all IDs are string type
            return [str(id) for id in ids]
    
    def _save_split_info(self) -> None:
        """
        Save train/test split information
        """
        split_info = {
            'split_method': self.split_method,
            'train_size': len(self.data_train),
            'test_size': len(self.data_test),
            'train_subjects': list(self.data_train.index),
            'test_subjects': list(self.data_test.index)
        }
        
        # Save to JSON
        with open(os.path.join(self.output_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=4)

    def normalization(self) -> 'Modeling':
        """
        Perform z-score standardization
        
        Returns:
            Modeling: Self instance for method chaining
        """
        ss = StandardScaler()
        x_train = ss.fit_transform(self.x_train)
        x_test = ss.transform(self.x_test)
        self.x_train.values[:,:] = x_train
        self.x_test.values[:,:] = x_test
        
        # Save scaler for future use
        self.scaler = ss
        # Save feature names used during fit
        self.scaler_feature_names = list(self.x_train.columns)
        
        return self

    def feature_selection(self) -> 'Modeling':
        """
        Perform feature selection
        
        Executes feature selection methods in sequence according to the configuration
        
        Returns:
            Modeling: Self instance for method chaining
        """
        # Check if feature selection methods are provided
        if self.feature_selection_methods is None:
            print("Warning: No feature selection methods provided, using original feature set")
            self.selected_features = list(self.x_train.columns)
            return self
        
        # Store current feature set
        selected_features = list(self.x_train.columns)
        
        # Execute feature selection methods in sequence
        selected_features_of_each_method = {}
        for selection_config in self.feature_selection_methods:
            method_name = selection_config['method']
            params = selection_config.get('params', {})
            
            print(f"\nExecuting {method_name} feature selection...")
            
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
                selected_features_of_each_method[method_name] = selected_features

                # print selected features and removed features
                print(f"Selected features: {selected_features}")
                
                # Warn if no features selected
                if not selected_features:
                    raise ValueError(f"No features remaining after {method_name} selection, please check settings")
                
            except Exception as e:
                print(f"Warning: {method_name} selection failed: {e}")
                print(f"Skipping this method, continuing with current feature set: {len(selected_features)} features")
        
        # Final selected features
        self.selected_features = selected_features
        print(f"\nFinal number of selected features: {len(self.selected_features)}")
        print(f"Selected features: {self.selected_features}")
        
        # Save feature selection results
        results_dict = {
            'selected_features': self.selected_features,
            'feature_selection_methods': self.feature_selection_methods,
            'selected_features_of_each_method': selected_features_of_each_method
        }
        
        with open(os.path.join(self.output_dir, 'feature_selection_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=4)
        
        return self

    def modeling(self) -> 'Modeling':
        """
        Train machine learning models
        
        Returns:
            Modeling: Self instance for method chaining
        """
        # Use direct model import instead of factory pattern
        results = {
            'train': {},
            'test': {}
        }
        models = {}
        
        # Iterate through models in configuration
        for model_name, model_config in self.config.get('models', {}).items():
            print(f"\nTraining model: {model_name}")
            
            # Create model
            model = ModelFactory.create_model(model_name, model_config)
            
            # Get feature data
            X_train = self.x_train[self.selected_features]
            X_test = self.x_test[self.selected_features]
            
            # Train model
            model.fit(X_train, self.y_train)
            
            # Save model object
            models[model_name] = model
        
        # Save results and models
        self.results = results
        self.models = models
        
        return self
        
    def evaluate_models(self) -> 'Modeling':
        """
        Evaluate models, visualize results, and save evaluation results
        
        Returns:
            Modeling: Self instance for method chaining
        """
        # Create evaluator
        evaluator = ModelEvaluator(self.output_dir)
        
        # Get feature data
        X_train = self.x_train[self.selected_features]
        X_test = self.x_test[self.selected_features]
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"\nEvaluating model: {model_name}")
            
            # Evaluate model - training set
            train_results = evaluator.evaluate(model, X_train, self.y_train, "train")
            self.results['train'][model_name] = train_results
            
            # Evaluate model - test set
            test_results = evaluator.evaluate(model, X_test, self.y_test, "test")
            self.results['test'][model_name] = test_results
            
            # Print feature importance (if available)
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance()
                print(f"Feature importance: {feature_importance}")
            
            # Plot SHAP values for each model
            self.plotter.plot_shap(model, X_test, self.selected_features, save_name=f'{model_name}_SHAP.pdf')  # save_name not used, only use .pdf suffix
        
        # Print performance table
        evaluator._print_performance_table(self.results)
        
        # Plot evaluation results
        if self.is_visualize:
            evaluator.plot_curves(self.results)
        
        # Save detailed prediction results
        self._save_prediction_results()
        
        # Save trained models and all necessary preprocessing information
        if self.is_save_model:
            self._save_complete_models()
        
        return self

    def _save_prediction_results(self) -> None:
        """
        Save detailed prediction results to CSV files
        """
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
    
    def _save_complete_models(self) -> None:
        """
        Save trained models and all necessary preprocessing information
        """
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
        with open(model_package_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"Saved complete model package to: {model_package_path}")
        
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
        with open(os.path.join(self.output_dir, 'model_usage_instructions.txt'), 'w') as f:
            f.write(usage_instructions)
            
    def predict_new_data(self, new_data: pd.DataFrame, model_name: str = None) -> pd.DataFrame:
        """
        Apply trained model to new data
        
        Args:
            new_data (pd.DataFrame): New data for prediction
            model_name (str, optional): Name of model to use. If None, use all models.
            
        Returns:
            pd.DataFrame: Dataframe with predictions (only essential columns)
        """
        # Check if the models are available
        if not hasattr(self, 'models'):
            raise ValueError("No trained models available. Run the modeling pipeline first.")
        
        # Create a new DataFrame with only necessary columns
        if self.subject_id_col in new_data.columns:
            # ID is a column
            result_df = pd.DataFrame({self.subject_id_col: new_data[self.subject_id_col]})
        elif isinstance(new_data.index, pd.Index) and new_data.index.name == self.subject_id_col:
            # ID is the index
            result_df = pd.DataFrame({self.subject_id_col: new_data.index})
        else:
            # Create numerical IDs
            print(f"Warning: Subject ID column '{self.subject_id_col}' not found. Using row numbers.")
            result_df = pd.DataFrame({self.subject_id_col: range(len(new_data))})
        
        # Add true label if available
        if self.label_col in new_data.columns:
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
                print(f"Warning: Missing columns in new data: {missing_cols}")
            
            # Transform only the columns that were present during fit
            if common_cols:
                data_to_transform = data[common_cols].copy()
                data_transformed = self.scaler.transform(data_to_transform)
                result = data.copy()
                result[common_cols] = data_transformed
                return result
            else:
                print("No common columns found for transformation")
                return data
        
        # Scale the data
        X_new_scaled = safe_transform(X_new)
        
        # Check for missing selected features
        missing_features = [f for f in self.selected_features if f not in X_new_scaled.columns]
        if missing_features:
            raise ValueError(f"Missing features in new data: {missing_features}")
        
        # Filter to selected features only
        X_new_selected = X_new_scaled[self.selected_features]
        
        # Get models to use
        models_to_use = {model_name: self.models[model_name]} if model_name else self.models
        
        # Make predictions with each model
        for name, model in models_to_use.items():
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
        # Load the model package
        import pickle
        try:
            with open(model_file_path, 'rb') as f:
                model_package = pickle.load(f)
                
            print(f"Successfully loaded model package from {model_file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model package: {e}")
        
        # Extract components
        models = model_package.get('models', {})
        scaler = model_package.get('scaler')
        scaler_feature_names = model_package.get('scaler_feature_names', [])
        selected_features = model_package.get('selected_features', [])
        preprocessing_info = model_package.get('preprocessing_info', {})
        
        # Check if required components are available
        if not models:
            raise ValueError("No models found in the model package")
        if model_name and model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in the model package. Available models: {list(models.keys())}")
        
        # Get models to use
        models_to_use = {model_name: models[model_name]} if model_name else models
        
        # Get subject ID and label column names
        subject_id_col = preprocessing_info.get('subject_id_col', 'subjID')
        label_col = preprocessing_info.get('label_col', 'label')
        
        # Create a new DataFrame with only necessary columns
        if subject_id_col in new_data.columns:
            # ID is a column
            result_df = pd.DataFrame({subject_id_col: new_data[subject_id_col]})
        elif isinstance(new_data.index, pd.Index) and new_data.index.name == subject_id_col:
            # ID is the index
            result_df = pd.DataFrame({subject_id_col: new_data.index})
        else:
            # Create numerical IDs
            print(f"Warning: Subject ID column '{subject_id_col}' not found. Using row numbers.")
            result_df = pd.DataFrame({subject_id_col: range(len(new_data))})
        
        # Add true label if available
        if label_col in new_data.columns:
            result_df[label_col] = new_data[label_col]
        
        # Remove label column if present for processing
        X_new = new_data.drop([label_col], axis=1) if label_col in new_data.columns else new_data.copy()
        
        # Apply scaling safely - only to columns that were present during fit
        def safe_transform(data, scaler, feature_names):
            if scaler is None:
                print("Warning: No scaler found in model package. Skipping scaling step.")
                return data
                
            # Get common columns between data and scaler features
            common_cols = [col for col in feature_names if col in data.columns]
            
            # Get missing columns
            missing_cols = [col for col in feature_names if col not in data.columns]
            if missing_cols:
                print(f"Warning: Missing columns in new data: {missing_cols}")
            
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
        
        # Scale the data
        X_new_scaled = safe_transform(X_new, scaler, scaler_feature_names)
        
        # Check for missing selected features
        missing_features = [f for f in selected_features if f not in X_new_scaled.columns]
        if missing_features:
            raise ValueError(f"Missing required features in new data: {missing_features}. These features are needed for prediction.")
        
        # Filter to selected features only
        X_new_selected = X_new_scaled[selected_features]
        
        # Make predictions with each model
        for name, model in models_to_use.items():
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
        
        # If true labels are available and evaluation is requested
        if label_col in new_data.columns and evaluate:
            # Create output directory if not exists
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                output_dir = os.path.dirname(model_file_path)
            
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
            
            # Print performance table
            evaluator._print_performance_table(results)
            
            # Plot evaluation results
            evaluator.plot_curves(results)
        
        return result_df
