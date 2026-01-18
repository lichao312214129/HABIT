"""
K-Fold Cross-Validation Modeling Pipeline

This module provides k-fold cross-validation functionality for machine learning models.

Author: Li Chao
Email: lich356@mail.sysu.edu.cn
"""
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, QuantileTransformer, PowerTransformer
from sklearn.model_selection import StratifiedKFold, KFold
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from .models.factory import ModelFactory
from .feature_selectors import run_selector, get_available_selectors
from .evaluation.model_evaluation import ModelEvaluator, calculate_metrics
from .visualization.plotting import Plotter 
from habit.utils.log_utils import setup_output_logger, setup_logger

# Ignore warnings
warnings.filterwarnings("ignore")

class ModelingKFold:
    """
    A class for radiomics modeling pipeline with K-fold cross-validation
    
    The pipeline includes the following steps:
    1. Reading data from one or multiple files
    2. Data preprocessing and cleaning
    3. K-fold cross-validation split
    4. Feature selection (performed within each fold to avoid data leakage)
    5. Model training and evaluation for each fold
    6. Aggregation of results across all folds
    
    Note: Feature selection is performed within each fold to avoid data leakage.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the k-fold modeling class
        
        Args:
            config: Configuration dictionary containing all modeling parameters
        """
        self.config = config
        self.data_file = config['input']
        self.output_dir = config['output']
        self.SEED = config.get('random_state', 42)
        
        # K-fold configuration
        self.n_splits = config.get('n_splits', 5)
        self.stratified = config.get('stratified', True)
        
        # Feature selection and normalization configuration
        self.feature_selection_methods = config.get('feature_selection_methods')
        self.normalization_config = config.get('normalization', {'method': 'z_score'})
        
        # Setup logger - check if CLI already configured logging
        from habit.utils.log_utils import LoggerManager, get_module_logger
        manager = LoggerManager()
        
        if manager.get_log_file() is not None:
            # Logging already configured by CLI, just get module logger
            self.logger = get_module_logger('modeling_kfold')
            self.logger.info("Using existing logging configuration from CLI entry point")
        else:
            # Logging not configured yet (e.g., direct class usage)
            self.logger = setup_logger(
                name="modeling_kfold",
                output_dir=self.output_dir,
                log_filename='kfold_cv.log',
                level=config.get('log_level', 20)
            )
        self.logger.info("Initializing k-fold modeling with config: %s", config)

        # Visualization and save configuration
        self.is_visualize = config.get('is_visualize', False)
        self.is_save_model = config.get('is_save_model', False)
        
        # Get available feature selectors
        try:
            self.available_selectors = get_available_selectors()
            self.logger.info("Available feature selectors: %s", self.available_selectors)
        except Exception as e:
            self.available_selectors = []
            error_msg = f"Warning: Failed to get available feature selectors: {e}"
            self.logger.warning(error_msg)
    
        self.plotter = Plotter(self.output_dir)
        self.logger.info("K-fold modeling initialization completed")
        
    def read_data(self) -> 'ModelingKFold':
        """
        Read data from file(s) - same as the standard modeling pipeline
        
        Returns:
            ModelingKFold: Self instance for method chaining
        """
        if isinstance(self.data_file, list):
            # Multiple files case
            self.logger.info("Reading data from multiple files: %d files", len(self.data_file))
            
            merged_df = None
            all_labels = {}
            first_subject_id_col = None
            first_label_col = None
            
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
                
                self.logger.info("Reading file: %s, Dataset name: %s", file_path, name)
                self.logger.info("Subject ID column: %s, Label column: %s", subject_id_col, label_col)
                
                # Read the file
                df = pd.read_csv(file_path)
                self.logger.info("File loaded with shape: %s", str(df.shape))

                # Check if subject_id_col and label_col exist in df
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
                
                # Record the first label column name
                if first_label_col is None:
                    first_label_col = label_col
                    self.label_col = first_label_col
                    label_values = df[label_col]
                    self.logger.info("Using label column: %s", self.label_col)
                
                # Record the first subject_id_col
                if first_subject_id_col is None:
                    first_subject_id_col = subject_id_col
                    self.subject_id_col = first_subject_id_col
                    self.logger.info("Using subject ID column: %s", self.subject_id_col)
                
                # Extract and save label data
                all_labels[file_path] = df.loc[:, [subject_id_col, label_col]]
                all_labels[file_path].set_index(subject_id_col, inplace=True)
                
                # Select columns to keep
                columns_to_keep = [subject_id_col]
                feature_columns = {}
                
                if features:
                    for feature in features:
                        if feature in df.columns:
                            columns_to_keep.append(feature)
                            feature_columns[feature] = f"{name}{feature}"
                        else:
                            warning_msg = f"Warning: Feature '{feature}' not found in {file_path}"
                            self.logger.warning(warning_msg)
                else:
                    for col in df.columns:
                        if col != subject_id_col and col != label_col:
                            columns_to_keep.append(col)
                            if name is not None:
                                feature_columns[col] = f"{name}{col}"
                            else:
                                feature_columns[col] = col
                
                df = df[columns_to_keep]
                rename_dict = {col: feature_columns[col] for col in feature_columns if col in df.columns}
                df = df.rename(columns=rename_dict)
                
                # Merge dataframes
                if merged_df is None:
                    merged_df = df
                    merged_df.set_index(subject_id_col, inplace=True)
                    merged_df.index = merged_df.index.astype(str)
                    self.logger.info("Created initial merged dataframe with shape: %s", str(merged_df.shape))
                else:
                    df.set_index(subject_id_col, inplace=True)
                    df.index = df.index.astype(str)
                    merged_df = merged_df.join(df, how='outer')
                    self.logger.info("Merged dataframe updated, new shape: %s", str(merged_df.shape))
                    merged_df.index.name = self.subject_id_col

            # Add label column back
            self.logger.info("Adding unified label column: %s", self.label_col)
            merged_df[self.label_col] = label_values.values
            merged_df = merged_df[[self.label_col] + [col for col in merged_df.columns if col != self.subject_id_col and col != self.label_col]]
            self.data = merged_df
            self.logger.info("Final merged dataframe shape: %s with %d features", 
                          str(self.data.shape), self.data.shape[1]-1)

        else:
            error_msg = f"Expected list for data_file, got {type(self.data_file)}"
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        
        return self
    
    def preprocess_data(self) -> 'ModelingKFold':
        """
        Preprocess the data
        
        Returns:
            ModelingKFold: Self instance for method chaining
        """
        # Convert to numeric
        self.logger.info("Converting data to numeric types")
        self.data = self.data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        self.logger.info("Data converted to numeric types")
       
        # Check for missing values
        missing_values = self.data.isnull().sum()
        self.logger.info(f"Missing values: {missing_values}")
        
        # Remove missing values
        self.logger.info("Starting data preprocessing")
        self.logger.info(f"Sample size before removing missing values: {self.data.shape[0]}")
        
        # Fill missing values for features with mean
        feature_columns = [col for col in self.data.columns if col != self.label_col]
        self.data[feature_columns] = self.data[feature_columns].fillna(self.data[feature_columns].mean())
        self.logger.info(f"Sample size after removing missing values: {self.data.shape[0]}")

        # Handle missing values in label column using KNN imputer
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        self.data[self.label_col] = imputer.fit_transform(self.data[self.label_col].values.reshape(-1, 1))
        self.data[self.label_col] = (self.data[self.label_col] > 0.5).astype(int)

        self.logger.info("Data preprocessing completed")
        return self

    def _create_kfold_splits(self) -> 'ModelingKFold':
        """
        Create K-fold cross-validation splits
        
        Returns:
            ModelingKFold: Self instance for method chaining
        """
        self.logger.info(f"Creating {self.n_splits}-fold cross-validation splits")
        
        # Extract features and labels
        X = self.data.drop(self.label_col, axis=1)
        y = self.data[self.label_col]
        
        # Create K-fold splitter
        if self.stratified:
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.SEED)
            self.logger.info("Using Stratified K-Fold with %d splits", self.n_splits)
        else:
            kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.SEED)
            self.logger.info("Using K-Fold with %d splits", self.n_splits)
        
        # Store splits
        self.splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            self.splits.append({
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx
            })
            self.logger.info(f"Fold {fold_idx + 1}: train={len(train_idx)}, val={len(val_idx)}")
        
        self.X = X
        self.y = y
        
        return self

    def _normalize_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """
        Normalize data using specified method
        
        Args:
            X_train: Training features
            X_val: Validation features
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Any]: Normalized training features, normalized validation features, and scaler
        """
        method = self.normalization_config.get('method', 'z_score')
        params = self.normalization_config.get('params', {})
        
        # Initialize scaler
        if method == 'z_score':
            scaler = StandardScaler()
        elif method == 'min_max':
            feature_range = params.get('feature_range', (0, 1))
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            quantile_range = params.get('quantile_range', (25.0, 75.0))
            with_centering = params.get('with_centering', True)
            with_scaling = params.get('with_scaling', True)
            scaler = RobustScaler(quantile_range=quantile_range, with_centering=with_centering, with_scaling=with_scaling)
        elif method == 'max_abs':
            scaler = MaxAbsScaler()
        elif method == 'normalizer':
            norm = params.get('norm', 'l2')
            scaler = Normalizer(norm=norm)
        elif method == 'quantile':
            n_quantiles = params.get('n_quantiles', 1000)
            output_distribution = params.get('output_distribution', 'uniform')
            scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution, random_state=self.SEED)
        elif method == 'power':
            power_method = params.get('method', 'yeo-johnson')
            standardize = params.get('standardize', True)
            scaler = PowerTransformer(method=power_method, standardize=standardize)
        else:
            scaler = StandardScaler()
        
        # Fit and transform
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        return X_train_scaled, X_val_scaled, scaler

    def _select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Perform feature selection on training data and apply to validation data
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            fold_idx: Current fold index
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, List[str]]: Selected training features, selected validation features, and selected feature names
        """
        if self.feature_selection_methods is None or len(self.feature_selection_methods) == 0:
            return X_train, X_val, list(X_train.columns)
        
        selected_features = list(X_train.columns)
        
        for selection_config in self.feature_selection_methods:
            method_name = selection_config['method']
            params = selection_config.get('params', {})
            
            # Create fold-specific output directory for feature selection
            fold_outdir = os.path.join(self.output_dir, f'fold_{fold_idx}', 'feature_selection')
            os.makedirs(fold_outdir, exist_ok=True)
            params['outdir'] = fold_outdir
            
            try:
                selected = run_selector(
                    method_name,
                    X_train,
                    y_train,
                    selected_features,
                    **params
                )
                
                # Take intersection with current features
                selected = list(set(selected_features) & set(selected))
                selected_features = selected
                
                if not selected_features:
                    error_msg = f"No features remaining after {method_name} selection in fold {fold_idx}"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
                    
            except Exception as e:
                warning_msg = f"Warning: {method_name} selection failed in fold {fold_idx}: {e}"
                self.logger.warning(warning_msg)
        
        return X_train[selected_features], X_val[selected_features], selected_features

    def run_kfold_cv(self) -> 'ModelingKFold':
        """
        Run k-fold cross-validation for all models
        
        Returns:
            ModelingKFold: Self instance for method chaining
        """
        self.logger.info("Starting k-fold cross-validation")
        
        # Initialize results storage
        self.cv_results = {
            'folds': [],
            'aggregated': {}
        }
        
        models_config = self.config.get('models', {})
        
        # Iterate through each fold
        for split in self.splits:
            fold_idx = split['fold']
            train_idx = split['train_idx']
            val_idx = split['val_idx']
            
            self.logger.info("Processing Fold %d/%d", fold_idx, self.n_splits)
            
            # Get train and validation data
            X_train = self.X.iloc[train_idx].copy()
            X_val = self.X.iloc[val_idx].copy()
            y_train = self.y.iloc[train_idx].copy()
            y_val = self.y.iloc[val_idx].copy()
            
            # Normalize data
            X_train_norm, X_val_norm, scaler = self._normalize_data(X_train, X_val)
            
            # Feature selection
            X_train_selected, X_val_selected, selected_features = self._select_features(
                X_train_norm, y_train, X_val_norm, fold_idx
            )
            
            self.logger.info("Selected %d features for fold %d", len(selected_features), fold_idx)
            
            # Train models
            fold_results = {
                'fold': fold_idx,
                'selected_features': selected_features,
                'models': {}
            }
            
            for model_name, model_config in models_config.items():
                self.logger.info("Training %s on fold %d", model_name, fold_idx)
                
                try:
                    # Create and train model
                    model = ModelFactory.create_model(model_name, model_config)
                    model.fit(X_train_selected, y_train)
                    
                    # Evaluate on validation set
                    y_val_pred = model.predict(X_val_selected)
                    y_val_proba = model.predict_proba(X_val_selected)
                    if len(y_val_proba.shape) > 1 and y_val_proba.shape[1] > 1:
                        y_val_proba = y_val_proba[:, 1]
                    
                    # Calculate metrics
                    metrics = calculate_metrics(y_val.values, y_val_pred, y_val_proba)
                    
                    fold_results['models'][model_name] = {
                        'metrics': metrics,
                        'y_true': y_val.tolist(),
                        'y_pred': y_val_pred.tolist(),
                        'y_pred_proba': y_val_proba.tolist()
                    }
                    
                    self.logger.info("%s fold %d - Val AUC: %.4f, Acc: %.4f", model_name, fold_idx, metrics['auc'], metrics['accuracy'])
                    
                except Exception as e:
                    error_msg = f"Error training {model_name} on fold {fold_idx}: {e}"
                    self.logger.error(error_msg)
            
            self.cv_results['folds'].append(fold_results)
        
        # Aggregate results across folds
        self._aggregate_results()
        
        return self

    def _aggregate_results(self) -> None:
        """
        Aggregate results across all folds
        """
        self.logger.info("Aggregating results across folds")
        
        models_config = self.config.get('models', {})
        
        for model_name in models_config.keys():
            # Collect metrics from all folds
            fold_metrics = []
            all_y_true = []
            all_y_pred = []
            all_y_pred_proba = []
            
            for fold_result in self.cv_results['folds']:
                if model_name in fold_result['models']:
                    fold_metrics.append(fold_result['models'][model_name]['metrics'])
                    all_y_true.extend(fold_result['models'][model_name]['y_true'])
                    all_y_pred.extend(fold_result['models'][model_name]['y_pred'])
                    all_y_pred_proba.extend(fold_result['models'][model_name]['y_pred_proba'])
            
            if not fold_metrics:
                continue
            
            # Calculate mean and std for each metric
            aggregated = {}
            metric_names = fold_metrics[0].keys()
            
            for metric_name in metric_names:
                values = [m[metric_name] for m in fold_metrics]
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
            
            # Calculate overall metrics using all predictions
            overall_metrics = calculate_metrics(
                np.array(all_y_true),
                np.array(all_y_pred),
                np.array(all_y_pred_proba)
            )
            
            self.cv_results['aggregated'][model_name] = {
                'fold_metrics': aggregated,
                'overall_metrics': overall_metrics,
                'all_predictions': {
                    'y_true': all_y_true,
                    'y_pred': all_y_pred,
                    'y_pred_proba': all_y_pred_proba
                }
            }
            
            self.logger.info("%s - Overall AUC: %.4f ± %.4f", model_name, overall_metrics['auc'], aggregated['auc_std'])
        
        # Save results
        self._save_cv_results()

    def _save_cv_results(self) -> None:
        """
        Save cross-validation results to files
        """
        self.logger.info("Saving cross-validation results")
        
        # Save detailed results to JSON
        results_path = os.path.join(self.output_dir, 'kfold_cv_results.json')
        
        # Prepare results for JSON serialization
        results_to_save = {
            'n_splits': self.n_splits,
            'stratified': self.stratified,
            'aggregated': self.cv_results['aggregated']
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=4)
        
        self.logger.info("Saved k-fold CV results to: %s", results_path)
        
        # Save performance summary table
        self._save_performance_summary()
        
        # Save predictions in format compatible with model comparison tool
        self._save_predictions_for_comparison()
        
        # Generate visualization plots if enabled
        if self.is_visualize:
            self._generate_visualizations()

    def _save_performance_summary(self) -> None:
        """
        Save a summary table of model performance across folds
        """
        summary_data = []
        
        for model_name, results in self.cv_results['aggregated'].items():
            fold_metrics = results['fold_metrics']
            overall_metrics = results['overall_metrics']
            
            summary_data.append({
                'Model': model_name,
                'AUC_mean': fold_metrics['auc_mean'],
                'AUC_std': fold_metrics['auc_std'],
                'AUC_overall': overall_metrics['auc'],
                'Accuracy_mean': fold_metrics['accuracy_mean'],
                'Accuracy_std': fold_metrics['accuracy_std'],
                'Accuracy_overall': overall_metrics['accuracy'],
                'Sensitivity_mean': fold_metrics['sensitivity_mean'],
                'Sensitivity_std': fold_metrics['sensitivity_std'],
                'Specificity_mean': fold_metrics['specificity_mean'],
                'Specificity_std': fold_metrics['specificity_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'kfold_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        self.logger.info("Saved performance summary to: %s", summary_path)

    def _save_predictions_for_comparison(self) -> None:
        """
        Save all predictions in a format compatible with the model comparison tool
        
        This generates an all_prediction_results.csv file similar to the standard ml pipeline,
        which can be directly used by the model comparison tool.
        """
        self.logger.info("Saving predictions in comparison-compatible format")
        
        try:
            # Create a dataframe with subject IDs from the original data
            if hasattr(self, 'subject_id_col') and self.subject_id_col in self.data.columns:
                results_df = pd.DataFrame({
                    self.subject_id_col: self.data[self.subject_id_col].values
                })
            else:
                # If no subject_id_col, use index
                results_df = pd.DataFrame({
                    'subject_id': self.data.index.values
                })
            
            # Add true labels
            results_df['true_label'] = self.y.values
            
            # Add a column to indicate this is test/validation data (compatible with compare tool)
            results_df['split'] = 'Test set'
            
            # For each model, add predictions and probabilities
            for model_name, model_results in self.cv_results['aggregated'].items():
                all_predictions = model_results['all_predictions']
                
                # Add predictions
                results_df[f'{model_name}_pred'] = all_predictions['y_pred']
                results_df[f'{model_name}_prob'] = all_predictions['y_pred_proba']
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, 'all_prediction_results.csv')
            results_df.to_csv(output_path, index=False)
            
            self.logger.info("Saved comparison-compatible predictions to: %s", output_path)
            
        except Exception as e:
            warning_msg = f"Warning: Failed to save comparison-compatible predictions: {e}"
            self.logger.warning(warning_msg)

    def _generate_visualizations(self) -> None:
        """
        Generate visualization plots for K-fold cross-validation results
        
        Generates ROC curves, calibration curves, confusion matrices, and DCA curves
        for all models using the aggregated predictions across all folds.
        """
        self.logger.info("Generating visualization plots")
        
        try:
            # Prepare data in the format expected by Plotter
            models_data = {}
            
            for model_name, model_results in self.cv_results['aggregated'].items():
                all_predictions = model_results['all_predictions']
                
                # Format: (y_true, y_pred_proba)
                # Convert lists to numpy arrays for plotting functions
                models_data[model_name] = (
                    np.array(all_predictions['y_true']),
                    np.array(all_predictions['y_pred_proba'])
                )
            
            # Generate ROC curves
            self.logger.info("Generating ROC curves")
            try:
                self.plotter.plot_roc_v2(
                    models_data, 
                    save_name='kfold_roc_curves.pdf',
                    title='K-Fold Cross-Validation ROC Curves'
                )
            except Exception as e:
                warning_msg = f"Warning: Failed to generate ROC curves: {e}"
                self.logger.warning(warning_msg)
            
            # Generate calibration curves
            self.logger.info("Generating calibration curves")
            try:
                self.plotter.plot_calibration_v2(
                    models_data,
                    save_name='kfold_calibration_curves.pdf',
                    title='K-Fold Cross-Validation Calibration Curves'
                )
            except Exception as e:
                warning_msg = f"Warning: Failed to generate calibration curves: {e}"
                self.logger.warning(warning_msg)
            
            # Generate DCA curves
            self.logger.info("Generating DCA curves")
            try:
                self.plotter.plot_dca_v2(
                    models_data,
                    save_name='kfold_dca_curves.pdf',
                    title='K-Fold Cross-Validation Decision Curve Analysis'
                )
            except Exception as e:
                warning_msg = f"Warning: Failed to generate DCA curves: {e}"
                self.logger.warning(warning_msg)
            
            # Generate confusion matrices for each model
            self.logger.info("Generating confusion matrices")
            for model_name, model_results in self.cv_results['aggregated'].items():
                try:
                    all_predictions = model_results['all_predictions']
                    # Convert lists to numpy arrays
                    y_true = np.array(all_predictions['y_true'])
                    y_pred = np.array(all_predictions['y_pred'])
                    
                    self.plotter.plot_confusion_matrix(
                        y_true,
                        y_pred,
                        save_name=f'kfold_confusion_matrix_{model_name}.pdf',
                        title=f'{model_name} - K-Fold CV Confusion Matrix'
                    )
                except Exception as e:
                    warning_msg = f"Warning: Failed to generate confusion matrix for {model_name}: {e}"
                    self.logger.warning(warning_msg)
            
            self.logger.info("All visualization plots generated successfully")
            
        except Exception as e:
            error_msg = f"Error generating visualizations: {e}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(traceback.format_exc())

    def run_pipeline(self) -> 'ModelingKFold':
        """
        Run the complete k-fold cross-validation pipeline
        
        Returns:
            ModelingKFold: Self instance for method chaining
        """
        self.logger.info("Starting K-Fold Cross-Validation Pipeline")
        
        self.read_data()
        self.preprocess_data()
        self._create_kfold_splits()
        self.run_kfold_cv()
        
        self.logger.info("K-Fold Cross-Validation Pipeline Completed")
        
        return self


