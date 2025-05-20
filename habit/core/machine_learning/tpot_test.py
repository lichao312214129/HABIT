"""
TPOT (Tree-based Pipeline Optimization Tool) example with custom feature selection
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from tpot import TPOTClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, chi2
from sklearn.linear_model import LogisticRegression

# from habit.utils.progress_utils import CustomTqdm


class CustomFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom feature selector based on feature importance using mutual information
    
    This is an example implementation that can be used within TPOT pipelines
    """
    
    def __init__(self, k: int = 10, method: str = 'mutual_info'):
        """
        Initialize feature selector
        
        Args:
            k (int): Number of features to select
            method (str): Method for feature selection ('mutual_info', 'chi2', or 'f_test')
        """
        self.k = k
        self.method = method
        self.selected_features_ = None
        self.feature_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomFeatureSelector':
        """
        Fit the feature selector
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target values
            
        Returns:
            self: Fitted selector
        """
        # Get the number of available features
        n_features = X.shape[1]
        
        # Enforce k to be less than or equal to the number of features
        self.k = min(self.k, n_features)
        
        # Import appropriate scoring method based on selected method
        if self.method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            scores = mutual_info_classif(X, y, random_state=42)
        elif self.method == 'chi2':
            from sklearn.feature_selection import chi2
            from sklearn.preprocessing import MinMaxScaler
            # Chi2 requires non-negative features
            X_scaled = MinMaxScaler().fit_transform(X)
            scores, _ = chi2(X_scaled, y)
        elif self.method == 'f_test':
            from sklearn.feature_selection import f_classif
            scores, _ = f_classif(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose 'mutual_info', 'chi2', or 'f_test'")
        
        # Store feature scores
        self.feature_scores_ = scores
        
        # Get indices of the top k features
        self.selected_features_ = np.argsort(scores)[-self.k:]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting only the top k features
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Transformed feature matrix with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")
        
        return X[:, self.selected_features_]
    
    def get_feature_names_out(self, input_features=None):
        """
        Get the feature names after selection
        
        Args:
            input_features: Original feature names (if provided)
            
        Returns:
            list: Names of selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet.")
        
        if input_features is None:
            return [f"feature{i}" for i in self.selected_features_]
        else:
            return [input_features[i] for i in self.selected_features_]


def register_custom_selector():
    """
    Register the custom feature selector with TPOT
    
    This adds our custom selector to TPOT's configuration
    """
    from tpot.config import classifier_config_dict
    
    # 定义自定义选择器
    custom_selector = {
        'selector': CustomFeatureSelector,
        'selector__k': [5, 10, 15, 20],
        'selector__method': ['mutual_info', 'chi2', 'f_test']
    }
    
    # 将自定义选择器添加到TPOT配置的"feature_selection"部分
    if 'feature_selection' not in classifier_config_dict:
        classifier_config_dict['feature_selection'] = {}
    
    # 添加我们的自定义选择器到现有的特征选择部分
    classifier_config_dict['feature_selection']['CustomFeatureSelector'] = custom_selector
    
    # 确保TPOT能够找到这个类
    # 将自定义选择器添加到全局命名空间
    import sys
    module = sys.modules[__name__]
    setattr(module, 'CustomFeatureSelector', CustomFeatureSelector)
    
    return classifier_config_dict


def run_tpot_example():
    """
    Run TPOT example with custom feature selection
    
    This function demonstrates how to use TPOT with a custom feature selector
    """
    # Load breast cancer dataset as an example
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"数据集形状: {X.shape}")
    print(f"目标类别分布: {np.bincount(y)}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 在运行TPOT之前确保自定义选择器已正确注册
    # 确保在当前模块中注册这个类
    from tpot.base import TPOTBase
    TPOTBase._setup_template()
    
    # 注册自定义选择器
    custom_config = register_custom_selector()

    custom_config = {
        # 特征预处理（可选）
        'sklearn.preprocessing': {
            'StandardScaler': {},  # 标准化
            'MinMaxScaler': {},     # 归一化
        },

        # 特征筛选方法
        'sklearn.feature_selection': {
            'VarianceThreshold': {},               # 方差阈值
            'SelectKBest': {                       # 基于统计检验选Top-K特征
                'k': [5, 10, 15],                  # 可选参数值
                'score_func': [f_classif, chi2]    # 评分函数（分类任务）
            },
            'RFE': {                               # 递归特征消除
                'estimator': [LogisticRegression()],
                'n_features_to_select': [0.5, 0.8]
            }
        },

        # 模型定义
        'sklearn.ensemble': {
            'RandomForestClassifier': {            # 随机森林
                'n_estimators': [50, 100],
                'max_depth': [3, 5, None]
            }
        },
        'sklearn.svm': {
            'SVC': {                              # SVM
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
    }
    
    # 检查自定义选择器是否成功注册
    if 'feature_selection' in custom_config and 'CustomFeatureSelector' in custom_config['feature_selection']:
        print("自定义特征选择器已成功注册")
    else:
        print("警告：自定义特征选择器未成功注册")
    
    # Progress bar for tracking
    # progress = CustomTqdm(total=5, desc="TPOT运行中")
    
    # Initialize and run TPOT with custom configuration
    tpot = TPOTClassifier(
        generations=5,
        population_size=20,
        verbosity=2,
        random_state=42,
        config_dict=custom_config,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc',
        max_time_mins=0.5,  # Limit runtime to 5 minutes for example
        periodic_checkpoint_folder='tpot_checkpoints',
        use_dask=False
    )
    
    # Set a callback for progress tracking
    # def update_progress(tpot_obj, gen_num):
    #     progress.update(1)
    
    # tpot._update_val_fitness_callback = update_progress
    
    # Train TPOT
    tpot.fit(X_train, y_train)
    
    # Export the best pipeline
    tpot.export('tpot_pipeline.py')
    print("\n最佳流水线已导出到 'tpot_pipeline.py'")
    
    # Make predictions
    y_pred = tpot.predict(X_test)
    y_pred_proba = tpot.predict_proba(X_test)[:, 1]
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n模型性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # If our custom selector is in the best pipeline, let's examine it
    if hasattr(tpot.fitted_pipeline_, 'steps'):
        # Extract the selector if it exists in the pipeline
        selector = None
        for name, step in tpot.fitted_pipeline_.steps:
            if isinstance(step, CustomFeatureSelector):
                selector = step
                break
        
        # Visualize selected features if our selector was used
        if selector is not None:
            selected_indices = selector.selected_features_
            selected_feature_names = [feature_names[i] for i in selected_indices]
            selected_scores = selector.feature_scores_[selected_indices]
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x=selected_feature_names, y=selected_scores)
            plt.title('Selected Features and Their Importance Scores')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('selected_features.png')
            plt.close()
            
            print(f"\n选中的特征 ({len(selected_indices)}):")
            for name, score in zip(selected_feature_names, selected_scores):
                print(f"- {name}: {score:.4f}")
    
    # Extract and visualize the best pipeline structure
    if hasattr(tpot, 'fitted_pipeline_'):
        pipeline_steps = []
        for name, step in tpot.fitted_pipeline_.steps:
            pipeline_steps.append(f"{name}: {type(step).__name__}")
        
        print("\n最佳流水线结构:")
        for i, step in enumerate(pipeline_steps):
            print(f"{i+1}. {step}")


def standalone_feature_selection_example():
    """
    Standalone example using only the custom feature selector
    
    This demonstrates how the selector can be used outside of TPOT
    """
    # Load breast cancer dataset as an example
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and fit the custom selector
    selector = CustomFeatureSelector(k=10, method='mutual_info')
    selector.fit(X_train, y_train)
    
    # Transform the data
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"原始特征数量: {X_train.shape[1]}")
    print(f"选择后的特征数量: {X_train_selected.shape[1]}")
    
    # Get selected feature names
    selected_indices = selector.selected_features_
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    print("\n选中的特征:")
    for name, score in zip(selected_feature_names, selector.feature_scores_[selected_indices]):
        print(f"- {name}: {score:.4f}")
    
    # Visualize selected features
    plt.figure(figsize=(12, 6))
    sns.barplot(x=selected_feature_names, y=selector.feature_scores_[selected_indices])
    plt.title('Selected Features and Their Importance Scores')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('custom_selected_features.png')
    plt.close()


if __name__ == "__main__":
    print("运行 TPOT 示例...")
    run_tpot_example()
    
    # print("\n\n运行独立特征选择示例...")
    # standalone_feature_selection_example()
