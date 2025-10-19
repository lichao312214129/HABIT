# Machine Learning Models Guide

This document describes the various classification models supported in the habit machine learning module and how to use them.

## Overview

The habit machine learning module provides a rich set of classification models, including:
- Linear models (Logistic Regression)
- Support Vector Machines (SVM)
- Tree-based models (Decision Tree, Random Forest, XGBoost, GradientBoosting, AdaBoost)
- Distance-based models (KNN)
- Neural networks (MLP)
- Probabilistic models (Naive Bayes family)
- Automated Machine Learning (AutoGluon)

## Available Models

### 1. Logistic Regression

**Use cases**: Linearly separable problems, independent features, need for interpretability

**Configuration example**:
```yaml
LogisticRegression:
  params:
    C: 1.0                    # Regularization strength
    penalty: "l2"             # l1, l2, elasticnet, none
    solver: "lbfgs"           # lbfgs, liblinear, newton-cg, sag, saga
    max_iter: 1000            # Maximum iterations
    random_state: 42
    class_weight: "balanced"  # Handle class imbalance
```

**Advantages**:
- Fast training
- Highly interpretable
- Supports feature importance analysis

**Disadvantages**:
- Only handles linearly separable problems
- Requires good feature engineering

---

### 2. Support Vector Machine (SVM)

**Use cases**: Small samples, high-dimensional data, non-linear problems

**Configuration example**:
```yaml
SVM:
  params:
    C: 1.0                    # Regularization parameter
    kernel: "rbf"             # linear, poly, rbf, sigmoid
    gamma: "scale"            # Kernel coefficient
    probability: true         # Enable probability estimates
    random_state: 42
    class_weight: "balanced"
```

**Advantages**:
- Excellent performance in high-dimensional space
- Suitable for small sample problems
- Handles non-linearity via kernel functions

**Disadvantages**:
- Relatively slow training
- Sensitive to parameters
- No feature importance

---

### 3. Random Forest

**Use cases**: General classification, need feature importance, robust to overfitting

**Configuration example**:
```yaml
RandomForest:
  params:
    n_estimators: 100         # Number of trees
    max_depth: null           # Maximum depth (null = unlimited)
    min_samples_split: 2      # Min samples to split
    min_samples_leaf: 1       # Min samples in leaf
    max_features: "sqrt"      # Features per split
    random_state: 42
    class_weight: "balanced"
```

**Advantages**:
- Resistant to overfitting
- Provides feature importance
- Robust to missing values and outliers
- No need for feature scaling

**Disadvantages**:
- Large model size
- Relatively slow prediction

---

### 4. XGBoost

**Use cases**: Competitions, high performance needs, structured data

**Configuration example**:
```yaml
XGBoost:
  params:
    n_estimators: 100         # Number of trees
    max_depth: 3              # Maximum depth
    learning_rate: 0.1        # Learning rate
    subsample: 0.8            # Sample ratio
    colsample_bytree: 0.8     # Feature ratio
    random_state: 42
    objective: "binary:logistic"
    eval_metric: "logloss"
```

**Advantages**:
- Usually excellent performance
- Provides feature importance
- Supports parallel computation
- Built-in regularization

**Disadvantages**:
- Many parameters, complex tuning
- May overfit

---

### 5. K-Nearest Neighbors (KNN)

**Use cases**: Simple classification, small datasets, non-parametric method

**Configuration example**:
```yaml
KNN:
  params:
    n_neighbors: 5            # Number of neighbors
    weights: "uniform"        # uniform or distance
    algorithm: "auto"         # auto, ball_tree, kd_tree, brute
    metric: "minkowski"       # Distance metric
    p: 2                      # Power for Minkowski
```

**Advantages**:
- Simple and intuitive
- No training required
- Suitable for irregular decision boundaries

**Disadvantages**:
- Slow prediction
- Sensitive to feature scales (needs scaling)
- No feature importance
- Poor performance on high-dimensional data

---

### 6. Multi-layer Perceptron (MLP)

**Use cases**: Complex non-linear problems, large datasets

**Configuration example**:
```yaml
MLP:
  params:
    hidden_layer_sizes: [100, 50]  # Hidden layer sizes
    activation: "relu"             # relu, tanh, logistic
    solver: "adam"                 # adam, sgd, lbfgs
    alpha: 0.0001                  # L2 penalty
    learning_rate: "constant"      # constant, adaptive
    learning_rate_init: 0.001      # Initial learning rate
    max_iter: 200                  # Maximum iterations
    random_state: 42
    early_stopping: false          # Use early stopping
```

**Advantages**:
- Can learn complex non-linear relationships
- Suitable for large-scale data

**Disadvantages**:
- Long training time
- Requires large amounts of data
- Sensitive to parameters
- Black box model, poor interpretability

---

### 7. Naive Bayes

#### 7.1 Gaussian Naive Bayes

**Use cases**: Continuous features, assumes Gaussian distribution

**Configuration example**:
```yaml
GaussianNB:
  params:
    var_smoothing: 1.0e-9     # Variance smoothing
```

#### 7.2 Multinomial Naive Bayes

**Use cases**: Discrete features, text classification, count data

**Configuration example**:
```yaml
MultinomialNB:
  params:
    alpha: 1.0                # Smoothing parameter
    fit_prior: true           # Learn class prior
```

**Note**: Requires non-negative feature values

#### 7.3 Bernoulli Naive Bayes

**Use cases**: Binary features, boolean data

**Configuration example**:
```yaml
BernoulliNB:
  params:
    alpha: 1.0                # Smoothing parameter
    binarize: 0.0             # Binarization threshold
    fit_prior: true           # Learn class prior
```

**Naive Bayes Family Advantages**:
- Extremely fast training
- Good for small datasets
- Highly interpretable

**Naive Bayes Family Disadvantages**:
- Assumes feature independence (often unrealistic)
- Sensitive to feature correlations

---

### 8. Gradient Boosting

**Use cases**: General classification, high performance needs

**Configuration example**:
```yaml
GradientBoosting:
  params:
    n_estimators: 100         # Number of boosting stages
    learning_rate: 0.1        # Learning rate
    max_depth: 3              # Maximum depth
    subsample: 1.0            # Sample fraction
    min_samples_split: 2      # Min samples to split
    random_state: 42
```

**Advantages**:
- Excellent performance
- Provides feature importance
- High flexibility

**Disadvantages**:
- Slow training
- Prone to overfitting
- Many parameters

---

### 9. AdaBoost

**Use cases**: Weak learner ensembling, need robustness

**Configuration example**:
```yaml
AdaBoost:
  params:
    n_estimators: 50          # Number of weak learners
    learning_rate: 1.0        # Learning rate
    algorithm: "SAMME.R"      # SAMME or SAMME.R
    random_state: 42
```

**Advantages**:
- Simple and effective
- Not very sensitive to parameters
- Provides feature importance

**Disadvantages**:
- Sensitive to noise and outliers
- May overfit

---

### 10. Decision Tree

**Use cases**: Need high interpretability, base learner

**Configuration example**:
```yaml
DecisionTree:
  params:
    criterion: "gini"         # gini or entropy
    splitter: "best"          # best or random
    max_depth: null           # Maximum depth
    min_samples_split: 2      # Min samples to split
    min_samples_leaf: 1       # Min samples in leaf
    random_state: 42
    class_weight: "balanced"
```

**Advantages**:
- Highly interpretable
- No need for feature scaling
- Can handle non-linear relationships
- Provides feature importance

**Disadvantages**:
- Prone to overfitting
- Sensitive to data variations
- Single tree limited performance

---

## Model Selection Guidelines

### By Task Type

1. **Linearly separable problems**: Logistic Regression
2. **Small sample problems**: SVM, Naive Bayes
3. **Large sample problems**: MLP, XGBoost, Random Forest
4. **Need interpretability**: Logistic Regression, Decision Tree, Naive Bayes
5. **High performance**: XGBoost, GradientBoosting, Random Forest
6. **Quick prototyping**: KNN, GaussianNB, Decision Tree

### By Data Characteristics

1. **High-dimensional data**: SVM, Logistic Regression
2. **Low-dimensional data**: KNN, Decision Tree
3. **Non-linear data**: SVM (RBF kernel), MLP, tree models
4. **Class imbalance**: Models with `class_weight='balanced'` parameter
5. **Missing values**: Tree models (RandomForest, XGBoost)
6. **Need feature scaling**: KNN, SVM, MLP, Logistic Regression
7. **No scaling needed**: Tree models, Naive Bayes

### Combination Strategy

Train multiple models simultaneously for comparison:

```yaml
models:
  LogisticRegression:
    params:
      C: 1.0
      random_state: 42
  
  RandomForest:
    params:
      n_estimators: 100
      random_state: 42
  
  XGBoost:
    params:
      n_estimators: 100
      random_state: 42
```

## Usage Examples

### Basic Usage

```python
from habit.core.machine_learning.machine_learning import Modeling
import yaml

# Load configuration
with open('config_machine_learning.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run modeling pipeline
modeling = Modeling(config)
modeling.read_data()\
    .preprocess_data()\
    ._split_data()\
    .feature_selection_before_normalization()\
    .normalization()\
    .feature_selection()\
    .modeling()\
    .evaluate_models()
```

### K-Fold Cross-Validation

```python
from habit.core.machine_learning.machine_learning_kfold import run_kfold_modeling
import yaml

# Load configuration
with open('config_machine_learning_kfold.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run k-fold CV
modeling = run_kfold_modeling(config)
```

## Feature Importance Analysis

Models supporting feature importance:
- Logistic Regression (coefficient-based)
- Random Forest (Gini importance-based)
- XGBoost (gain-based)
- GradientBoosting (split improvement-based)
- AdaBoost (weak learner importance-based)
- Decision Tree (Gini importance or information gain-based)

Models not supporting feature importance:
- SVM
- KNN
- MLP
- Naive Bayes family

## Important Notes

1. **Data Preprocessing**:
   - KNN, SVM, MLP need feature scaling
   - MultinomialNB needs non-negative features
   - Tree models don't need feature scaling

2. **Class Imbalance**:
   - Use `class_weight='balanced'` parameter
   - Or use sampling techniques (SMOTE, etc.)

3. **Hyperparameter Tuning**:
   - Use grid search (GridSearchCV)
   - Or Bayesian optimization (Optuna, etc.)

4. **Overfitting Control**:
   - Use regularization (L1/L2)
   - Limit model complexity (max_depth, etc.)
   - Use cross-validation
   - Increase training data

5. **Computational Resources**:
   - Avoid KNN for large datasets
   - Use simple models with limited resources
   - Use parallel computation (n_jobs=-1)

## FAQ

### Q: Which model should I choose?
A: There is no best model, only the most suitable one. Recommendations:
   1. Start with simple models (Logistic Regression)
   2. Try tree models (Random Forest, XGBoost)
   3. Select best model based on results
   4. Use k-fold cross-validation for evaluation

### Q: How to handle class imbalance?
A: Options:
   1. Use `class_weight='balanced'`
   2. Use sampling techniques like SMOTE
   3. Use appropriate metrics (AUC, F1, etc.)

### Q: What if model training is too slow?
A: Solutions:
   1. Reduce number of features
   2. Reduce number of samples
   3. Use faster models
   4. Use parallel computation
   5. Reduce model complexity

### Q: How to improve model performance?
A: Strategies:
   1. Increase training data
   2. Feature engineering
   3. Hyperparameter tuning
   4. Model ensembling
   5. Use more complex models

## Related Documentation

- [Machine Learning Configuration Guide](app_of_machine_learning.md)
- [Feature Selection Methods](../habit/core/machine_learning/feature_selectors/README.md)
- [Model API Documentation](../habit/core/machine_learning/models/README.md)

