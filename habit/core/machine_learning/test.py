from autogluon.tabular import TabularPredictor
import pandas as pd
import sklearn
import shap
# make data
from sklearn.datasets import make_classification
# 导入sklearn的逻辑回归模型
from sklearn.linear_model import LogisticRegression
# 导入交叉验证评分
from sklearn.model_selection import cross_val_score
# 导入sklearn的评估
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# pip install shap
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')  # 注释掉非交互式设置，允许图形显示

import warnings
warnings.filterwarnings('ignore')


X,y = make_classification(n_samples=1000, n_features=200, n_classes=2, n_informative=20, n_redundant=100, random_state=1)
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=7)

# to dataframe
X_train = pd.DataFrame(X_train, columns=['feature'+str(i) for i in range(X_train.shape[1])])
X_valid = pd.DataFrame(X_valid, columns=['feature'+str(i) for i in range(X_valid.shape[1])])


import warnings
warnings.filterwarnings('ignore')


label = 'label'
feature_names = X_train.columns
train_data = X_train.copy()
train_data[label] = y_train
val_data = X_valid.copy()


predictor = TabularPredictor(label=label, problem_type='binary')\
    .fit(train_data, time_limit=5)

print("positive class:", predictor.positive_class)

# 在val中的预测
y_pred = predictor.predict(val_data)

# 评估
val_data[label] = y_valid
predictor.evaluate(val_data)

# 构建lr模型
lr = LogisticRegression()
lr.fit(X_train, y_train)

# 评估
y_pred = lr.predict(X_valid)

# 计算准确率
accuracy = accuracy_score(y_valid, y_pred)

# 计算精确率
precision = precision_score(y_valid, y_pred)

# 计算召回率
recall = recall_score(y_valid, y_pred)

# 计算F1分数
f1 = f1_score(y_valid, y_pred)

# auc
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_valid, y_pred)





class AutogluonWrapper:
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.ag_model.predict_proba(X, as_multiclass=False)



med = X_train.median()  # X_train.mode() would be a more appropriate baseline for ordinally-encoded categorical features
print("Baseline feature-values: \n", med)



ag_wrapper = AutogluonWrapper(predictor, feature_names)
explainer = shap.KernelExplainer(ag_wrapper.predict_binary_prob, med)

NSHAP_SAMPLES = 100  # how many samples to use to approximate each Shapely value, larger values will be slower
N_VAL = 30  # how many datapoints from validation data should we interpret predictions for, larger values will be slower

ROW_INDEX = 0  # index of an example datapoint
single_datapoint = X_train.iloc[[ROW_INDEX]]
single_prediction = ag_wrapper.predict_binary_prob(single_datapoint)

shap_values_single = explainer.shap_values(single_datapoint, nsamples=NSHAP_SAMPLES)
shap.force_plot(explainer.expected_value, shap_values_single, X_train.iloc[ROW_INDEX,:])

shap_values = explainer.shap_values(X_valid.iloc[0:N_VAL,:], nsamples=NSHAP_SAMPLES)
shap.force_plot(explainer.expected_value, shap_values, X_valid.iloc[0:N_VAL,:])

# 绘制SHAP汇总图
plt.figure(figsize=(10, 8))  # 设置图形大小
shap.summary_plot(shap_values, X_valid.iloc[0:N_VAL,:])
plt.tight_layout()  # 调整布局
plt.savefig("shap_summary_plot.png")  # 保存图形到文件
plt.show()  # 显示图形
plt.close()  # 关闭图形

# 修复dependence_plot的特征名称
plt.figure(figsize=(10, 6))  # 设置图形大小
first_feature = X_valid.columns[0]  
shap.dependence_plot(first_feature, shap_values, X_valid.iloc[0:N_VAL,:])
plt.tight_layout()
plt.savefig("shap_dependence_plot.png")  # 保存图形到文件
plt.show()  # 显示图形
plt.close()  # 关闭图形


val_data[label] = y_valid  # add labels to validation DataFrame
predictor.feature_importance(val_data)

