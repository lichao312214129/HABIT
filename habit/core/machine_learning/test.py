from autogluon.tabular import TabularPredictor
from tpot import TPOTClassifier
import pandas as pd
import sklearn
import re
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from IPython.display import display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


file_x = r'F:\work\research\radiomics_TLSs\data\results_365_v2\parsed_features\msi_features.csv'
file_x1 = r'F:\work\research\radiomics_TLSs\data\results_365_v2\parsed_features\habitat_basic_features.csv'
file_x2 = r'F:\work\research\radiomics_TLSs\data\results_365_v2\parsed_features\ith_scores.csv'
file_x3 = r'F:\work\research\radiomics_TLSs\data\results_365_v2\parsed_features\raw_image_radiomics.csv'
file_y = r'F:\work\research\radiomics_TLSs\data\PathologyInfo_new.xlsx'

X = pd.read_csv(file_x, index_col=0)
X1 = pd.read_csv(file_x1, index_col=0)
X2 = pd.read_csv(file_x2, index_col=0)
X3 = pd.read_csv(file_x3, index_col=0)

# 把X的index转换为int，第一个连续的数字是id
X.index = X.index.map(lambda x: int(re.search(r'\d+', x).group()))
X1.index = X1.index.map(lambda x: int(re.search(r'\d+', x).group()))
X2.index = X2.index.map(lambda x: int(re.search(r'\d+', x).group()))
X3.index = X3.index.map(lambda x: int(re.search(r'\d+', x).group()))

y = pd.read_excel(file_y, index_col=0)

y.index = y.index.map(lambda x: int(x))

its = y.index.intersection(X.index)
print(len(its))
# 不相同的index是哪些
print(set(y.index) - set(its))
print(set(X.index) - set(its))

# Merge X and y based on index to align TLS data
# Ensure only subjects with both feature and label data are kept
X_merged = X.loc[its]
X_merged1 = X1.loc[its]
X_merged2 = X2.loc[its]
X_merged3 = X3.loc[its]

y_merged = y.loc[its]

merged_data = pd.concat([X_merged, X_merged1, X_merged2, y_merged["TLSs"]], axis=1)

# TLSs to label
merged_data["label"] = merged_data["TLSs"].apply(lambda x: 1 if x > 0 else 0)

# 删除TLSs列
merged_data = merged_data.drop(columns=["TLSs"])

# Verify the merge was successful
print(f"Number of subjects after merging: {len(X_merged)}")
print(f"Merged X shape: {X_merged.shape}")
print(f"Merged y shape: {y_merged.shape}")

#%%
# 找到merged_data中哪些列名是重复的
print(merged_data.columns.duplicated())
# 删除重复的列
merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

# 缺失值大于70%的列
missing_rate = merged_data.isnull().sum() / len(merged_data)
missing_rate = missing_rate[missing_rate > 0.1]
print(missing_rate)
# 删除缺失值大于70%的列
merged_data = merged_data.drop(columns=missing_rate.index)

# fillna
merged_data = merged_data.fillna(merged_data.median())

# save to csv
merged_data.to_csv("./ml_data/merged_data.csv", index=True)


data_train, data_valid = sklearn.model_selection.train_test_split(merged_data, test_size=0.2, random_state=7)
X_train = data_train.drop(columns=["label"])
y_train = data_train["label"]
X_valid = data_valid.drop(columns=["label"])
y_valid = data_valid["label"]


#%% 使用autogluon进行训练
label = 'label'
feature_names = X_train.columns
train_data = X_train.copy()
train_data[label] = y_train
val_data = X_valid.copy()

predictor = TabularPredictor(label=label, problem_type='binary').fit(train_data, time_limit=40)

print("positive class:", predictor.positive_class)

leaderboard = predictor.leaderboard(silent=False)  # 显示所有模型的详细信息
print(leaderboard)

# 在val中的预测
y_pred = predictor.predict(val_data)

# 评估
val_data[label] = y_valid
predictor.evaluate(val_data)
predictor.evaluate(train_data)

# 特征重要性
predictor.feature_importance(val_data)


#%% TPOT
# Initialize and run TPOT with custom configuration
tpot = TPOTClassifier(
    generations=2,
    population_size=5,
    verbosity=2,
    random_state=42,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc',
    # max_time_mins=2,  # Limit runtime to 5 minutes for example
    periodic_checkpoint_folder='tpot_checkpoints',
    use_dask=False,
)

# Train TPOT
tpot.fit(X_train, y_train)

# Export the best pipeline
tpot.export('tpot_pipeline.py')
print("\n最佳流水线已导出到 'tpot_pipeline.py'")

# 查看tpot的pipeline
print(tpot.__dict__['_config_dict'].keys())
tpot.fitted_pipeline_

for idx, (pipe, score) in enumerate(tpot.evaluated_individuals_.items()):
    print(f"Pipeline {idx}: {pipe}\nScore: {score}\n")

# Make predictions
y_pred = tpot.predict(X_valid)
y_pred_proba = tpot.predict_proba(X_valid)[:, 1]

# Evaluation
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred)
recall = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
auc = roc_auc_score(y_valid, y_pred_proba)

print(f"\n模型性能:")
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("\n分类报告:")
print(classification_report(y_valid, y_pred))



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

shap.initjs()
shap_values_single = explainer.shap_values(single_datapoint, nsamples=NSHAP_SAMPLES)
shap.force_plot(explainer.expected_value, shap_values_single, X_train.iloc[ROW_INDEX,:])
plt.show()

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

