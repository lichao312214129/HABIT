机器学习集成
==============

本教程将教你如何将 habitat 分析与机器学习结合。

从 Habitat 特征到预测模型
------------------------------

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report
   
   # 加载 habitat 特征
   habitat_features = pd.read_csv('output/habitat_features.csv')
   
   # 准备数据
   X = habitat_features.drop(['subject_id', 'label'], axis=1)
   y = habitat_features['label']
   
   # 划分训练集和测试集
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   
   # 训练模型
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   
   # 评估
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))

使用 sklearn Pipeline
-------------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler
   from sklearn.svm import SVC
   
   # 创建 pipeline
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('svm', SVC(kernel='rbf', C=1.0))
   ])
   
   # 训练
   pipeline.fit(X_train, y_train)
   
   # 预测
   y_pred = pipeline.predict(X_test)

交叉验证
----------

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   
   # 5 折交叉验证
   scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
   print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

超参数调优
----------

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   
   # 定义参数网格
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [None, 10, 20],
       'min_samples_split': [2, 5, 10]
   }
   
   # 网格搜索
   grid_search = GridSearchCV(
       RandomForestClassifier(random_state=42),
       param_grid,
       cv=5,
       scoring='accuracy'
   )
   
   grid_search.fit(X_train, y_train)
   
   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best score: {grid_search.best_score_:.3f}")

保存和加载模型
---------------

.. code-block:: python

   import joblib
   
   # 保存模型
   joblib.dump(model, 'habitat_predictor.pkl')
   
   # 加载模型
   loaded_model = joblib.load('habitat_predictor.pkl')
   
   # 使用加载的模型预测
   predictions = loaded_model.predict(X_new)

下一步
--------

* 查看 `基础教程 <basic_habitat_analysis.html>`_
* 学习 `自定义特征提取 <custom_features.html>`_
