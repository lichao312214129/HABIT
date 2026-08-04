Table ML domain API
===================

Tabular preprocessing, feature selection, classification, metrics, and
``TablePipeline``. All components follow ``Registry.create(name, **params)``.

.. code-block:: python

   from habit.domain import (
       ClassifierRegistry,
       FeatureSelectorRegistry,
       MetricRegistry,
       TablePipeline,
       TablePreprocessorRegistry,
   )

Table preprocessors
-------------------

Domain: ``table_preprocessor``

.. code-block:: python

   from habit.domain import TablePreprocessorRegistry

   z = TablePreprocessorRegistry.create("zscore")
   mm = TablePreprocessorRegistry.create("minmax")
   rob = TablePreprocessorRegistry.create("robust")
   bins = TablePreprocessorRegistry.create("binning")
   win = TablePreprocessorRegistry.create("winsorize")
   log = TablePreprocessorRegistry.create("log")
   vf = TablePreprocessorRegistry.create("variance_filter")
   cf = TablePreprocessorRegistry.create("correlation_filter")

   z.fit(train_table)
   scaled = z.transform(test_table)

Names: ``zscore``, ``minmax``, ``robust``, ``binning``, ``winsorize``,
``log``, ``variance_filter``, ``correlation_filter``.

Feature selectors
-----------------

Domain: ``feature_selector``

.. code-block:: python

   from habit.domain import FeatureSelectorRegistry

   var = FeatureSelectorRegistry.create("variance", threshold=0.01)
   corr = FeatureSelectorRegistry.create("correlation")
   vif = FeatureSelectorRegistry.create("vif")
   anova = FeatureSelectorRegistry.create("anova")
   chi2 = FeatureSelectorRegistry.create("chi2")
   stat = FeatureSelectorRegistry.create("statistical_test")
   uni = FeatureSelectorRegistry.create("univariate_logistic")
   step = FeatureSelectorRegistry.create("stepwise")
   rfecv = FeatureSelectorRegistry.create("rfecv")
   lasso = FeatureSelectorRegistry.create("lasso")
   icc = FeatureSelectorRegistry.create("icc")
   mrmr = FeatureSelectorRegistry.create("mrmr")

   var.fit(train_table)
   reduced = var.transform(train_table)

Names: ``variance``, ``correlation``, ``vif``, ``anova``, ``chi2``,
``statistical_test``, ``univariate_logistic``, ``stepwise``, ``rfecv``,
``lasso``, ``icc``, ``mrmr``.

Classifiers
-----------

Domain: ``classifier`` (not ``model`` — avoids clashing with ``HabitatModel``)

.. code-block:: python

   from habit.domain import ClassifierRegistry

   lr = ClassifierRegistry.create("LogisticRegression", max_iter=500)
   svm = ClassifierRegistry.create("SVM")
   svc = ClassifierRegistry.create("SVC")
   knn = ClassifierRegistry.create("KNN")
   dt = ClassifierRegistry.create("DecisionTree")
   rf = ClassifierRegistry.create("RandomForest")
   gb = ClassifierRegistry.create("GradientBoosting")
   xgb = ClassifierRegistry.create("XGBoost")
   ada = ClassifierRegistry.create("AdaBoost")
   mlp = ClassifierRegistry.create("MLP")
   gnb = ClassifierRegistry.create("GaussianNB")
   mnb = ClassifierRegistry.create("MultinomialNB")
   bnb = ClassifierRegistry.create("BernoulliNB")
   ag = ClassifierRegistry.create("AutoGluonTabular")

   lr.fit(train_table)
   labels = lr.predict(test_table)
   proba = lr.predict_proba(test_table)

Names: ``LogisticRegression``, ``SVM``, ``SVC``, ``KNN``, ``DecisionTree``,
``RandomForest``, ``GradientBoosting``, ``XGBoost``, ``AdaBoost``, ``MLP``,
``GaussianNB``, ``MultinomialNB``, ``BernoulliNB``, ``AutoGluonTabular``.

Metrics
-------

Domain: ``metric``

.. code-block:: python

   from habit.domain import MetricRegistry

   acc = MetricRegistry.create("accuracy")
   sens = MetricRegistry.create("sensitivity")
   spec = MetricRegistry.create("specificity")
   ppv = MetricRegistry.create("ppv")
   npv = MetricRegistry.create("npv")
   f1 = MetricRegistry.create("f1_score")
   auc = MetricRegistry.create("auc")
   hl = MetricRegistry.create("hosmer_lemeshow_p_value", n_groups=10)
   sp = MetricRegistry.create("spiegelhalter_z_p_value")

Names: ``accuracy``, ``sensitivity``, ``specificity``, ``ppv``, ``npv``,
``f1_score``, ``auc``, ``hosmer_lemeshow_p_value``,
``spiegelhalter_z_p_value``.

Statistical helpers
-------------------

.. code-block:: python

   from habit.domain import (
       auc_confidence_interval,
       calibration_tests,
       delong_test,
       icc_analysis,
       repeat_measurement_matrix,
   )

   delong = delong_test(y_true, scores_a, scores_b)
   ci = auc_confidence_interval(y_true, scores)
   cal = calibration_tests(y_true, scores)
   icc_df = icc_analysis(repeat_measurement_matrix(...))

``TablePipeline``
-----------------

.. code-block:: python

   from habit.domain import (
       AccuracyMetric,
       AucMetric,
       ClassifierRegistry,
       FeatureSelectorRegistry,
       LogisticRegressionClassifier,
       TablePipeline,
       TablePreprocessorRegistry,
       VarianceSelector,
       ZScorePreprocessor,
   )

   pipe = TablePipeline(
       steps=[VarianceSelector(threshold=0.01), ZScorePreprocessor()],
       classifier=LogisticRegressionClassifier(max_iter=500),
   )
   pipe.set_random_state(42)
   pipe.fit(train_table)

   y_hat = pipe.predict(test_table)
   proba = pipe.predict_proba(test_table)
   X_ready = pipe.transform(test_table)
   scores = pipe.evaluate(test_table, [AccuracyMetric(), AucMetric()])

   pipe.save("out/table_pipeline.habittable")
   loaded = TablePipeline.load("out/table_pipeline.habittable")

Registry form::

   pipe = TablePipeline(
       steps=[
           FeatureSelectorRegistry.create("variance", threshold=0.01),
           TablePreprocessorRegistry.create("zscore"),
       ],
       classifier=ClassifierRegistry.create(
           "LogisticRegression",
           max_iter=500,
       ),
   )
