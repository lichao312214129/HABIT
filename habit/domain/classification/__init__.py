# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Built-in classifiers and the ``classifier`` registry."""

from __future__ import annotations

from habit.domain.classification.autogluon import (
    AutogluonTabularClassifier,
    AutogluonTabularClassifierParams,
)
from habit.domain.classification.models import (
    AdaboostClassifier,
    AdaboostClassifierParams,
    BernoulliNbClassifier,
    BernoulliNbClassifierParams,
    DecisionTreeClassifier,
    DecisionTreeClassifierParams,
    GaussianNbClassifier,
    GaussianNbClassifierParams,
    GradientBoostingClassifier,
    GradientBoostingClassifierParams,
    KnnClassifier,
    KnnClassifierParams,
    LogisticRegressionClassifier,
    LogisticRegressionClassifierParams,
    MlpClassifier,
    MlpClassifierParams,
    MultinomialNbClassifier,
    MultinomialNbClassifierParams,
    RandomForestClassifier,
    RandomForestClassifierParams,
    SvcClassifier,
    SvcClassifierParams,
    SvmClassifier,
    SvmClassifierParams,
    XgboostClassifier,
    XgboostClassifierParams,
)
from habit.domain.classification.registry import ClassifierRegistry

__all__ = [
    "ClassifierRegistry",
    "DecisionTreeClassifier",
    "DecisionTreeClassifierParams",
    "KnnClassifier",
    "KnnClassifierParams",
    "SvmClassifier",
    "SvmClassifierParams",
    "SvcClassifier",
    "SvcClassifierParams",
    "MlpClassifier",
    "MlpClassifierParams",
    "LogisticRegressionClassifier",
    "LogisticRegressionClassifierParams",
    "RandomForestClassifier",
    "RandomForestClassifierParams",
    "GradientBoostingClassifier",
    "GradientBoostingClassifierParams",
    "XgboostClassifier",
    "XgboostClassifierParams",
    "AdaboostClassifier",
    "AdaboostClassifierParams",
    "GaussianNbClassifier",
    "GaussianNbClassifierParams",
    "MultinomialNbClassifier",
    "MultinomialNbClassifierParams",
    "BernoulliNbClassifier",
    "BernoulliNbClassifierParams",
    "AutogluonTabularClassifier",
    "AutogluonTabularClassifierParams",
]
