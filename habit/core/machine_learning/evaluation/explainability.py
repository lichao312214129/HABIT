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
"""
Model-agnostic explanation measures computed from a fitted estimator.

SHAP explains the model that sits at the end of the pipeline, on the features
as the model sees them after scaling and selection. Permutation importance is
complementary: it is measured on the RAW input columns through the complete
pipeline, so it answers "how much does the reported performance depend on this
input variable" rather than "how does the final estimator weigh this
transformed feature". Reporting both is what lets a reader distinguish a
feature the model relies on from one it merely correlates with.
"""

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    scoring: str = 'roc_auc',
    n_repeats: int = 10,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Measure how much each input feature contributes to a fitted model's score.

    Args:
        estimator: Fitted estimator or pipeline accepting ``X`` as given.
        X: Feature matrix in the exact column order used at fit time.
        y: Ground-truth labels aligned with ``X`` rows.
        scoring: sklearn scoring name used as the reference metric.
        n_repeats: Number of shuffles per feature. More repeats shrink the
            standard deviation of the estimate but cost proportionally more
            model evaluations.
        random_state: Seed controlling the shuffles, for reproducible tables.
        n_jobs: Parallel workers passed to sklearn.

    Returns:
        pd.DataFrame: Columns ``feature``, ``importance_mean`` and
        ``importance_std``, sorted by descending mean importance. A value near
        zero means shuffling the feature did not degrade the score; a negative
        value means the shuffled version scored better, which happens for
        uninformative features and should not be read as a protective effect.

    Raises:
        ValueError: If ``X`` and ``y`` lengths disagree.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X has {len(X)} rows but y has {len(y)} values; they must match."
        )

    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=int(n_repeats),
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance = pd.DataFrame(
        {
            'feature': list(X.columns),
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        }
    )
    return importance.sort_values(
        'importance_mean', ascending=False, ignore_index=True
    )


def top_features(
    importance: pd.DataFrame,
    top_k: int = 10,
) -> Sequence[str]:
    """
    Return the highest-ranked feature names from an importance table.

    Args:
        importance: Table produced by :func:`compute_permutation_importance`.
        top_k: Number of names to return.

    Returns:
        Sequence[str]: Feature names in descending importance order.
    """
    return list(importance.head(max(int(top_k), 0))['feature'])
