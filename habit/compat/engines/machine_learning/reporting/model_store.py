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
Model persistence helpers for machine-learning workflows.

The :class:`ModelStore` knows how to persist trained estimators in two
shapes:

* **holdout** (:class:`RunResult`): one ``<model>_final_pipeline.pkl`` per
  configured model.
* **k-fold** (:class:`KFoldRunResult`): one
  ``<model>_ensemble_final.pkl`` per configured model, wrapping all per-fold
  estimators inside :class:`HabitEnsembleModel`.

Putting both flows behind a single class keeps the workflow code free of
``joblib.dump`` calls and gives the K-Fold workflow parity with holdout.
"""

from __future__ import annotations

import os
from typing import Dict

import joblib

from ..contracts.results import KFoldRunResult, RunResult
from ..models.ensemble import HabitEnsembleModel


class ModelStore:
    """Save trained estimators from a structured workflow result."""

    def __init__(self, output_dir: str, is_save_model: bool = True) -> None:
        """
        Args:
        output_dir:
            Root directory where artefacts are written; a ``models``
            sub-directory is created automatically.
        is_save_model:
            When ``False`` the store is a no-op (keeps configs without model
            persistence working unchanged).
        """
        self.output_dir = output_dir
        self.is_save_model = is_save_model

    # ------------------------------------------------------------------
    # Holdout
    # ------------------------------------------------------------------

    def save(self, run_result: RunResult) -> Dict[str, str]:
        """
        Save the final pipelines from a holdout run.

        Returns:
        Dict[str, str]
            Mapping ``model_name -> saved_path``.
        """
        if not self.is_save_model:
            return {}

        model_dir = self._ensure_model_dir()

        saved_paths: Dict[str, str] = {}
        for model_name, model_result in run_result.models.items():
            save_path = os.path.join(model_dir, f"{model_name}_final_pipeline.pkl")
            joblib.dump(model_result.fitted_estimator, save_path)
            saved_paths[model_name] = save_path
        return saved_paths

    # ------------------------------------------------------------------
    # K-Fold
    # ------------------------------------------------------------------

    def save_kfold_ensembles(
        self, kfold_result: KFoldRunResult, voting: str = "soft"
    ) -> Dict[str, str]:
        """
        Wrap each model's fold-trained estimators into a single
        :class:`HabitEnsembleModel` and persist it on disk.

        Args:
        kfold_result:
            Structured K-Fold output.
        voting:
            Either ``"soft"`` (average probabilities) or ``"hard"``
            (majority vote).  Defaults to ``"soft"``.

        Returns:
        Dict[str, str]
            Mapping ``model_name -> saved_path``.  Empty dict when
            ``is_save_model`` is False or when no fold estimators exist.
        """
        if not self.is_save_model:
            return {}

        model_dir = self._ensure_model_dir()

        saved_paths: Dict[str, str] = {}
        for model_name, model in kfold_result.models.items():
            estimators = list(model.fold_estimators)
            if not estimators:
                continue
            ensemble = HabitEnsembleModel(estimators=estimators, voting=voting)
            save_path = os.path.join(model_dir, f"{model_name}_ensemble_final.pkl")
            joblib.dump(ensemble, save_path)
            saved_paths[model_name] = save_path
        return saved_paths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_model_dir(self) -> str:
        """Create and return ``<output_dir>/models``."""
        model_dir = os.path.join(self.output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
