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
ThresholdManager Module
Handles the calculation, storage, and application of decision thresholds.
"""

from typing import Dict, Any
from .metrics import calculate_metrics_youden
from .prediction_container import PredictionContainer

class ThresholdManager:
    """
    Manages optimal decision thresholds across datasets.
    
    This class is central to clinical research workflows where a threshold
    (e.g., from Youden's Index) is determined on a training/validation set
    and then applied to a separate test set.
    """
    def __init__(self):
        self.store: Dict[str, Dict[str, float]] = {} # {model_name: {threshold_type: value}}

    def find_and_store(self, model_name: str, container: PredictionContainer, method: str = 'youden'):
        """
        Calculates and stores the optimal threshold from a given dataset (usually training).
        """
        if method == 'youden':
            result = calculate_metrics_youden(container.y_true, container.y_prob)
            threshold = result.get('threshold')
            if threshold is not None:
                if model_name not in self.store:
                    self.store[model_name] = {}
                self.store[model_name]['youden'] = threshold

    def get_threshold(self, model_name: str, method: str = 'youden') -> float:
        """Retrieves a stored threshold."""
        return self.store.get(model_name, {}).get(method, 0.5) # Default to 0.5 if not found
