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
#!/usr/bin/env python
"""
Utility functions for habitat feature extraction
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging

class FeatureUtils:
    """Utility class for feature extraction"""
    
    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flatten a nested dictionary
        
        Args:
            d: Dictionary to flatten
            parent_key: Key of parent dictionary (used in recursion)
            sep: Separator between keys in flattened dictionary
            
        Returns:
            Dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(FeatureUtils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def get_n_habitats_from_csv(habitat_folder: str) -> Optional[int]:
        """
        Read the number of habitats from the habitats results table.

        Supports ``habitats.parquet`` (default) and legacy ``habitats.csv``.

        Args:
            habitat_folder: Path to the folder containing the habitats table.

        Returns:
            int: Number of habitats if found, None otherwise.
        """
        from habit.utils.habitats_results_io import (
            find_habitats_results_file,
            load_habitats_results,
        )

        try:
            results_path = find_habitats_results_file(habitat_folder)
            if results_path is None:
                logging.error(
                    "Habitats results file not found in folder: %s",
                    habitat_folder,
                )
                return None

            df = load_habitats_results(results_path)
            # Current HABIT outputs use the canonical lowercase ``habitats``
            # column, while historical CSV exports used ``Habitats``. Resolve
            # the semantic column case-insensitively so valid legacy results do
            # not fall back to an interactive prompt during unattended CLI runs.
            habitat_columns = [
                column
                for column in df.columns
                if str(column).strip().casefold() == "habitats"
            ]
            if len(habitat_columns) == 1:
                habitat_column = habitat_columns[0]
                unique_habitats = int(df[habitat_column].nunique())
                logging.info(
                    "Read %s habitats from %s",
                    unique_habitats,
                    results_path.name,
                )
                return unique_habitats

            if len(habitat_columns) > 1:
                logging.error(
                    "Multiple case-insensitive Habitats columns found in %s: %s",
                    results_path,
                    habitat_columns,
                )
                return None

            logging.error(
                "Habitats column not found in habitats results file: %s",
                results_path,
            )
        except Exception as exc:
            logging.error("Error reading habitats results table: %s", exc)

        return None
    
    @staticmethod
    def create_empty_dataframe_like(reference_df: pd.DataFrame, index: List[str] = None) -> pd.DataFrame:
        """
        Create an empty DataFrame with the same structure as a reference DataFrame
        
        Args:
            reference_df: Reference DataFrame to copy structure from
            index: List of index values for the new DataFrame
            
        Returns:
            pd.DataFrame: Empty DataFrame with same structure as reference_df
        """
        if index is None:
            index = [0]
            
        return pd.DataFrame(
            data=np.nan, 
            index=index, 
            columns=reference_df.columns
        ) 