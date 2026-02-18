import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

class DataManager:
    """
    Handles data loading, merging, and splitting.
    Expects a Pydantic MLConfig object.
    """
    def __init__(self, config: Any, logger: logging.Logger = None):
        """
        Initialize DataManager.
        
        Args:
            config: MLConfig object.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Strictly use object attributes
        self.input_config = config.input
        self.split_method = getattr(config, 'split_method', 'stratified')
        self.test_size = getattr(config, 'test_size', 0.3)
        self.seed = getattr(config, 'random_state', 42)
        self.train_ids_file = getattr(config, 'train_ids_file', None)
        self.test_ids_file = getattr(config, 'test_ids_file', None)
        
        self.data = None
        self.subject_id_col = None
        self.label_col = None
        
    def load_data(self) -> 'DataManager':
        """Loads and merges data from configured files."""
        if not isinstance(self.input_config, list):
            raise TypeError("Input config must be a list of file configurations.")
            
        self.logger.info("Loading data from %d files", len(self.input_config))
        merged_df = None
        first_label_col = None
        label_values = None
        
        for file_idx, file_config in enumerate(self.input_config):
            # Use Pydantic object attributes
            path = file_config.path
            name = getattr(file_config, 'name', '')
            subj_col = getattr(file_config, 'subject_id_col', None)
            lbl_col = getattr(file_config, 'label_col', None)
            features = getattr(file_config, 'features', [])
            # 'add_prefix' is not in current InputFileConfig schema, assuming extra='allow' or removed
            add_prefix = getattr(file_config, 'add_prefix', False)
            
            if not subj_col or not lbl_col:
                raise ValueError(f"subject_id_col and label_col are required for {path}")
                
            self.logger.info(f"Reading {path} (Subject: {subj_col}, Label: {lbl_col})")
            # Read subject ID as string at file-load time to preserve exact formatting
            # (e.g., leading zeros like "0002886419"). Converting to string after
            # numeric inference is too late because leading zeros are already lost.
            df = pd.read_csv(path, dtype={subj_col: str})
            
            # Unify subject ID type
            df[subj_col] = df[subj_col].astype(str)
            
            # Setup global subject/label cols based on first file
            if self.subject_id_col is None:
                self.subject_id_col = subj_col
                self.label_col = lbl_col
                first_label_col = lbl_col
                label_values = df.set_index(subj_col)[lbl_col]
            
            # Prepare feature columns
            df.set_index(subj_col, inplace=True)
            cols_to_keep = []
            rename_map = {}
            
            # Identify feature columns
            available_cols = [c for c in df.columns if c != lbl_col]
            target_cols = features if features else available_cols
            
            for col in target_cols:
                if col in df.columns:
                    cols_to_keep.append(col)
                    # Handle prefix logic
                    if add_prefix and name:
                        rename_map[col] = f"{name}{col}"
                else:
                    self.logger.warning(f"Feature {col} not found in {path}")
            
            subset = df[cols_to_keep].rename(columns=rename_map)
            
            # Merge
            if merged_df is None:
                merged_df = subset
            else:
                # Resolve feature-name collisions before join.
                # This is critical for fusion workflows where each input file often uses
                # the same feature name (e.g., "LogisticRegression_prob"). Pandas join
                # raises a ValueError when overlapping column names are present without
                # suffixes. We rename only overlapping columns to keep behavior stable.
                overlap_cols = merged_df.columns.intersection(subset.columns).tolist()
                if overlap_cols:
                    # Prefer user-provided input name as prefix; fallback to file index.
                    # Ensure prefix ends with "_" so generated names are readable.
                    prefix_source = str(name).strip() if str(name).strip() else f"input{file_idx}"
                    safe_prefix = prefix_source if prefix_source.endswith("_") else f"{prefix_source}_"
                    collision_rename_map = {}

                    for col in overlap_cols:
                        base_new_col = f"{safe_prefix}{col}"
                        new_col = base_new_col
                        suffix_idx = 1

                        # Guarantee uniqueness against existing columns and current batch.
                        while (
                            new_col in merged_df.columns
                            or new_col in subset.columns
                            or new_col in collision_rename_map.values()
                        ):
                            new_col = f"{base_new_col}_{suffix_idx}"
                            suffix_idx += 1

                        collision_rename_map[col] = new_col

                    subset = subset.rename(columns=collision_rename_map)
                    self.logger.warning(
                        "Detected overlapping feature columns in %s; auto-renamed %d columns. "
                        "Examples: %s",
                        path,
                        len(collision_rename_map),
                        dict(list(collision_rename_map.items())[:5]),
                    )

                merged_df = merged_df.join(subset, how='outer')
        
        # Re-attach label
        # Ensure we only keep samples that have labels (from the first file's perspective usually, 
        # or we intersection. For now, assuming first file drives the cohort)
        # Using the index of the merged dataframe to align labels
        if label_values is not None:
            common_indices = merged_df.index.intersection(label_values.index)
            merged_df = merged_df.loc[common_indices]
            merged_df[self.label_col] = label_values.loc[common_indices]
        
        self.data = merged_df
        self.logger.info(f"Data loaded: {self.data.shape}")
        
        # Basic type cleaning
        self.data = self.data.apply(lambda x: pd.to_numeric(x, errors='coerce'))
        
        # Drop rows where Label is NaN (Critical for training)
        original_len = len(self.data)
        self.data = self.data.dropna(subset=[self.label_col])
        if len(self.data) < original_len:
            self.logger.warning(f"Dropped {original_len - len(self.data)} rows with missing labels.")
            
        return self

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into X_train, X_test, y_train, y_test.
        Handles Custom, Random, and Stratified splits.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        X = self.data.drop(columns=[self.label_col])
        y = self.data[self.label_col]
        
        if self.split_method == 'custom':
            # Use stored attributes (already extracted in __init__)
            train_path = self.train_ids_file
            test_path = self.test_ids_file
            
            if not train_path or not test_path:
                raise ValueError("Custom split requires train_ids_file and test_ids_file")
                
            train_ids = self._read_ids(train_path)
            test_ids = self._read_ids(test_path)
            
            # Intersect with available data
            valid_train = [i for i in train_ids if i in X.index]
            valid_test = [i for i in test_ids if i in X.index]
            missing_train = [i for i in train_ids if i not in X.index]
            missing_test = [i for i in test_ids if i not in X.index]

            if missing_train:
                self.logger.warning(
                    "Custom split: %d train IDs not found in data index. Sample: %s",
                    len(missing_train),
                    missing_train[:10]
                )
            if missing_test:
                self.logger.warning(
                    "Custom split: %d test IDs not found in data index. Sample: %s",
                    len(missing_test),
                    missing_test[:10]
                )
            
            X_train = X.loc[valid_train]
            y_train = y.loc[valid_train]
            X_test = X.loc[valid_test]
            y_test = y.loc[valid_test]
            
        else:
            stratify = y if self.split_method == 'stratified' else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.seed, 
                stratify=stratify
            )
            
        self.logger.info(f"Split results: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    def _read_ids(self, path: str) -> List[str]:
        with open(path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['): 
                return [str(i) for i in json.loads(content)]
            if ',' in content:
                return [i.strip() for i in content.split(',')]
            return [line.strip() for line in content.split('\n') if line.strip()]
