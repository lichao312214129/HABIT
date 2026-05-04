import json
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class TabularLoader:
    """
    Read tabular files with a single, shared format adapter.

    This class is intentionally independent from split/merge logic so we can
    evolve file-format support (CSV/TSV/Excel today, Parquet tomorrow) without
    touching data assembly code.
    """

    SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")

    def load(self, path: str, subject_id_col: Optional[str] = None) -> pd.DataFrame:
        """
        Load one table file.

        Args:
            path: Input file path.
            subject_id_col: Subject-ID column used for string dtype preservation.

        Returns:
            pd.DataFrame: Loaded table.
        """
        ext: str = os.path.splitext(str(path))[1].lower()
        dtype = {subject_id_col: str} if subject_id_col else None

        if ext == ".csv":
            return pd.read_csv(path, dtype=dtype)
        if ext in (".tsv", ".txt"):
            return pd.read_csv(path, sep="\t", dtype=dtype)
        if ext in (".xlsx", ".xls"):
            # Read first, then caller may normalize subject column explicitly.
            return pd.read_excel(path)

        raise ValueError(
            f"Unsupported input file format: {path}. "
            f"Supported formats are {', '.join(self.SUPPORTED_EXTENSIONS)}."
        )


@dataclass
class MergedDataset:
    """Structured result from feature-table assembly."""

    data: pd.DataFrame
    subject_id_col: str
    label_col: str


class FeatureTableAssembler:
    """
    Merge multiple feature tables into one training dataframe.

    Responsibilities:
    - Read each configured table.
    - Keep a stable subject index.
    - Resolve duplicate feature names deterministically.
    - Re-attach labels from the first input file as cohort truth.
    """

    def __init__(self, loader: TabularLoader, logger: logging.Logger) -> None:
        self.loader = loader
        self.logger = logger

    def assemble(self, input_config: List[Any]) -> MergedDataset:
        if not isinstance(input_config, list):
            raise TypeError("Input config must be a list of file configurations.")
        if not input_config:
            raise ValueError("Input config cannot be empty.")

        merged_df: Optional[pd.DataFrame] = None
        subject_id_col: Optional[str] = None
        label_col: Optional[str] = None
        label_values: Optional[pd.Series] = None

        self.logger.info("Loading data from %d files", len(input_config))

        for file_idx, file_config in enumerate(input_config):
            path = file_config.path
            name = getattr(file_config, "name", "")
            subj_col = getattr(file_config, "subject_id_col", None)
            lbl_col = getattr(file_config, "label_col", None)
            features = getattr(file_config, "features", []) or []
            add_prefix = getattr(file_config, "add_prefix", False)

            if not subj_col or not lbl_col:
                raise ValueError(f"subject_id_col and label_col are required for {path}")

            self.logger.info("Reading %s (Subject: %s, Label: %s)", path, subj_col, lbl_col)
            df = self.loader.load(path=path, subject_id_col=subj_col)
            df[subj_col] = df[subj_col].astype(str)

            if subject_id_col is None:
                subject_id_col = subj_col
                label_col = lbl_col
                label_values = df.set_index(subj_col)[lbl_col]

            subset = self._extract_features(
                df=df,
                subject_col=subj_col,
                label_col=lbl_col,
                target_features=features,
                add_prefix=add_prefix,
                input_name=name,
            )

            if merged_df is None:
                merged_df = subset
            else:
                merged_df = self._merge_with_collision_resolution(
                    merged_df=merged_df,
                    subset=subset,
                    input_name=name,
                    file_idx=file_idx,
                    path=path,
                )

        if merged_df is None or label_values is None or subject_id_col is None or label_col is None:
            raise ValueError("Failed to assemble merged dataset from input configuration.")

        common_indices = merged_df.index.intersection(label_values.index)
        merged_df = merged_df.loc[common_indices]
        merged_df[label_col] = label_values.loc[common_indices]

        return MergedDataset(
            data=merged_df,
            subject_id_col=subject_id_col,
            label_col=label_col,
        )

    def _extract_features(
        self,
        df: pd.DataFrame,
        subject_col: str,
        label_col: str,
        target_features: List[str],
        add_prefix: bool,
        input_name: str,
    ) -> pd.DataFrame:
        df = df.set_index(subject_col)

        available_cols = [col for col in df.columns if col != label_col]
        selected_cols = target_features if target_features else available_cols

        cols_to_keep: List[str] = []
        rename_map: dict = {}
        for col in selected_cols:
            if col in df.columns:
                cols_to_keep.append(col)
                if add_prefix and input_name:
                    rename_map[col] = f"{input_name}{col}"
            else:
                self.logger.warning("Feature %s not found in input table", col)

        return df[cols_to_keep].rename(columns=rename_map)

    def _merge_with_collision_resolution(
        self,
        merged_df: pd.DataFrame,
        subset: pd.DataFrame,
        input_name: str,
        file_idx: int,
        path: str,
    ) -> pd.DataFrame:
        overlap_cols = merged_df.columns.intersection(subset.columns).tolist()
        if overlap_cols:
            prefix_source = str(input_name).strip() if str(input_name).strip() else f"input{file_idx}"
            safe_prefix = prefix_source if prefix_source.endswith("_") else f"{prefix_source}_"
            collision_rename_map = {}

            for col in overlap_cols:
                base_new_col = f"{safe_prefix}{col}"
                new_col = base_new_col
                suffix_idx = 1
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
                "Detected overlapping feature columns in %s; auto-renamed %d columns. Examples: %s",
                path,
                len(collision_rename_map),
                dict(list(collision_rename_map.items())[:5]),
            )

        return merged_df.join(subset, how="outer")


class SplitStrategy:
    """Encapsulate train/test split policies for DataManager."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_method: str,
        test_size: float,
        random_state: int,
        train_ids_file: Optional[str] = None,
        test_ids_file: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if split_method == "custom":
            if not train_ids_file or not test_ids_file:
                raise ValueError("Custom split requires train_ids_file and test_ids_file")

            train_ids = self._read_ids(train_ids_file)
            test_ids = self._read_ids(test_ids_file)
            valid_train = [item for item in train_ids if item in X.index]
            valid_test = [item for item in test_ids if item in X.index]
            missing_train = [item for item in train_ids if item not in X.index]
            missing_test = [item for item in test_ids if item not in X.index]

            if missing_train:
                self.logger.warning(
                    "Custom split: %d train IDs not found in data index. Sample: %s",
                    len(missing_train),
                    missing_train[:10],
                )
            if missing_test:
                self.logger.warning(
                    "Custom split: %d test IDs not found in data index. Sample: %s",
                    len(missing_test),
                    missing_test[:10],
                )

            X_train = X.loc[valid_train]
            y_train = y.loc[valid_train]
            X_test = X.loc[valid_test]
            y_test = y.loc[valid_test]
            return X_train, X_test, y_train, y_test

        stratify = y if split_method == "stratified" else None
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    @staticmethod
    def _read_ids(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as file_handle:
            content = file_handle.read().strip()
            if content.startswith("["):
                return [str(item) for item in json.loads(content)]
            if "," in content:
                return [item.strip() for item in content.split(",")]
            return [line.strip() for line in content.split("\n") if line.strip()]


class DataManager:
    """
    Coordinates tabular data loading, assembly, and splitting.

    External API is preserved for compatibility:
    - ``load_data`` populates ``self.data``.
    - ``split_data`` returns train/test split.
    - ``load_inference_data`` loads one predict-mode table.
    """

    def __init__(self, config: Any, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.input_config = config.input
        self.split_method = getattr(config, "split_method", "stratified")
        self.test_size = getattr(config, "test_size", 0.3)
        self.seed = getattr(config, "random_state", 42)
        self.train_ids_file = getattr(config, "train_ids_file", None)
        self.test_ids_file = getattr(config, "test_ids_file", None)

        self.data: Optional[pd.DataFrame] = None
        self.subject_id_col: Optional[str] = None
        self.label_col: Optional[str] = None

        self.loader = TabularLoader()
        self.assembler = FeatureTableAssembler(loader=self.loader, logger=self.logger)
        self.splitter = SplitStrategy(logger=self.logger)

    def load_data(self) -> "DataManager":
        merged = self.assembler.assemble(self.input_config)
        self.data = merged.data
        self.subject_id_col = merged.subject_id_col
        self.label_col = merged.label_col

        self.logger.info("Data loaded: %s", self.data.shape)

        # Keep historical behavior: attempt numeric conversion for model inputs.
        self.data = self.data.apply(lambda series: pd.to_numeric(series, errors="coerce"))
        original_len = len(self.data)
        self.data = self.data.dropna(subset=[self.label_col])
        if len(self.data) < original_len:
            self.logger.warning(
                "Dropped %d rows with missing labels.",
                original_len - len(self.data),
            )
        return self

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into X_train, X_test, y_train, y_test.
        Handles Custom, Random, and Stratified splits.
        """
        if self.data is None or self.label_col is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X = self.data.drop(columns=[self.label_col])
        y = self.data[self.label_col]

        X_train, X_test, y_train, y_test = self.splitter.split(
            X=X,
            y=y,
            split_method=self.split_method,
            test_size=self.test_size,
            random_state=self.seed,
            train_ids_file=self.train_ids_file,
            test_ids_file=self.test_ids_file,
        )

        self.logger.info("Split results: Train=%d, Test=%d", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    def load_inference_data(self, path: str) -> pd.DataFrame:
        """
        Load a single tabular file for inference (predict mode).

        Uses the same :meth:`_read_table_file` adapter as the training path so
        that all supported formats (csv, tsv, xlsx/xls) work identically in
        both train and predict, and future format additions are automatically
        available to both paths.

        Args:
            path: Absolute or relative path to the input file.

        Returns:
            pd.DataFrame: Loaded table (no label column handling; raw content).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Inference data file not found: {path}")

        # Reuse the same adapter used in train mode so supported formats and
        # parsing behavior stay aligned across run modes.
        return self.loader.load(path=path, subject_id_col=None)
