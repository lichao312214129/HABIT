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
Background: The labels from two habitat analyses (mainly due to the randomness of cluster assignment) may not correspond. 
For example, label1 in the first analysis and label1 in the second analysis may represent different feature attributes. 
Direct analysis would lead to mismatched feature pairs in subsequent ICC analysis.

Module for mapping habitat labels between test-retest medical imaging data.

Purpose:
- Implements correlation-based mapping of habitat labels between longitudinal scans
- Handles medical image processing and label remapping operations
- Supports multiprocessing for efficient batch processing

Key Features:
1. Cross-modal correlation analysis between test/retest features
2. Automated habitat label mapping based on statistical similarity
3. Parallel processing of medical image files (NRRD format)
4. Comprehensive logging and progress tracking
5. Configurable through YAML configuration files

Inputs:
- Paired CSV/Excel files with test/retest habitat features
- NRRD format medical images for processing
- YAML configuration specifying analysis parameters

Outputs:
- Remapped medical images with consistent labeling
- Detailed logging of mapping operations and errors
- Preservation of spatial characteristics in output images

Dependencies:
- Utilizes SimpleITK for medical image I/O operations
- Employs pandas for feature data analysis
- Uses multiprocessing for parallel execution
- Requires YAML configuration for analysis parameters

Usage Example:
>>> python habitat_test_retest_mapper.py \
    --config ../configuration.yaml \
    --test ../data/results/habitats.csv \
    --retest ../data/results_of_icc/habitats.csv \
    --input-dir ../data/results_of_icc \
    --out-dir ../data/results_of_icc \
    --processes 8

Note: Maintains original image geometry and metadata during remapping
      operations. Handles edge cases through comprehensive error logging.
"""


import pandas as pd
import numpy as np
import SimpleITK as sitk
import yaml
import argparse
import codecs
import sys
import logging
import glob
import os
import multiprocessing
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, List, Any, Optional
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr, spearmanr, kendalltau

from habit.kernels.habitat_label_match import match_labels_by_features
from habit.utils.log_utils import setup_logger, get_module_logger
from habit.utils.optional_deps import require_excel_backend, require_parquet_backend
from habit.utils.progress_utils import CustomTqdm

# Module logger - will inherit configuration from root logger
logger = get_module_logger(__name__)


def calculate_similarity(x: np.ndarray, y: np.ndarray, method: str = 'pearson') -> float:
    """
    Calculate similarity between two feature vectors using specified method.
    
    Args:
        x: First feature vector
        y: Second feature vector
        method: Similarity calculation method, options:
               - 'pearson': Pearson correlation coefficient
               - 'spearman': Spearman rank correlation
               - 'kendall': Kendall rank correlation
               - 'euclidean': Euclidean distance (normalized and negated)
               - 'cosine': Cosine similarity
               - 'manhattan': Manhattan distance (normalized and negated)
               - 'chebyshev': Chebyshev distance (normalized and negated)
    
    Returns:
        float: Similarity value ranging from [-1, 1]
    """
    if method == 'pearson':
        return pearsonr(x, y)[0]
    elif method == 'spearman':
        return spearmanr(x, y)[0]
    elif method == 'kendall':
        return kendalltau(x, y)[0]
    elif method == 'euclidean':
        # 归一化后取负，使得距离越小相似度越高
        return -euclidean(x, y) / np.sqrt(len(x))
    elif method == 'cosine':
        return 1 - cosine(x, y)  # cosine距离转为相似度
    elif method == 'manhattan':
        return -np.sum(np.abs(x - y)) / len(x)
    elif method == 'chebyshev':
        return -np.max(np.abs(x - y))
    else:
        raise ValueError(f"不支持的相似度计算方法: {method}")


def _read_habitat_table(table_path: str, argument_name: str) -> pd.DataFrame:
    """
    Read a habitats feature table in any of the three supported formats.

    ``.parquet`` needs pyarrow and ``.xlsx`` needs openpyxl; both live in the
    optional ``tables`` extra, so the read is gated to report the extra name
    instead of pandas' internal engine error. ``.csv`` needs nothing.

    Args:
        table_path: Path to a ``.csv``, ``.parquet`` or ``.xlsx`` table.
        argument_name: Caller's parameter name, quoted in the error message.

    Returns:
        pd.DataFrame: The loaded habitats table.

    Raises:
        ValueError: When the file extension is not one of the three supported.
        OptionalDependencyError: When the format's pandas engine is missing.
    """
    if table_path.endswith('.csv'):
        return pd.read_csv(table_path)
    if table_path.endswith('.parquet'):
        require_parquet_backend(
            purpose=f"reading the habitats table {os.path.basename(table_path)}"
        )
        return pd.read_parquet(table_path)
    if table_path.endswith('.xlsx'):
        require_excel_backend(
            purpose=f"reading the habitats table {os.path.basename(table_path)}"
        )
        return pd.read_excel(table_path)
    raise ValueError(
        f'{argument_name} should be a csv, parquet, or excel file'
    )


def find_habitat_mapping(test_habitat_table: str, retest_habitat_table: str, 
                        features: List[str] = None,
                        similarity_method: str = 'pearson') -> Dict[int, int]:
    """
    Find habitat label mapping between two analyses based on feature similarity.
    
    Args:
        test_habitat_table: Path to the habitat feature table file for test group
        retest_habitat_table: Path to the habitat feature table file for retest group
        features: List of feature names used for similarity calculation. If None, 
                 uses columns 4 to second-to-last
        similarity_method: Method for calculating similarity between features
    
    Returns:
        Dict[int, int]: Mapping from retest habitat labels to test habitat labels
    """
    # Load data
    test_df = _read_habitat_table(test_habitat_table, 'test_habitat_table')
    retest_df = _read_habitat_table(retest_habitat_table, 'retest_habitat_table')
    
    # If features not specified, use columns 4 to second-to-last
    if features is None:
        all_columns = test_df.columns.tolist()
        features = all_columns[3:-1]
        logger.info(f"Using default feature columns: {features}")
    else:
        # Verify all specified features exist in the data
        missing_features = [f for f in features if f not in test_df.columns or f not in retest_df.columns]
        if missing_features:
            raise ValueError(f"The following features do not exist in the data: {missing_features}")
    
    # Per-habitat median of the unscaled table columns. Cohort z-score and
    # Hungarian assignment live in match_labels_by_features: greedy
    # argmax can map two retest ids onto one test id.
    test_ids = np.asarray(np.unique(test_df["habitats"]), dtype=np.int64)
    retest_ids = np.asarray(np.unique(retest_df["habitats"]), dtype=np.int64)
    if test_ids.size == 0 or retest_ids.size == 0:
        return {}
    test_rows = np.vstack(
        [
            test_df.loc[test_df["habitats"] == habitat_id, features]
            .median()
            .to_numpy(dtype=np.float64)
            for habitat_id in test_ids.tolist()
        ]
    )
    retest_rows = np.vstack(
        [
            retest_df.loc[retest_df["habitats"] == habitat_id, features]
            .median()
            .to_numpy(dtype=np.float64)
            for habitat_id in retest_ids.tolist()
        ]
    )
    requested = str(similarity_method).strip().lower()
    if requested == "kendall":
        logger.warning(
            "find_habitat_mapping: kendall assignment now uses spearman "
            "(rank correlation + Hungarian)."
        )
        requested = "spearman"
    metric_alias = {
        "pearson": "pearson",
        "spearman": "spearman",
        "euclidean": "euclidean",
        "cosine": "cosine",
        "manhattan": "manhattan",
        "chebyshev": "chebyshev",
    }
    metric = metric_alias.get(requested)
    if metric is None:
        raise ValueError(
            "find_habitat_mapping: similarity_method must be one of "
            f"{sorted(metric_alias) + ['kendall']}; got {similarity_method!r}."
        )
    return match_labels_by_features(
        test_ids,
        test_rows,
        retest_ids,
        retest_rows,
        metric=metric,
        standardize="zscore",
    )


def change_habitat_label(retest_nrrd: str, habitat_mapping: Dict[int, int], out_dir: str) -> str:
    """
    Change habitat labels in retest NRRD file according to the mapping.
    
    Args:
        retest_nrrd: Path to the retest NRRD file containing habitat labels
        habitat_mapping: Dictionary mapping retest habitat labels to test labels
        out_dir: Output directory to save processed files
    
    Returns:
        str: Basename of the processed file
    """
    # Read NRRD image
    retest_nrrd_img = sitk.ReadImage(retest_nrrd)
    retest_nrrd_array = sitk.GetArrayFromImage(retest_nrrd_img)

    # Temporarily shift non-zero values to avoid label conflicts
    max_value = np.max(retest_nrrd_array)
    retest_nrrd_array[retest_nrrd_array != 0] += max_value

    # Apply habitat mapping
    for retest_label, test_label in habitat_mapping.items():
        logger.debug(f"Mapping habitat {retest_label} to {test_label}")
        retest_nrrd_array[(retest_nrrd_array-max_value) == retest_label] = test_label

    # Save processed NRRD file
    base_name = os.path.basename(retest_nrrd).split('.')[0] + '_remapped.nrrd'
    out_file = os.path.join(out_dir, base_name)
    img = sitk.GetImageFromArray(retest_nrrd_array)
    img.CopyInformation(retest_nrrd_img)
    sitk.WriteImage(img, out_file)
    return os.path.basename(retest_nrrd)


def configure_logging(output_dir: Path = None, debug: bool = False) -> logging.Logger:
    """
    Configure logging settings using centralized logging system.
    
    Args:
        output_dir: Directory to save log file. If None, only console logging.
        debug: If True, set logging level to DEBUG, otherwise INFO
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if debug else logging.INFO
    
    return setup_logger(
        name='habitat_test_retest_mapper',
        output_dir=output_dir,
        log_filename='habitat_mapping.log',
        level=level
    )


def process_single_file(retest_nrrd: str, habitat_mapping: Dict[int, int], out_dir: str) -> Tuple[str, bool]:
    """
    Process a single NRRD file with habitat mapping.
    
    Args:
        retest_nrrd: Path to the NRRD file
        habitat_mapping: Dictionary mapping retest habitat labels to test labels
        out_dir: Output directory to save processed files
    
    Returns:
        Tuple[str, bool]: (filename, success_flag)
    """
    try:
        filename = change_habitat_label(retest_nrrd, habitat_mapping, out_dir)
        return filename, True
    except Exception as e:
        return os.path.basename(retest_nrrd), False


def batch_process_files(input_dir: str, habitat_mapping: Dict[int, int], out_dir: str, n_processes: int = 4) -> None:
    """
    Process multiple NRRD files in parallel using multiprocessing.
    
    Args:
        input_dir: Directory containing NRRD files to process
        habitat_mapping: Dictionary mapping retest habitat labels to test labels
        out_dir: Output directory to save processed files
        n_processes: Number of parallel processes to use
    """
    # Find all NRRD files
    nrrd_files = glob.glob(os.path.join(input_dir, "*habitats.nrrd"))
    total = len(nrrd_files)
    logger.info(f"Found {total} files to process")
    
    if total == 0:
        logger.warning(f"No files found in {input_dir} matching pattern *habitats.nrrd")
        return
    
    # Create processing function with fixed arguments
    process_func = partial(process_single_file, habitat_mapping=habitat_mapping, out_dir=out_dir)
    
    # Initialize counters
    success_count = 0
    failure_count = 0
    
    # Process files using multiprocessing pool
    with multiprocessing.Pool(processes=n_processes) as pool:
        for filename, success in CustomTqdm(
            pool.imap_unordered(process_func, nrrd_files),
            total=total,
            desc="Mapping habitats"
        ):
            if success:
                success_count += 1
            else:
                failure_count += 1
    
    logger.info(f"Processing complete. Success: {success_count}, Failed: {failure_count}")


def detect_file_encoding(file_path: str) -> Optional[str]:
    """
    Guess the text encoding of a file.

    ``chardet`` used to be a REQUIRED HABIT dependency for this one function.
    It is no longer declared at all: the caller
    (:func:`read_file_with_encoding`) already retries a fixed candidate list
    (utf-8 / gbk / gb2312 / gb18030 / big5), so statistical detection is only
    a first guess that shortens the retry loop -- never the sole answer. When
    chardet happens to be installed (it is a common transitive dependency) it
    is still consulted, so installations that have it behave exactly as before.

    Without chardet this returns ``None`` rather than guessing, which hands the
    decision to the caller's deterministic candidate list. Guessing from
    :func:`locale.getpreferredencoding` was measured to be actively harmful
    here: on a Windows machine with a Chinese locale it reports ``cp936``, and
    because cp936 decodes most UTF-8 byte sequences without raising, a UTF-8
    config -- the common case -- would be silently decoded into mojibake with
    no ``UnicodeDecodeError`` for the retry loop to catch. Trying utf-8 first
    and falling back on failure is both platform-independent and correct for
    every encoding in the candidate list.

    Args:
        file_path: Path to the file whose encoding should be guessed.

    Returns:
        Optional[str]: Best-guess encoding name, or ``None`` when no guess is
        available (the caller then relies on its candidate list alone).
    """
    # Read the first 1000 bytes: enough for a statistical guess, cheap enough
    # to do on every call.
    with open(file_path, 'rb') as f:
        raw_data = f.read(1000)

    # A UTF-8 BOM is a definitive answer that needs no detector at all.
    if raw_data.startswith(codecs.BOM_UTF8):
        return 'utf-8-sig'

    try:
        import chardet
    except ImportError:
        # No guess at all: the caller's candidate list starts with utf-8, which
        # raises UnicodeDecodeError on non-UTF-8 input and therefore lets the
        # loop advance to the correct codec. See the docstring for why a
        # locale-derived guess is worse than no guess.
        return None

    result = chardet.detect(raw_data)
    return result['encoding']


def read_file_with_encoding(file_path: str) -> Any:
    """
    Attempt to read file using detected encoding, fallback to other encodings if failed.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Any: File contents
    
    Raises:
        UnicodeDecodeError: If all encoding attempts fail
    """
    # First try detected encoding
    detected_encoding = detect_file_encoding(file_path)
    if detected_encoding:
        try:
            with open(file_path, 'r', encoding=detected_encoding) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            pass
    
    # If detected encoding fails, try other encodings
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']
    for encoding in encodings:
        if encoding == detected_encoding:
            continue  # Skip already tried encoding
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to read file with any supported encoding: {file_path}")


def main():
    """
    Main program entry point for habitat test-retest mapping.
    """
    parser = argparse.ArgumentParser(
        description='Map habitat labels between test-retest data'
    )
    parser.add_argument('--test-habitat-table', type=str, required=True,
                      help='测试组的habitat特征表格文件路径')
    parser.add_argument('--retest-habitat-table', type=str, required=True,
                      help='重测组的habitat特征表格文件路径')
    parser.add_argument('--features', type=str, nargs='+',
                      help='用于计算相似性的特征名称列表，如果不指定则使用第4到倒数第1列')
    parser.add_argument('--similarity-method', type=str, default='pearson',
                      choices=['pearson', 'spearman', 'kendall', 'euclidean', 
                              'cosine', 'manhattan', 'chebyshev'],
                      help='相似度计算方法')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='包含重测组NRRD文件的目录')
    parser.add_argument('--out-dir', type=str, required=True,
                      help='处理后文件的输出目录')
    parser.add_argument('--processes', type=int,
                      default=2,
                      help='使用的进程数 (默认: 2)')
    parser.add_argument('--debug', action='store_true',
                      help='启用调试日志')
    
    args = parser.parse_args()
    
    # Configure logging with output directory for log file
    output_path = Path(args.out_dir)
    configure_logging(output_dir=output_path, debug=args.debug)
    
    try:
        # 创建输出目录
        os.makedirs(args.out_dir, exist_ok=True)
        
        # 找到habitat映射
        logger.info("Calculating habitat mapping between test and retest data...")
        habitat_mapping = find_habitat_mapping(
            args.test_habitat_table, args.retest_habitat_table, 
            args.features, args.similarity_method)
        logger.debug(f"Habitat mapping: {habitat_mapping}")
        
        # Process files
        logger.info(f"Starting batch processing with {args.processes} processes...")
        batch_process_files(args.input_dir, habitat_mapping, args.out_dir, args.processes)
        
        logger.info("Processing complete")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # If no command line arguments are provided
        # Default arguments for debugging
        sys.argv.extend([
            '--test-habitat-table', 'F:\\work\\research\\radiomics_TLSs\\data\\results\\habitats.csv',
            '--retest-habitat-table', 'F:\\work\\research\\radiomics_TLSs\\data\\results_of_icc\\habitats.csv',
            '--input-dir', 'F:\\work\\research\\radiomics_TLSs\\data\\results_of_icc1',
            '--out-dir', 'F:\\work\\research\\radiomics_TLSs\\data\\results_of_icc1',
            '--processes', '4',
            '--debug'
        ])
    main() 
    # 命令行的写法是：python habitat_test_retest_mapper.py --test-habitat-table F:\\work\\research\\radiomics_TLSs\\data\\results\\habitats.csv --retest-habitat-table F:\\work\\research\\radiomics_TLSs\\data\\results_of_icc\\habitats.csv --input-dir F:\\work\\research\\radiomics_TLSs\\data\\results_of_icc1 --out-dir F:\\work\\research\\radiomics_TLSs\\data\\results_of_icc1 --processes 4 --debug