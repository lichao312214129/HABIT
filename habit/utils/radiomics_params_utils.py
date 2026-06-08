# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Utilities for loading PyRadiomics parameter YAML files with robust encodings.

PyRadiomics reads parameter files through ruamel.yaml, which on Windows may default
to the system encoding (e.g. GBK). UTF-8 files with non-ASCII comment characters
then fail with UnicodeDecodeError. HABIT loads YAML explicitly and passes a dict
to RadiomicsFeatureExtractor instead.
"""

from __future__ import annotations

import locale
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from radiomics import featureextractor

# Stable GLCM features for voxel-based extraction (small kernelRadius neighborhoods).
# Excludes MCC, Imc1, Imc2 — they need eigenvalue / mutual-information stats that crash
# or return NaN when local GLCM matrices degenerate (flat 1×1 patches).
VOXEL_SAFE_GLCM_FEATURES: tuple[str, ...] = (
    "Contrast",
    "Correlation",
    "JointEnergy",
    "Idm",
    "Autocorrelation",
    "JointAverage",
    "JointEntropy",
    "DifferenceAverage",
    "DifferenceEntropy",
    "SumAverage",
    "SumEntropy",
    "SumSquares",
    "MaximumProbability",
    "Idmn",
    "Id",
    "Idn",
    "InverseVariance",
    "DifferenceVariance",
    "ClusterTendency",
    "ClusterShade",
    "ClusterProminence",
)

_UNSAFE_VOXEL_GLCM_FEATURES: frozenset[str] = frozenset({"MCC", "Imc1", "Imc2"})

# Common encodings for radiomics parameter YAML on Windows / cross-platform setups.
_FALLBACK_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "gbk",
    "cp936",
    "cp1252",
)


def _detect_bom_encoding(raw: bytes) -> Optional[str]:
    """
    Detect byte-order mark and return the corresponding text encoding.

    Args:
        raw: Raw file bytes.

    Returns:
        Encoding name when a known BOM is present, otherwise None.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"
    return None


def _candidate_encodings() -> List[str]:
    """
    Build an ordered list of encodings to try when reading parameter YAML.

    Returns:
        Encoding names: locale preference first, then common UTF-8 / Windows fallbacks.
    """
    candidates: List[str] = []
    preferred: str = locale.getpreferredencoding(False) or ""
    if preferred:
        candidates.append(preferred)
    for encoding in _FALLBACK_ENCODINGS:
        if encoding not in candidates:
            candidates.append(encoding)
    return candidates


def load_radiomics_params_yaml(params_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a PyRadiomics parameter YAML file with UTF-8-first multi-encoding fallback.

    Tries, in order: BOM hint (if any), locale preferred encoding, then utf-8-sig,
    utf-8, gbk/cp936, and cp1252. This covers UTF-8 (with or without BOM), legacy
    GBK files edited on Chinese Windows, and Western Windows ANSI saves.

    Args:
        params_path: Path to the PyRadiomics parameter YAML file.

    Returns:
        Parsed YAML mapping for ``RadiomicsFeatureExtractor(params_dict)``.

    Raises:
        FileNotFoundError: If ``params_path`` does not exist.
        ValueError: If no encoding yields valid YAML mapping content.
        yaml.YAMLError: Propagated when decoded text is invalid YAML under an encoding
            that otherwise decoded successfully (rare; usually folded into ValueError).
    """
    path: Path = Path(params_path)
    if not path.is_file():
        raise FileNotFoundError(f"Radiomics parameter file not found: {path}")

    raw: bytes = path.read_bytes()
    bom_encoding: Optional[str] = _detect_bom_encoding(raw)
    encodings_to_try: List[str] = []
    if bom_encoding is not None:
        encodings_to_try.append(bom_encoding)
    for encoding in _candidate_encodings():
        if encoding not in encodings_to_try:
            encodings_to_try.append(encoding)

    errors: Dict[str, str] = {}
    last_decode_error: Optional[UnicodeDecodeError] = None

    for encoding in encodings_to_try:
        try:
            text: str = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            errors[encoding] = str(exc)
            last_decode_error = exc
            continue

        try:
            params: Any = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            errors[encoding] = f"YAML parse error: {exc}"
            continue

        if not isinstance(params, dict):
            errors[encoding] = f"expected mapping, got {type(params).__name__}"
            continue

        return params

    detail: str = "; ".join(f"{enc}: {msg}" for enc, msg in errors.items())
    message: str = (
        f"Could not load radiomics parameter file {path} "
        f"(tried encodings: {', '.join(encodings_to_try)}). {detail}"
    )
    if last_decode_error is not None:
        raise ValueError(message) from last_decode_error
    raise ValueError(message)


def create_radiomics_feature_extractor(
    params_source: Union[str, Path, Dict[str, Any], None],
) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Create a PyRadiomics feature extractor without relying on system file encoding.

    Args:
        params_source: Path to parameter YAML, parsed parameter dict, inline YAML
            string (when not an existing file path), or None for PyRadiomics defaults.

    Returns:
        Initialized ``RadiomicsFeatureExtractor`` instance.
    """
    if params_source is None:
        return featureextractor.RadiomicsFeatureExtractor()

    if isinstance(params_source, dict):
        return featureextractor.RadiomicsFeatureExtractor(params_source)

    path: Path = Path(str(params_source))
    if path.is_file():
        params: Dict[str, Any] = load_radiomics_params_yaml(path)
        return featureextractor.RadiomicsFeatureExtractor(params)

    # Legacy path: treat non-existent path as inline YAML text (supervoxel extractor).
    parsed: Any = yaml.safe_load(str(params_source))
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Invalid radiomics parameters: expected mapping, got {type(parsed).__name__}"
        )
    return featureextractor.RadiomicsFeatureExtractor(parsed)


def _glcm_requests_all_features(glcm_config: Any) -> bool:
    """
    Return True when GLCM is enabled without an explicit per-feature list.

    PyRadiomics treats bare ``glcm:`` (YAML null) as "compute every GLCM feature".

    Args:
        glcm_config: Value of ``enabledFeatures['glcm']`` from a feature extractor.

    Returns:
        True if the config enables the full GLCM class (no explicit subset).
    """
    return glcm_config is None


def apply_voxel_glcm_defaults(
    enabled_features: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Replace unrestricted GLCM with the voxel-safe feature subset when needed.

    Voxel-based radiomics uses tiny neighborhoods (e.g. 3×3×3 at kernelRadius=1).
    Many voxels yield degenerate GLCM matrices; MCC/Imc1/Imc2 then trigger CUDA/MKL
    eigenvalue failures or NaN. When ``glcm`` is enabled without a feature list,
    substitute ``VOXEL_SAFE_GLCM_FEATURES`` and log the reason.

    Explicit feature lists in ``params_file`` are left unchanged.

    Args:
        enabled_features: ``RadiomicsFeatureExtractor.enabledFeatures`` mapping.
        logger: Optional logger for the substitution warning.

    Returns:
        Updated enabled-features mapping (may share keys with the input dict).
    """
    if "glcm" not in enabled_features:
        return enabled_features

    glcm_config: Any = enabled_features["glcm"]
    if not _glcm_requests_all_features(glcm_config):
        if isinstance(glcm_config, list):
            unsafe: List[str] = [
                name for name in glcm_config if name in _UNSAFE_VOXEL_GLCM_FEATURES
            ]
            if unsafe and logger is not None:
                logger.warning(
                    "voxel_radiomics: params_file lists GLCM feature(s) %s that are "
                    "unstable on small kernel neighborhoods (MCC/Imc1/Imc2). "
                    "Expect CUDA/MKL eigvals errors or NaN unless kernelRadius is large.",
                    unsafe,
                )
        return enabled_features

    updated: Dict[str, Any] = dict(enabled_features)
    updated["glcm"] = list(VOXEL_SAFE_GLCM_FEATURES)

    if logger is not None:
        logger.warning(
            "voxel_radiomics: glcm enabled without an explicit feature list in "
            "params_file; defaulting to %d stable GLCM features (excluding "
            "MCC/Imc1/Imc2). Small kernel neighborhoods produce degenerate GLCM "
            "matrices; unrestricted GLCM can crash TorchRadiomics (eigvals/MKL) "
            "or return NaN. Use config/radiomics/params_voxel_radiomics.yaml or "
            "list GLCM features explicitly to override.",
            len(VOXEL_SAFE_GLCM_FEATURES),
        )

    return updated


def configure_voxel_glcm_on_extractor(
    extractor: featureextractor.RadiomicsFeatureExtractor,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Apply voxel-safe GLCM defaults on a PyRadiomics feature extractor instance.

    Args:
        extractor: Initialized ``RadiomicsFeatureExtractor``.
        logger: Optional logger for warnings.
    """
    updated: Dict[str, Any] = apply_voxel_glcm_defaults(
        extractor.enabledFeatures,
        logger=logger,
    )
    if updated is extractor.enabledFeatures:
        return

    glcm_features: Any = updated.get("glcm")
    if glcm_features is None:
        return

    # PyRadiomics 3.x: enableFeatureClassByName(featureClass, enabled=True) toggles a
    # whole class only. Per-feature lists must go through enableFeaturesByName(**kwargs).
    extractor.enableFeaturesByName(glcm=glcm_features)
