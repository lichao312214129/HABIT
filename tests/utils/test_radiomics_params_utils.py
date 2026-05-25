"""Tests for multi-encoding PyRadiomics parameter YAML loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from habit.utils.radiomics_params_utils import (
    VOXEL_SAFE_GLCM_FEATURES,
    apply_voxel_glcm_defaults,
    configure_voxel_glcm_on_extractor,
    create_radiomics_feature_extractor,
    load_radiomics_params_yaml,
)


class TestRadiomicsParamsUtils(unittest.TestCase):
    def test_load_utf8_with_em_dash_comment(self) -> None:
        content: str = (
            "featureClass:\n"
            "  firstorder:\n"
            "setting:\n"
            "  binWidth: 25  # avoid MCC/Imc1/Imc2 — eigvals crash\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "params.yaml"
            path.write_bytes(content.encode("utf-8"))
            params = load_radiomics_params_yaml(path)
        self.assertEqual(params["setting"]["binWidth"], 25)

    def test_load_gbk_encoded_file(self) -> None:
        content: str = (
            "featureClass:\n"
            "  firstorder:\n"
            "setting:\n"
            "  binWidth: 25  # 中文注释\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "params.yaml"
            path.write_bytes(content.encode("gbk"))
            params = load_radiomics_params_yaml(path)
        self.assertEqual(params["setting"]["binWidth"], 25)

    def test_load_utf8_sig_with_bom(self) -> None:
        content: str = "featureClass:\n  firstorder:\nsetting:\n  binWidth: 10\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "params.yaml"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            params = load_radiomics_params_yaml(path)
        self.assertEqual(params["setting"]["binWidth"], 10)

    def test_create_extractor_from_file(self) -> None:
        content: str = "featureClass:\n  firstorder:\nsetting:\n  binWidth: 25\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path: Path = Path(tmp_dir) / "params.yaml"
            path.write_text(content, encoding="utf-8")
            extractor = create_radiomics_feature_extractor(path)
        self.assertIn("firstorder", extractor.enabledFeatures)


class TestVoxelGlcmDefaults(unittest.TestCase):
    def test_apply_defaults_when_glcm_unrestricted(self) -> None:
        enabled: dict = {"firstorder": None, "glcm": None}
        updated = apply_voxel_glcm_defaults(enabled, logger=None)
        self.assertEqual(updated["glcm"], list(VOXEL_SAFE_GLCM_FEATURES))
        self.assertNotIn("MCC", updated["glcm"])
        self.assertNotIn("Imc1", updated["glcm"])
        self.assertNotIn("Imc2", updated["glcm"])

    def test_respect_explicit_glcm_list(self) -> None:
        explicit: list[str] = ["Contrast", "Correlation"]
        enabled: dict = {"firstorder": None, "glcm": explicit}
        updated = apply_voxel_glcm_defaults(enabled, logger=None)
        self.assertIs(updated, enabled)
        self.assertEqual(updated["glcm"], explicit)

    def test_no_glcm_key_unchanged(self) -> None:
        enabled: dict = {"firstorder": None}
        updated = apply_voxel_glcm_defaults(enabled, logger=None)
        self.assertIs(updated, enabled)

    def test_configure_voxel_glcm_uses_enable_features_by_name(self) -> None:
        from unittest.mock import MagicMock

        extractor = MagicMock()
        extractor.enabledFeatures = {"firstorder": None, "glcm": None}
        configure_voxel_glcm_on_extractor(extractor, logger=None)
        extractor.enableFeaturesByName.assert_called_once_with(
            glcm=list(VOXEL_SAFE_GLCM_FEATURES),
        )
        extractor.enableFeatureClassByName.assert_not_called()

    def test_configure_voxel_glcm_skips_explicit_list(self) -> None:
        from unittest.mock import MagicMock

        explicit: list[str] = ["Contrast", "Correlation"]
        extractor = MagicMock()
        extractor.enabledFeatures = {"glcm": explicit}
        configure_voxel_glcm_on_extractor(extractor, logger=None)
        extractor.enableFeaturesByName.assert_not_called()


if __name__ == "__main__":
    unittest.main()
