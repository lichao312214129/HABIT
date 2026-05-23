"""Unit tests for optional TorchRadiomics backend resolution."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from habit.utils.torch_radiomics_utils import (
    DEFAULT_TORCH_DTYPE,
    apply_torch_gpu_count,
    log_torch_gpu_install_hint,
    normalize_use_torch_radiomics,
    parse_torch_gpu_indices,
    reset_torch_gpu_install_hint_log,
    resolve_voxel_radiomics_backend,
    select_torch_gpu_device,
    stable_gpu_slot,
)


class TestTorchRadiomicsUtils(unittest.TestCase):
    def test_default_torch_dtype(self) -> None:
        self.assertEqual(DEFAULT_TORCH_DTYPE, "float32")

    def test_normalize_use_torch_radiomics(self) -> None:
        self.assertEqual(normalize_use_torch_radiomics("auto"), "auto")
        self.assertEqual(normalize_use_torch_radiomics("true"), "true")
        self.assertEqual(normalize_use_torch_radiomics("false"), "false")
        self.assertEqual(normalize_use_torch_radiomics(True), "true")
        self.assertEqual(normalize_use_torch_radiomics(False), "false")

    def test_parse_torch_gpu_indices(self) -> None:
        self.assertEqual(parse_torch_gpu_indices([0, 1, 2]), [0, 1, 2])
        self.assertEqual(parse_torch_gpu_indices("0,1,2"), [0, 1, 2])
        self.assertEqual(parse_torch_gpu_indices("cuda:0,cuda:2"), [0, 2])
        self.assertEqual(parse_torch_gpu_indices(1), [1])

    def test_apply_torch_gpu_count(self) -> None:
        self.assertEqual(apply_torch_gpu_count([0, 1, 2], 2), [0, 1])

    def test_stable_gpu_slot(self) -> None:
        self.assertEqual(stable_gpu_slot("sub001", 3), stable_gpu_slot("sub001", 3))

    def test_select_torch_gpu_device_by_subject(self) -> None:
        device = select_torch_gpu_device([0, 1, 2], subject="sub164")
        self.assertIn(device, {"cuda:0", "cuda:1", "cuda:2"})

    @patch("habit.utils.torch_radiomics_utils.validate_torch_gpu_indices")
    @patch("habit.utils.torch_radiomics_utils.is_cuda_available", return_value=True)
    @patch("habit.utils.torch_radiomics_utils.is_torch_available", return_value=True)
    def test_resolve_backend_with_torch_gpus(
        self,
        _mock_torch: object,
        _mock_cuda: object,
        _mock_validate: object,
    ) -> None:
        backend, device = resolve_voxel_radiomics_backend(
            use_torch_radiomics="true",
            torch_gpus=[0, 1],
            subject="sub164",
        )
        self.assertEqual(backend, "torch")
        self.assertIn(device, {"cuda:0", "cuda:1"})

    def test_resolve_backend_false(self) -> None:
        backend, device = resolve_voxel_radiomics_backend(use_torch_radiomics=False)
        self.assertEqual(backend, "pyradiomics")
        self.assertIsNone(device)

    @patch("habit.utils.torch_radiomics_utils.is_torch_available", return_value=False)
    def test_resolve_backend_auto_without_torch(self, _mock_torch: object) -> None:
        backend, device = resolve_voxel_radiomics_backend(
            use_torch_radiomics="auto",
            torch_device="auto",
        )
        self.assertEqual(backend, "pyradiomics")
        self.assertIsNone(device)

    @patch("habit.utils.torch_radiomics_utils.is_cuda_available", return_value=False)
    @patch("habit.utils.torch_radiomics_utils.is_torch_available", return_value=True)
    def test_resolve_backend_auto_without_cuda(
        self,
        _mock_torch: object,
        _mock_cuda: object,
    ) -> None:
        backend, device = resolve_voxel_radiomics_backend(
            use_torch_radiomics="auto",
            torch_device="auto",
        )
        self.assertEqual(backend, "pyradiomics")
        self.assertIsNone(device)

    @patch("habit.utils.torch_radiomics_utils.is_cuda_available", return_value=True)
    @patch("habit.utils.torch_radiomics_utils.is_torch_available", return_value=True)
    def test_resolve_backend_auto_with_cuda(
        self,
        _mock_torch: object,
        _mock_cuda: object,
    ) -> None:
        backend, device = resolve_voxel_radiomics_backend(
            use_torch_radiomics="auto",
            torch_device="auto",
        )
        self.assertEqual(backend, "torch")
        self.assertEqual(device, "cuda:0")

    @patch("habit.utils.torch_radiomics_utils.is_torch_available", return_value=False)
    def test_resolve_backend_true_without_torch_raises(self, _mock_torch: object) -> None:
        with self.assertRaises(RuntimeError):
            resolve_voxel_radiomics_backend(use_torch_radiomics=True)

    @patch("habit.utils.torch_radiomics_utils.logger")
    @patch("habit.utils.torch_radiomics_utils.is_torch_available", return_value=False)
    def test_log_torch_gpu_install_hint_once_for_missing_torch(
        self,
        _mock_torch: object,
        mock_logger: object,
    ) -> None:
        reset_torch_gpu_install_hint_log()
        resolve_voxel_radiomics_backend(use_torch_radiomics="auto")
        resolve_voxel_radiomics_backend(use_torch_radiomics="auto")
        mock_logger.warning.assert_called_once()
        self.assertIn("pip install torch", mock_logger.warning.call_args[0][1])

    @patch("habit.utils.torch_radiomics_utils.logger")
    @patch("habit.utils.torch_radiomics_utils.is_cuda_available", return_value=False)
    @patch("habit.utils.torch_radiomics_utils.is_torch_available", return_value=True)
    def test_log_torch_gpu_install_hint_once_for_cpu_only_torch(
        self,
        _mock_torch: object,
        _mock_cuda: object,
        mock_logger: object,
    ) -> None:
        reset_torch_gpu_install_hint_log()
        resolve_voxel_radiomics_backend(use_torch_radiomics="auto")
        resolve_voxel_radiomics_backend(use_torch_radiomics="auto")
        mock_logger.warning.assert_called_once()
        self.assertIn("CUDA is unavailable", mock_logger.warning.call_args[0][0])

    def test_log_torch_gpu_install_hint_direct(self) -> None:
        reset_torch_gpu_install_hint_log()
        with patch("habit.utils.torch_radiomics_utils.logger") as mock_logger:
            log_torch_gpu_install_hint("torch_not_installed")
            log_torch_gpu_install_hint("torch_not_installed")
            mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
