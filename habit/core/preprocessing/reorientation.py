from typing import Any, Dict, List, Union

import SimpleITK as sitk

from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory
from ...utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


@PreprocessorFactory.register("reorientation")
class ReorientationPreprocessor(BasePreprocessor):
    """Preprocessor for reorienting images to a target coordinate system (e.g., LPS).

    This preprocessor allows adjusting the image orientation to a specific canonical
    direction. It supports two modes:
    1. 'strict': Exact spatial resampling to a perfect canonical grid (uses interpolation).
                 Linear for modality images; nearest neighbor for masks (same naming as
                 ``resample``: ``mask_<modality>`` when present in ``data``).
    2. 'closest': Only flips and permutes axes to get as close to the target
                  orientation as possible without interpolation.
    """

    def __init__(
        self,
        keys: Union[str, List[str]],
        target_orientation: str = "LPS",
        mode: str = "closest",
        allow_missing_keys: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the reorientation preprocessor.

        Args:
            keys (Union[str, List[str]]): Modality keys to reorient (e.g. ``t1``, ``t2``).
                Masks are not listed here: for each modality ``m``, if ``mask_m`` exists in
                ``data``, it is reoriented automatically (``resample`` convention).
            target_orientation (str): Target anatomical orientation (e.g., 'LPS', 'RAS').
            mode (str): 'closest' (no interpolation, flip/permute only) or 'strict' (resampling).
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            **kwargs: Ignored except ``is_label`` (deprecated): if present, a warning is logged
                and the value is ignored. Previously used to pick interpolator per key; mask
                handling is now automatic via ``mask_<modality>`` keys.
        """
        if "is_label" in kwargs:
            logger.warning(
                "reorientation: 'is_label' is deprecated and ignored. "
                "Masks are detected via keys 'mask_<modality>' (same as resample)."
            )
            kwargs.pop("is_label", None)
        if kwargs:
            logger.warning(
                "reorientation: ignoring unknown keyword arguments: %s",
                list(kwargs.keys()),
            )

        super().__init__(keys, allow_missing_keys)
        self.target_orientation = target_orientation.upper()
        self.mode = mode.lower()

        if self.mode not in ["closest", "strict"]:
            raise ValueError("mode must be either 'closest' or 'strict'")

        # Modality keys only (masks use mask_<mod> like resample.py); BasePreprocessor normalizes str -> list.
        self.img_keys: List[str] = list(self.keys)
        self.mask_keys: List[str] = [f"mask_{key}" for key in self.img_keys]

    def _reorient_sitk(
        self,
        image: sitk.Image,
        *,
        is_mask: bool,
        key: str,
    ) -> sitk.Image:
        """Apply reorientation to one SimpleITK volume.

        In ``strict`` mode, modality images use linear interpolation; masks use nearest
        neighbor so label IDs are preserved.

        Args:
            image (sitk.Image): Input volume.
            is_mask (bool): If True, use nearest-neighbor resampling in ``strict`` mode.
            key (str): Dictionary key (for logging only).

        Returns:
            sitk.Image: Reoriented volume.
        """
        if self.mode == "closest":
            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation(self.target_orientation)
            try:
                out = orient_filter.Execute(image)
                logger.debug(
                    "Reoriented '%s' to %s using 'closest' mode.",
                    key,
                    self.target_orientation,
                )
                return out
            except Exception as e:
                logger.error(
                    "Failed to reorient '%s' using 'closest' mode: %s",
                    key,
                    e,
                )
                raise

        # strict: full resampling to orthogonal grid (interpolation required)
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation(self.target_orientation)

        dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
        dummy.SetDirection(image.GetDirection())
        dummy_reoriented = orient_filter.Execute(dummy)
        target_direction = dummy_reoriented.GetDirection()

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetOutputDirection(target_direction)

        if is_mask:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            mm = sitk.MinimumMaximumImageFilter()
            mm.Execute(image)
            resampler.SetDefaultPixelValue(mm.GetMinimum())

        transform = sitk.Transform()
        resampler.SetTransform(transform)

        try:
            out = resampler.Execute(image)
            logger.debug(
                "Reoriented '%s' to %s using 'strict' mode (interpolated).",
                key,
                self.target_orientation,
            )
            return out
        except Exception as e:
            logger.error(
                "Failed to reorient '%s' using 'strict' mode: %s",
                key,
                e,
            )
            raise

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data by reorienting images and optional masks.

        Args:
            data (Dict[str, Any]): Input data dictionary containing SimpleITK images.

        Returns:
            Dict[str, Any]: Processed data dictionary with reoriented images.
        """
        self._check_keys(data)

        for key in self.img_keys:
            if key not in data:
                continue
            image = data[key]
            if not isinstance(image, sitk.Image):
                logger.warning(
                    "Data for key '%s' is not a SimpleITK Image. Skipping reorientation.",
                    key,
                )
                continue
            data[key] = self._reorient_sitk(image, is_mask=False, key=key)

        for mask_key in self.mask_keys:
            if mask_key not in data:
                continue
            image = data[mask_key]
            if not isinstance(image, sitk.Image):
                logger.warning(
                    "Data for key '%s' is not a SimpleITK Image. Skipping reorientation.",
                    mask_key,
                )
                continue
            data[mask_key] = self._reorient_sitk(image, is_mask=True, key=mask_key)

        return data
