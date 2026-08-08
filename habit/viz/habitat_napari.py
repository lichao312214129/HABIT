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
"""Optional napari viewer for habitat label maps on source images.

Pure array API: one or more greyscale images + integer habitat labels in, a
napari ``Viewer`` out. No filesystem I/O. napari (and its Qt binding) live
behind the ``view`` pip extra; missing them raises
:class:`~habit.exceptions.OptionalDependencyError` with an install hint.

``habit view`` prefers this viewer (``--backend auto`` / ``napari``) and
falls back to matplotlib (:func:`plot_habitat_overlay`) when napari is
missing — Qt is never a hard dependency.

With ``show=True`` (default), :func:`view_habitat_napari` calls
``napari.run()`` so a plain ``python script.py`` keeps the window open until
the user closes it. Without that, the process exits and the window appears to
flash then close.

Orientation matches the matplotlib overlay for in-plane A-P / L-R via
:mod:`habit.viz.orientation`. Default convention is radiological. Whole-volume
``z`` is not flipped by default so axial slider indices match file order /
ITK-SNAP / matplotlib axis-0 indices.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from habit.exceptions import HABITAPIError, OptionalDependencyError
from habit.utils.optional_deps import (
    INSTALLATION_DOCS_URL,
    install_command,
    require,
)
from habit.viz.labels import sanitize_label
from habit.viz.orientation import (
    DEFAULT_DISPLAY_CONVENTION,
    DisplayConvention,
    apply_radiological_flips,
    normalize_display_convention,
    volume_display_flips,
)

__all__ = ["view_habitat_napari", "napari_radiological_flips", "napari_display_flips"]

#: What habit.viz needs napari for.
_VIEW_PURPOSE = "interactive habitat viewing (image + labels layers)"

#: Extra install paths shown when napari / Viewer is missing or broken.
#: Prefer clone editable install: PyPI 1.0.x does not declare ``[view]`` yet,
#: and some China mirrors do not host ``habitat-analysis`` at all.
_VIEW_INSTALL_ALTERNATIVES: Tuple[str, ...] = (
    'From a cloned HABIT repo (recommended for development): '
    'pip install -e ".[view]"',
    'Direct napari stack (works even when the habitat-analysis wheel '
    'has no [view] extra, or your PyPI mirror lacks the package): '
    'pip install "napari[pyqt5]" "npe2>=0.8.2" '
    '"pydantic!=2.11.*,>=2.8,<3" -i https://pypi.org/simple',
    'Or fall back to matplotlib: habit view --backend matplotlib '
    '(requires the "viz" extra)',
)

#: Accept a single volume or a sequence of volumes for multi-sequence viewing.
ImageInput = Union[np.ndarray, Sequence[np.ndarray]]


def _resolve_viewer_class(napari_module: Any) -> Any:
    """
    Resolve ``Viewer`` from an imported napari module.

    Prefers ``napari.Viewer``, then ``napari.viewer.Viewer``. A separate
    ``from napari.viewer import Viewer`` is only attempted when the module
    looks like a real package (has ``__file__``), so empty namespace stubs
    and test doubles without Viewer stay incomplete.

    Args:
        napari_module: Module object returned by :func:`require`.

    Returns:
        The Viewer class, or ``None`` when it cannot be resolved.
    """
    viewer_cls = getattr(napari_module, "Viewer", None)
    if viewer_cls is not None:
        return viewer_cls
    viewer_sub = getattr(napari_module, "viewer", None)
    if viewer_sub is not None:
        viewer_cls = getattr(viewer_sub, "Viewer", None)
        if viewer_cls is not None:
            return viewer_cls
    # Real napari packages have __file__; broken namespace leftovers do not.
    if getattr(napari_module, "__file__", None):
        try:
            from napari.viewer import Viewer as viewer_cls  # type: ignore

            return viewer_cls
        except Exception:  # noqa: BLE001
            return None
    return None


def _require_napari() -> Any:
    """
    Import napari and resolve ``Viewer``, or raise ``OptionalDependencyError``.

    A bare ``import napari`` can succeed for a *broken* install: an empty
    namespace directory left after a failed uninstall/reinstall has no
    ``__init__.py`` and therefore no ``Viewer``. That used to surface as a
    cryptic ``AttributeError``. Convert it into the same installable hint
    users already get for a missing ``[view]`` extra.

    Returns:
        The imported ``napari`` module (must expose ``Viewer``).

    Raises:
        OptionalDependencyError: When napari is missing, or present but
            incomplete (no ``Viewer``).
    """
    napari = require(
        "napari",
        extra="view",
        purpose=_VIEW_PURPOSE,
        alternatives=_VIEW_INSTALL_ALTERNATIVES,
    )
    viewer_cls = _resolve_viewer_class(napari)
    if viewer_cls is None:
        raise OptionalDependencyError(
            "napari is importable but incomplete (no Viewer). "
            "This usually means a broken or partial install left an empty "
            "napari namespace under site-packages.\n\n"
            "Reinstall the view stack:\n"
            f"  {install_command('view')}\n"
            '  # from a clone: pip install -e ".[view]"\n'
            '  # or: pip install "napari[pyqt5]" "npe2>=0.8.2" '
            '"pydantic!=2.11.*,>=2.8,<3" -i https://pypi.org/simple\n\n'
            f"Every extra and what it unlocks: {INSTALLATION_DOCS_URL}"
        )
    # Ensure callers can use napari.Viewer even when only the submodule
    # path worked (broken top-level re-export).
    if getattr(napari, "Viewer", None) is None:
        try:
            setattr(napari, "Viewer", viewer_cls)
        except Exception:  # noqa: BLE001
            pass
    return napari


def _as_volume(array: np.ndarray, name: str) -> np.ndarray:
    """
    Coerce ``array`` to a 2D or 3D volume (drop singleton leading axes).

    Args:
        array: Candidate image or label array.
        name: Name used in error messages.

    Returns:
        Array with ndim in ``{2, 3}``.

    Raises:
        HABITAPIError: When the array cannot be interpreted as a volume.
    """
    volume = np.asarray(array)
    while volume.ndim > 3 and volume.shape[0] == 1:
        volume = np.squeeze(volume, axis=0)
    if volume.ndim == 4:
        # Multi-channel volumes: average channels for display only.
        volume = np.mean(volume, axis=-1) if volume.shape[-1] <= 4 else volume[0]
    if volume.ndim not in (2, 3):
        raise HABITAPIError(
            f"view_habitat_napari: {name} must be 2D or 3D after squeeze; "
            f"got shape {tuple(np.asarray(array).shape)}."
        )
    if volume.size == 0:
        raise HABITAPIError(f"view_habitat_napari: {name} must not be empty.")
    return volume


def _normalize_images(images: ImageInput) -> List[np.ndarray]:
    """
    Normalize a single array or a sequence of arrays to a list of volumes.

    A bare ``np.ndarray`` is treated as one image (not a sequence of slices).

    Args:
        images: One greyscale volume or a sequence of volumes.

    Returns:
        Non-empty list of 2D/3D volumes.

    Raises:
        HABITAPIError: When ``images`` is empty or not array-like.
    """
    if isinstance(images, np.ndarray):
        return [_as_volume(images, "image")]

    if isinstance(images, (str, bytes)):
        raise HABITAPIError(
            "view_habitat_napari: images must be ndarray(s), not a file path."
        )

    try:
        sequence = list(images)
    except TypeError as exc:
        raise HABITAPIError(
            "view_habitat_napari: images must be an ndarray or a sequence of ndarrays."
        ) from exc

    if not sequence:
        raise HABITAPIError("view_habitat_napari: images must not be empty.")

    volumes: List[np.ndarray] = []
    for index, item in enumerate(sequence):
        volumes.append(_as_volume(np.asarray(item), f"images[{index}]"))
    return volumes


def _napari_scale(
    spacing: Optional[Sequence[float]],
    ndim: int,
) -> Optional[Tuple[float, ...]]:
    """
    Convert SimpleITK-order spacing ``(x, y[, z])`` to napari ``scale``.

    ``ImageVolume.data`` / SimpleITK arrays are ``(z, y, x)`` (or ``(y, x)``),
    while ``ImageVolume.spacing`` stays in SimpleITK physical order. napari
    ``scale`` must match array axis order, so we reverse the spacing tuple.

    Args:
        spacing: Physical voxel size in SimpleITK axis order, or ``None``.
        ndim: Spatial dimensionality of the displayed volume (2 or 3).

    Returns:
        Scale tuple for napari, or ``None`` when spacing is omitted / invalid.
    """
    if spacing is None:
        return None
    values = tuple(float(v) for v in spacing)
    if len(values) < ndim:
        return None
    if any(not np.isfinite(v) or v <= 0.0 for v in values):
        return None
    # Use the last ``ndim`` spacing entries then reverse to array axis order.
    physical = values[-ndim:]
    return tuple(reversed(physical))


def napari_display_flips(
    direction: Optional[Sequence[float]],
    *,
    ndim: int,
    convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
    preserve_axial_index: bool = True,
) -> Tuple[bool, ...]:
    """
    Compute display-axis flips for napari layers (testable without Qt).

    Args:
        direction: SimpleITK flattened 3x3 direction, or ``None`` (LPS identity
            for 3D, matching :func:`~habit.viz.plot_habitat_overlay`).
        ndim: Array dimensionality (2 or 3).
        convention: ``\"radiological\"`` (default), ``\"neurological\"``, or
            ``\"native\"``.
        preserve_axial_index: When ``True`` (default), do not flip ``z`` so
            axial slider indices match file / ITK-SNAP order.

    Returns:
        Booleans for each array axis. For typical RAS volumes under
        radiological + preserve, this is ``(False, True, True)`` — in-plane
        A/P and L/R only. LPS identity needs no in-plane flips:
        ``(False, False, False)``.
    """
    try:
        convention_key = normalize_display_convention(convention)
        return volume_display_flips(
            direction,
            ndim=ndim,
            convention=convention_key,
            preserve_axial_index=preserve_axial_index,
        )
    except HABITAPIError as exc:
        raise HABITAPIError(f"view_habitat_napari: {exc}") from exc


def napari_radiological_flips(
    direction: Optional[Sequence[float]],
    *,
    ndim: int,
    preserve_axial_index: bool = True,
) -> Tuple[bool, ...]:
    """
    Radiological flips for napari (alias of :func:`napari_display_flips`).

    Preserves axial index by default so demo basal tips stay on file-order
    slice numbers.
    """
    return napari_display_flips(
        direction,
        ndim=ndim,
        convention="radiological",
        preserve_axial_index=preserve_axial_index,
    )


def _resolve_image_names(
    count: int,
    image_names: Optional[Sequence[str]],
) -> List[str]:
    """
    Build unique layer names for source image layers.

    Args:
        count: Number of image layers.
        image_names: Optional caller-provided names (stems, etc.).

    Returns:
        List of ASCII-safe unique names, length ``count``.
    """
    names: List[str] = []
    if image_names is not None:
        for raw in image_names:
            safe = sanitize_label(str(raw)) or "image"
            names.append(safe)

    if len(names) < count:
        for index in range(len(names), count):
            names.append("image" if count == 1 and index == 0 else f"image_{index + 1}")
    names = names[:count]

    # Ensure uniqueness so napari does not silently rename colliding layers.
    seen: dict[str, int] = {}
    unique: List[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            unique.append(name)
            continue
        seen[name] += 1
        unique.append(f"{name}_{seen[name]}")
    return unique


def _maybe_run_event_loop(napari: Any, *, show: bool) -> None:
    """
    Block on the Qt event loop so a script does not exit while the window is open.

    Creating ``napari.Viewer(show=True)`` alone does not keep a plain Python
    process alive: when the caller's script returns, the interpreter tears down
    Qt and the window appears to flash then close. Calling ``napari.run()``
    starts ``QApplication.exec_()`` until the user closes all viewers.

    ``napari.run()`` is a no-op when an IPython/Jupyter ``%gui qt`` loop (or
    another Qt loop at ``max_loop_level``) is already running, so notebooks and
    nested callers are not double-blocked.

    Args:
        napari: The imported ``napari`` module (from :func:`require`).
        show: When ``False``, skip the event loop (headless / unit tests).
    """
    if not show:
        return
    run = getattr(napari, "run", None)
    if not callable(run):
        return
    try:
        run()
    except Exception:  # noqa: BLE001
        # Do not turn a missing display / Qt teardown into an API crash; the
        # viewer was already constructed and returned to the caller.
        pass


def view_habitat_napari(
    images: ImageInput,
    labels: np.ndarray,
    *,
    opacity: float = 0.45,
    title: str = "HABIT habitat",
    show: bool = True,
    viewer: Optional[Any] = None,
    image_names: Optional[Sequence[str]] = None,
    spacing: Optional[Sequence[float]] = None,
    direction: Optional[Sequence[float]] = None,
    display_convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
) -> Any:
    """
    Open (or populate) a napari viewer with image layer(s) + habitat labels.

    Habitat IDs are added as a napari *labels* layer so background ``0`` stays
    transparent and discrete colours stay categorical. Each greyscale anatomy
    volume is an *image* layer underneath (multi-sequence studies can pass
    several arrays).

    Volumes are oriented using ``direction`` and ``display_convention`` (same
    policy as :func:`~habit.viz.plot_habitat_overlay` for in-plane A-P / L-R).
    Image and habitat layers share the same flip + ``scale`` so they stay
    aligned. When ``direction`` is omitted for 3D data, LPS identity is
    assumed (not RAS). Axial ``z`` is not flipped by default so slider indices
    match file order.

    Args:
        images: Greyscale source volume (2D or 3D; NumPy ``(z, y, x)`` order
            matches ``ImageVolume.data`` / SimpleITK array convention), or a
            sequence of such volumes sharing the same spatial shape as
            ``labels``.
        labels: Integer habitat map with the same spatial shape as each image.
            Background should be ``0``; habitats are ``>= 1``.
        opacity: Labels-layer opacity in ``(0, 1]``.
        title: Window title (ASCII-sanitised for journal-safe defaults).
        show: When ``True`` (default), show the Qt window and **block** until
            the user closes it (via ``napari.run()``). Pass ``False`` for
            headless / unit tests (``napari.Viewer(show=False)``) — then the
            caller owns lifetime and should call ``viewer.close()``.
        viewer: Existing napari ``Viewer`` to reuse, or ``None`` to create one.
        image_names: Optional display names for image layers (e.g. file stems).
            Defaults to ``\"image\"`` / ``\"image_1\"``, ``\"image_2\"``, ...
        spacing: Optional SimpleITK-order spacing ``(x, y[, z])`` from
            ``ImageVolume.spacing``. Converted to napari ``scale=(z, y, x)``
            so anisotropic voxels display correctly.
        direction: Optional SimpleITK direction cosines (9 floats for 3D).
            Same layout as ``ImageVolume.direction``. Controls
            anterior/posterior and left/right in-plane flips.
        display_convention: ``\"radiological\"`` (default), ``\"neurological\"``,
            or ``\"native\"``. See :mod:`habit.viz.orientation`.

    Returns:
        The napari ``Viewer`` instance. With ``show=True`` this returns only
        after the event loop exits (window closed); with ``show=False`` it
        returns immediately and the caller should ``viewer.close()``.

    Raises:
        HABITAPIError: On shape / opacity validation errors.
        OptionalDependencyError: When napari (``view`` extra) is not installed
            or is a broken/partial install without ``Viewer``.
    """
    napari = _require_napari()
    viewer_cls = napari.Viewer

    if not (0.0 < float(opacity) <= 1.0):
        raise HABITAPIError(
            f"view_habitat_napari: opacity must be in (0, 1]; got {opacity}."
        )

    image_vols = _normalize_images(images)
    label_vol = _as_volume(labels, "labels")
    for index, image_vol in enumerate(image_vols):
        if image_vol.shape != label_vol.shape:
            raise HABITAPIError(
                "view_habitat_napari: each image and labels must share the "
                f"same shape; got images[{index}] {image_vol.shape} vs labels "
                f"{label_vol.shape}."
            )

    safe_title = sanitize_label(title) or "HABIT habitat"
    flips = napari_display_flips(
        direction,
        ndim=label_vol.ndim,
        convention=display_convention,
        preserve_axial_index=True,
    )
    # Apply the same flips to every layer so overlays stay registered.
    image_vols = [
        np.ascontiguousarray(
            apply_radiological_flips(image_vol, flips), dtype=np.float32
        )
        for image_vol in image_vols
    ]
    label_data = np.ascontiguousarray(
        apply_radiological_flips(label_vol, flips), dtype=np.int32
    )
    layer_names = _resolve_image_names(len(image_vols), image_names)
    scale = _napari_scale(spacing, label_vol.ndim)

    if viewer is None:
        # show=False keeps Qt from raising on headless CI while still
        # constructing real Image / Labels layers for smoke tests.
        # Use the resolved Viewer class (not bare napari.Viewer attribute
        # access) so incomplete namespace packages raise OptionalDependencyError
        # above instead of AttributeError here.
        viewer = viewer_cls(title=safe_title, show=bool(show))
    else:
        if show:
            try:
                viewer.show()
            except Exception:  # noqa: BLE001
                pass

    add_image_kwargs: dict[str, Any] = {"colormap": "gray"}
    if scale is not None:
        add_image_kwargs["scale"] = scale

    for image_vol, name in zip(image_vols, layer_names):
        viewer.add_image(
            image_vol,
            name=name,
            **add_image_kwargs,
        )

    add_labels_kwargs: dict[str, Any] = {
        "name": "habitats",
        "opacity": float(opacity),
    }
    if scale is not None:
        add_labels_kwargs["scale"] = scale
    viewer.add_labels(label_data, **add_labels_kwargs)

    # Stash flips on the viewer for unit tests / debugging (not a napari API).
    try:
        viewer._habit_radiological_flips = flips  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    # Keep the process alive for script / CLI usage (no-op in IPython %gui qt).
    _maybe_run_event_loop(napari, show=bool(show))
    return viewer
