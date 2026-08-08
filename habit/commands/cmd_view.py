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
"""Implementation of the ``habit view`` command.

L5 wiring only: read image file(s) + a habitat map, call ``habit.viz`` helpers,
and either launch napari (preferred, optional ``view`` extra) or write a PNG
(matplotlib fallback / forced backend). All drawing / viewing logic lives in
``habit.viz``; this module is a thin sink.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import click

from habit.commands.common import echo_error, echo_success

#: Supported ``habit view --backend`` values.
#: ``auto`` (default) and ``napari`` prefer the interactive viewer and fall
#: back to a matplotlib PNG when the ``[view]`` extra is missing.
_VIEW_BACKENDS = ("auto", "napari", "matplotlib")

#: Install hint printed when napari is unavailable (fallback path).
#: Prefer documenting the clone path: published PyPI 1.0.x wheels do not
#: declare ``[view]`` yet; China mirrors may also omit ``habitat-analysis``.
_VIEW_EXTRA_HINT = (
    'pip install "habitat-analysis[view]" '
    '(from a clone: pip install -e ".[view]"; '
    'or: pip install "napari[pyqt5]" -i https://pypi.org/simple)'
)


def _open_with_system_viewer(path: Path) -> None:
    """Open a file with the platform default application."""
    target = str(path.resolve())
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.run(["open", target], check=False)
        return
    subprocess.run(["xdg-open", target], check=False)


def _simpleitk_overlay_snippet(
    image_path: Union[str, Path],
    habitat_path: Union[str, Path],
) -> str:
    """Short SimpleITK snippet printed after a successful overlay."""
    return (
        "import SimpleITK as sitk\n"
        f"image = sitk.ReadImage(r\"{Path(image_path)}\")\n"
        f"habitats = sitk.ReadImage(r\"{Path(habitat_path)}\")\n"
        "overlay = sitk.LabelOverlay(\n"
        "    sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8),\n"
        "    sitk.Cast(habitats, sitk.sitkUInt8),\n"
        "    opacity=0.45,\n"
        ")\n"
        "sitk.Show(overlay, 'HABIT habitat overlay')\n"
    )


def _format_image_list(image_paths: Sequence[Path]) -> str:
    """Human-readable list of source image paths for CLI summary lines."""
    if len(image_paths) == 1:
        return str(image_paths[0])
    return "; ".join(str(path) for path in image_paths)


def _run_view_matplotlib(
    image_paths: Sequence[Path],
    habitat_path: Path,
    *,
    output: Optional[str],
    no_open: bool,
    alpha: float,
    display_convention: str = "radiological",
) -> Path:
    """
    Write a PNG overlay via :func:`habit.viz.plot_habitat_overlay`.

    Matplotlib overlays a single greyscale volume. When several source images
    are provided, only the first is used and a warning is printed (extra
    series are for the napari backend).

    Returns:
        Path to the written PNG.
    """
    from habit.api.image import read_image, read_mask
    from habit.viz import plot_habitat_overlay

    if len(image_paths) > 1:
        click.echo(
            "Note: matplotlib backend uses the first source image only; "
            "extra --image series are shown with --backend napari (or auto)."
        )

    image_path = image_paths[0]
    if output is None:
        png_path = habitat_path.with_name(f"{habitat_path.stem}_overlay.png")
    else:
        png_path = Path(output)

    image_vol = read_image(image_path)
    label_vol = read_mask(habitat_path)
    fig = plot_habitat_overlay(
        image_vol.data,
        label_vol.data,
        alpha=alpha,
        title=f"Habitat overlay — {habitat_path.name}",
        direction=getattr(image_vol, "direction", None),
        spacing=getattr(image_vol, "spacing", None),
        display_convention=display_convention,  # type: ignore[arg-type]
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass

    if not no_open:
        _open_with_system_viewer(png_path)

    return png_path


def _run_view_napari(
    image_paths: Sequence[Path],
    habitat_path: Path,
    *,
    no_open: bool,
    alpha: float,
    display_convention: str = "radiological",
) -> None:
    """
    Launch napari with one or more image layers + habitat labels.

    ``no_open`` maps to ``show=False`` so tests can exercise the path without
    a visible Qt window. With ``show=True`` (default),
    :func:`~habit.viz.view_habitat_napari` blocks via ``napari.run()`` until
    the user closes the window.
    """
    from habit.api.image import read_image, read_mask
    from habit.viz import view_habitat_napari

    image_vols = [read_image(path) for path in image_paths]
    label_vol = read_mask(habitat_path)
    arrays = [vol.data for vol in image_vols]
    names = [path.stem for path in image_paths]
    # Geometry from the first series: spacing → napari scale; direction →
    # radiological flips shared by all image + habitat layers.
    spacing = getattr(image_vols[0], "spacing", None)
    direction = getattr(image_vols[0], "direction", None)

    viewer = view_habitat_napari(
        arrays if len(arrays) > 1 else arrays[0],
        label_vol.data,
        opacity=alpha,
        title=f"HABIT habitat — {habitat_path.name}",
        show=not no_open,
        image_names=names,
        spacing=spacing,
        direction=direction,
        display_convention=display_convention,  # type: ignore[arg-type]
    )
    if no_open:
        # Headless / smoke path: construct layers then tear down immediately.
        try:
            viewer.close()
        except Exception:  # noqa: BLE001
            pass


def _try_napari_then_matplotlib(
    image_paths: Sequence[Path],
    habitat_path: Path,
    *,
    output: Optional[str],
    no_open: bool,
    alpha: float,
    display_convention: str = "radiological",
) -> Tuple[str, Optional[Path]]:
    """
    Prefer napari; on missing ``[view]`` extra, fall back to a matplotlib PNG.

    Args:
        image_paths: Source image paths.
        habitat_path: Habitat label map path.
        output: Optional PNG destination used only on the matplotlib path.
        no_open: Passed through to both backends.
        alpha: Overlay / labels opacity.
        display_convention: Radiological / neurological / native.

    Returns:
        ``(effective_backend, png_path)`` where ``effective_backend`` is
        ``"napari"`` or ``"matplotlib"``, and ``png_path`` is set only for the
        matplotlib path.
    """
    from habit.exceptions import OptionalDependencyError

    try:
        _run_view_napari(
            image_paths,
            habitat_path,
            no_open=no_open,
            alpha=alpha,
            display_convention=display_convention,
        )
        return "napari", None
    except OptionalDependencyError as exc:
        click.echo("")
        click.echo(
            f"Note: napari is not available ({exc}). "
            f"Install the optional viewer with:\n"
            f"  {_VIEW_EXTRA_HINT}\n"
            "Falling back to matplotlib PNG overlay."
        )
        click.echo("")
        png_path = _run_view_matplotlib(
            image_paths,
            habitat_path,
            output=output,
            no_open=no_open,
            alpha=alpha,
            display_convention=display_convention,
        )
        return "matplotlib", png_path


def run_view(
    image: Union[str, Sequence[str]],
    habitat: str,
    *,
    output: Optional[str] = None,
    no_open: bool = False,
    alpha: float = 0.45,
    backend: str = "auto",
    convention: str = "radiological",
) -> None:
    """
    Overlay a habitat map on one or more source images (napari or PNG).

    Args:
        image: Path to a greyscale source image, or a sequence of paths
            (multi-sequence; all series load in napari; matplotlib uses the
            first and warns).
        habitat: Path to the habitat label map (``*_habitats.nrrd`` etc.).
        output: Optional PNG destination (matplotlib path only); defaults
            next to the habitat map.
        no_open: Matplotlib: write PNG but do not launch the OS viewer.
            Napari: construct layers with ``show=False`` and close immediately.
        alpha: Habitat colour / labels opacity in ``(0, 1]``.
        backend: ``"auto"`` (default, prefer napari then fall back to PNG),
            ``"napari"`` (same fallback when ``[view]`` is missing), or
            ``"matplotlib"`` (force static PNG; needs ``[viz]``).
        convention: Display orientation ``\"radiological\"`` (default),
            ``\"neurological\"``, or ``\"native\"``.
    """
    from habit.exceptions import HABITAPIError, OptionalDependencyError
    from habit.viz.orientation import normalize_display_convention

    if isinstance(image, (str, Path)):
        image_paths: List[Path] = [Path(image)]
    else:
        image_paths = [Path(path) for path in image]
        if not image_paths:
            echo_error("Error: At least one source image path is required.")
            raise SystemExit(1)

    habitat_path = Path(habitat)
    for image_path in image_paths:
        if not image_path.is_file():
            echo_error(f"Error: Source image not found: {image_path}")
            raise SystemExit(1)
    if not habitat_path.is_file():
        echo_error(f"Error: Habitat map not found: {habitat_path}")
        raise SystemExit(1)

    backend_key = (backend or "auto").strip().lower()
    if backend_key not in _VIEW_BACKENDS:
        echo_error(
            f"Error: Unknown view backend {backend!r}. "
            f"Choose one of: {', '.join(_VIEW_BACKENDS)}."
        )
        raise SystemExit(1)

    try:
        display_convention = normalize_display_convention(convention)
    except HABITAPIError as exc:
        echo_error(f"Error: {exc}")
        raise SystemExit(1) from exc

    fell_back_from_napari = False
    try:
        if backend_key in ("auto", "napari"):
            effective_backend, png_path = _try_napari_then_matplotlib(
                image_paths,
                habitat_path,
                output=output,
                no_open=no_open,
                alpha=alpha,
                display_convention=display_convention,
            )
            fell_back_from_napari = effective_backend == "matplotlib"
        else:
            effective_backend = "matplotlib"
            png_path = _run_view_matplotlib(
                image_paths,
                habitat_path,
                output=output,
                no_open=no_open,
                alpha=alpha,
                display_convention=display_convention,
            )
    except FileNotFoundError as exc:
        echo_error(f"Error: {exc}")
        raise SystemExit(1) from exc
    except (HABITAPIError, OptionalDependencyError, ValueError) as exc:
        # OptionalDependencyError from matplotlib / viz still hard-fails;
        # napari-missing is handled inside _try_napari_then_matplotlib.
        echo_error(f"Error: {exc}")
        raise SystemExit(1) from exc

    click.echo("")
    click.echo("Habitat overlay preview")
    click.echo(f"  Image     : {_format_image_list(image_paths)}")
    click.echo(f"  Habitat   : {habitat_path}")
    if fell_back_from_napari:
        click.echo(
            f"  Backend   : matplotlib (fallback; requested {backend_key})"
        )
    else:
        click.echo(f"  Backend   : {effective_backend}")
    if effective_backend == "napari":
        n_img = len(image_paths)
        if no_open:
            click.echo(
                "  Viewer    : napari layers built (--no-open; window not shown)"
            )
        else:
            click.echo(
                f"  Viewer    : napari ({n_img} image layer(s) + habitats labels)"
            )
    else:
        assert png_path is not None
        click.echo(f"  PNG       : {png_path}")
        if no_open:
            click.echo("  Viewer    : skipped (--no-open); open the PNG manually")
        else:
            click.echo("  Viewer    : opened with the system default image viewer")

    primary_image = image_paths[0]
    click.echo("")
    click.echo("Also view with ITK-SNAP / 3D Slicer:")
    click.echo(f"  1. Open the source image: {primary_image}")
    if len(image_paths) > 1:
        for extra in image_paths[1:]:
            click.echo(f"     (additional series: {extra})")
    click.echo(f"  2. Add the habitat map as a segmentation / overlay: {habitat_path}")
    click.echo("")
    click.echo("Or with SimpleITK (Python):")
    click.echo(_simpleitk_overlay_snippet(primary_image, habitat_path))
    if effective_backend == "matplotlib" and not fell_back_from_napari:
        # Forced matplotlib path: remind users about the interactive option.
        click.echo("")
        click.echo("Optional interactive napari (install once):")
        click.echo(f"  {_VIEW_EXTRA_HINT}")
        if len(image_paths) == 1:
            click.echo(
                f"  habit view \"{primary_image}\" \"{habitat_path}\""
            )
        else:
            img_flags = " ".join(f'--image "{path}"' for path in image_paths)
            click.echo(
                f"  habit view {img_flags} --habitat \"{habitat_path}\""
            )

    echo_success(
        "Overlay ready. Coloured regions are habitats on top of the source image."
    )
