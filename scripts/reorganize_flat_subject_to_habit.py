# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.
"""
Standalone tool: reorganize flat subject folders into the HABIT layout.

No HABIT package install required (stdlib only). Safe to copy this file
(and the .bat launcher) to any folder.

Input (per subject under the chosen input folder)::

    sub001/
      fla.nii
      t1.nii
      t1c.nii
      t2.nii
      mask.nii          # shared by all modalities; name must contain "mask"

Output::

    <output_dir>/
      images/sub001/<modality>/<file>
      masks/sub001/<modality>/mask_<modality><suffix>

Usage:
  - Double-click ``reorganize_flat_subject_to_habit.bat`` (GUI)
  - Or: ``python reorganize_flat_subject_to_habit.py`` (GUI)
  - Or: ``python reorganize_flat_subject_to_habit.py --input_dir ...`` (CLI)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Core logic (stdlib only)
# ---------------------------------------------------------------------------

_LOGGER = logging.getLogger("reorganize_to_habit")

# Longer suffixes first so ``.nii.gz`` wins over ``.nii``.
_IMAGE_SUFFIXES: Tuple[str, ...] = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")


def _strip_image_suffix(filename: str) -> str:
    """Return basename without a known image suffix."""
    lower = filename.lower()
    for suffix in _IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    return Path(filename).stem


def _image_suffix(filename: str) -> str:
    """Return the matched image suffix (including the leading dot)."""
    lower = filename.lower()
    for suffix in _IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return filename[len(filename) - len(suffix) :]
    return ""


def _is_image_file(path: Path) -> bool:
    """Return True if ``path`` is a supported medical image file."""
    if not path.is_file() or path.name.startswith("."):
        return False
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in _IMAGE_SUFFIXES)


def _is_mask_file(path: Path) -> bool:
    """Return True if file name contains ``mask`` (case-insensitive)."""
    return _is_image_file(path) and ("mask" in path.name.lower())


def list_subject_dirs(input_dir: Path) -> List[Path]:
    """List immediate child directories (one per subject), sorted."""
    return sorted(
        [p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )


def resolve_subject_files(subject_dir: Path) -> Tuple[List[Path], Optional[Path]]:
    """
    Split subject-folder files into modality images and one shared mask.

    Returns
    -------
    Tuple[List[Path], Optional[Path]]
        ``(image_files, mask_file)``.
    """
    image_files: List[Path] = []
    mask_files: List[Path] = []

    for path in sorted(subject_dir.iterdir()):
        if not _is_image_file(path):
            continue
        if _is_mask_file(path):
            mask_files.append(path)
        else:
            image_files.append(path)

    mask_file: Optional[Path] = None
    if not mask_files:
        _LOGGER.warning("No mask file found in %s", subject_dir)
    elif len(mask_files) > 1:
        mask_file = mask_files[0]
        _LOGGER.warning(
            "Multiple mask files in %s; using %s (ignored: %s)",
            subject_dir,
            mask_file.name,
            ", ".join(p.name for p in mask_files[1:]),
        )
    else:
        mask_file = mask_files[0]

    return image_files, mask_file


def default_output_dir(input_dir: Path) -> Path:
    """Sibling of ``input_dir`` named ``<input_name>_habit``."""
    return input_dir.parent / f"{input_dir.name}_habit"


def reorganize_subject(
    subject_dir: Path,
    images_root: Path,
    masks_root: Path,
    dry_run: bool = False,
) -> int:
    """
    Copy one subject into HABIT ``images/`` and ``masks/`` layout.

    Returns
    -------
    int
        Number of modality files handled for this subject.
    """
    subject_id: str = subject_dir.name
    image_files, mask_file = resolve_subject_files(subject_dir)

    if not image_files:
        _LOGGER.error("No modality images found in %s; skipping", subject_dir)
        return 0

    for src_img in image_files:
        modality: str = _strip_image_suffix(src_img.name)
        if not modality:
            _LOGGER.error("Cannot parse modality from %s; skipping file", src_img)
            continue

        dst_img_dir: Path = images_root / subject_id / modality
        dst_img: Path = dst_img_dir / src_img.name

        if dry_run:
            _LOGGER.info("[dry-run] copy %s -> %s", src_img, dst_img)
        else:
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dst_img)

        if mask_file is None:
            continue

        mask_suffix: str = _image_suffix(mask_file.name) or Path(mask_file.name).suffix
        dst_mask_dir: Path = masks_root / subject_id / modality
        dst_mask: Path = dst_mask_dir / f"mask_{modality}{mask_suffix}"

        if dry_run:
            _LOGGER.info("[dry-run] copy %s -> %s", mask_file, dst_mask)
        else:
            dst_mask_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_file, dst_mask)

    return len(image_files)


def reorganize_dataset(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    """
    Reorganize all subjects under ``input_dir`` into HABIT layout.

    Returns
    -------
    Path
        Resolved output directory path.
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if output_dir is None:
        output_dir = default_output_dir(input_dir)
    else:
        output_dir = output_dir.resolve()

    images_root: Path = output_dir / "images"
    masks_root: Path = output_dir / "masks"

    subject_dirs: List[Path] = list_subject_dirs(input_dir)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject subdirectories found in: {input_dir}")

    _LOGGER.info("Input : %s", input_dir)
    _LOGGER.info("Output: %s", output_dir)
    _LOGGER.info("Subjects: %d", len(subject_dirs))
    if dry_run:
        _LOGGER.info("Dry-run mode: no files will be written")

    if not dry_run:
        images_root.mkdir(parents=True, exist_ok=True)
        masks_root.mkdir(parents=True, exist_ok=True)

    n_modalities: int = 0
    total: int = len(subject_dirs)
    for idx, subject_dir in enumerate(subject_dirs, start=1):
        _LOGGER.info("[%d/%d] %s", idx, total, subject_dir.name)
        n_modalities += reorganize_subject(
            subject_dir=subject_dir,
            images_root=images_root,
            masks_root=masks_root,
            dry_run=dry_run,
        )

    _LOGGER.info(
        "Done. Subjects=%d, modality copies=%d, output=%s",
        len(subject_dirs),
        n_modalities,
        output_dir,
    )
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. Empty argv means launch GUI."""
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize flat subject folders (shared mask) into HABIT "
            "images/<subject>/<modality>/ + masks/<subject>/<modality>/ layout."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Directory that directly contains subject folders",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="HABIT output root (default: sibling <input_name>_habit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned copies without writing files",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force open the GUI",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def _configure_logging(debug: bool = False) -> None:
    """Configure root logger once for CLI or GUI."""
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(levelname)s: %(message)s",
        )
    else:
        root.setLevel(level)


def _run_cli(args: argparse.Namespace) -> int:
    """Run reorganization from CLI flags."""
    if args.input_dir is None:
        _LOGGER.error("--input_dir is required in CLI mode (or run with no args for GUI)")
        return 1
    try:
        reorganize_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        _LOGGER.error("%s", exc)
        return 1
    return 0


# ---------------------------------------------------------------------------
# GUI (tkinter, stdlib)
# ---------------------------------------------------------------------------


def run_gui() -> int:
    """Open the folder-picker GUI. Returns process exit code."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, scrolledtext, ttk
    except ImportError:
        print(
            "ERROR: tkinter is not available in this Python.\n"
            "Install a Python build that includes tkinter, or use CLI:\n"
            "  python reorganize_flat_subject_to_habit.py --input_dir <path>",
            file=sys.stderr,
        )
        return 1

    class _TextHandler(logging.Handler):
        """Forward log records into a Tk text widget (thread-safe via after)."""

        def __init__(self, widget: scrolledtext.ScrolledText) -> None:
            super().__init__()
            self._widget = widget

        def emit(self, record: logging.LogRecord) -> None:
            msg: str = self.format(record) + "\n"
            self._widget.after(0, self._append, msg)

        def _append(self, msg: str) -> None:
            self._widget.configure(state=tk.NORMAL)
            self._widget.insert(tk.END, msg)
            self._widget.see(tk.END)
            self._widget.configure(state=tk.DISABLED)

    class ReorganizeApp(tk.Tk):
        """Minimal GUI: pick folders and run copy."""

        def __init__(self) -> None:
            super().__init__()
            self.title("HABIT Data Reorganize")
            self.geometry("720x480")
            self.minsize(560, 400)

            self._input_var = tk.StringVar()
            self._output_var = tk.StringVar()
            self._dry_run_var = tk.BooleanVar(value=False)
            self._busy = False

            self._build_ui()
            self._setup_logging()

        def _build_ui(self) -> None:
            pad = {"padx": 10, "pady": 6}
            frm = ttk.Frame(self)
            frm.pack(fill=tk.BOTH, expand=True, **pad)

            ttk.Label(
                frm, text="Input folder (contains sub001, sub002, ...)"
            ).grid(row=0, column=0, columnspan=3, sticky="w")
            ttk.Entry(frm, textvariable=self._input_var).grid(
                row=1, column=0, columnspan=2, sticky="ew", padx=(0, 6)
            )
            ttk.Button(frm, text="Browse...", command=self._browse_input).grid(
                row=1, column=2, sticky="e"
            )

            ttk.Label(
                frm,
                text="Output folder (optional; leave empty for sibling *_habit)",
            ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(12, 0))
            ttk.Entry(frm, textvariable=self._output_var).grid(
                row=3, column=0, columnspan=2, sticky="ew", padx=(0, 6)
            )
            ttk.Button(frm, text="Browse...", command=self._browse_output).grid(
                row=3, column=2, sticky="e"
            )

            opts = ttk.Frame(frm)
            opts.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
            ttk.Checkbutton(
                opts,
                text="Dry-run (preview only, do not copy)",
                variable=self._dry_run_var,
            ).pack(side=tk.LEFT)
            self._run_btn = ttk.Button(opts, text="Run", command=self._on_run)
            self._run_btn.pack(side=tk.RIGHT)

            ttk.Label(frm, text="Log").grid(row=5, column=0, sticky="w", pady=(12, 0))
            self._log = scrolledtext.ScrolledText(frm, height=16, state=tk.DISABLED)
            self._log.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(4, 0))

            frm.columnconfigure(0, weight=1)
            frm.rowconfigure(6, weight=1)

        def _setup_logging(self) -> None:
            handler = _TextHandler(self._log)
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            root = logging.getLogger()
            root.setLevel(logging.INFO)
            for h in list(root.handlers):
                if isinstance(h, _TextHandler):
                    root.removeHandler(h)
            root.addHandler(handler)
            # Also keep console if present.
            if not any(
                isinstance(h, logging.StreamHandler) and not isinstance(h, _TextHandler)
                for h in root.handlers
            ):
                sh = logging.StreamHandler(sys.stdout)
                sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
                root.addHandler(sh)

        def _browse_input(self) -> None:
            path = filedialog.askdirectory(title="Select input folder")
            if path:
                self._input_var.set(path)
                if not self._output_var.get().strip():
                    self._output_var.set(str(default_output_dir(Path(path))))

        def _browse_output(self) -> None:
            path = filedialog.askdirectory(title="Select output folder")
            if path:
                self._output_var.set(path)

        def _on_run(self) -> None:
            if self._busy:
                return

            input_text: str = self._input_var.get().strip()
            if not input_text:
                messagebox.showwarning("Missing input", "Please select the input folder.")
                return

            input_dir = Path(input_text)
            if not input_dir.is_dir():
                messagebox.showerror(
                    "Invalid input", f"Folder does not exist:\n{input_dir}"
                )
                return

            output_text: str = self._output_var.get().strip()
            output_dir: Optional[Path] = Path(output_text) if output_text else None
            dry_run: bool = bool(self._dry_run_var.get())

            self._busy = True
            self._run_btn.configure(state=tk.DISABLED)

            def worker() -> None:
                try:
                    out = reorganize_dataset(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        dry_run=dry_run,
                    )
                    self.after(0, self._on_success, str(out), dry_run)
                except Exception as exc:
                    self.after(0, self._on_error, str(exc))

            threading.Thread(target=worker, daemon=True).start()

        def _on_success(self, output_dir: str, dry_run: bool) -> None:
            self._busy = False
            self._run_btn.configure(state=tk.NORMAL)
            if dry_run:
                messagebox.showinfo(
                    "Dry-run finished",
                    f"Preview complete.\nWould write to:\n{output_dir}",
                )
            else:
                messagebox.showinfo(
                    "Done", f"Reorganize finished.\nOutput:\n{output_dir}"
                )

        def _on_error(self, message: str) -> None:
            self._busy = False
            self._run_btn.configure(state=tk.NORMAL)
            messagebox.showerror("Failed", message)

    app = ReorganizeApp()
    app.mainloop()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point: GUI when no meaningful CLI args; otherwise CLI.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Argument list without program name.

    Returns
    -------
    int
        Process exit code.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Double-click / no args -> GUI.
    if len(argv) == 0:
        _configure_logging(False)
        return run_gui()

    args = _parse_args(argv)
    _configure_logging(args.debug)

    if args.gui or args.input_dir is None:
        return run_gui()
    return _run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
