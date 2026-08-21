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
"""API-first report declaration: what to persist and what to draw.

:class:`Report` is the presentation / persistence counterpart of
:class:`~habit.spec.specs.HabitatSpec`. The spec is the scientific identity
(and enters fingerprints / checkpoints). A Report is a run-scoped object
you pass to :meth:`~habit.recipes.study.Study.fit` / ``fit_predict``:

* ``persist`` -- artefacts written the moment a subject completes;
* ``retain`` -- what stays in the in-memory :class:`StudyResult`;
* ``figures`` -- :class:`FigureAtom` objects drawn in the parent process
  after persist, before heavy payloads are dropped;
* ``writer`` -- where artefacts go (directory, or a third-party writer).

Nothing here belongs in ``HabitatSpec``. Changing a colormap or adding an
overlay must not invalidate a radiomics checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from habit.contracts.habitat import HabitatMap, HabitatModel
from habit.contracts.ops import ResultWriter
from habit.contracts.subject import Subject
from habit.exceptions import HABITAPIError
from habit.utils.write_access import write_via_temp_then_replace

__all__ = [
    "PERSIST_HABITAT_MAP",
    "PERSIST_SUBJECT_MODEL",
    "PERSIST_KINDS",
    "RETAIN_MODES",
    "FIGURE_LAYOUTS",
    "SubjectContext",
    "FigureAtom",
    "Report",
    "coerce_report",
]

#: Persist the subject's habitat label image through the writer.
PERSIST_HABITAT_MAP = "habitat_map"
#: Persist the subject's own ``.habitatmodel`` (one-step design).
PERSIST_SUBJECT_MODEL = "subject_model"
#: Persist kinds the first-phase consumer understands.
PERSIST_KINDS = (PERSIST_HABITAT_MAP, PERSIST_SUBJECT_MODEL)

#: In-memory retention modes. ``tables`` requires a writer so maps are not lost.
RETAIN_MODES = ("all", "maps", "tables")

#: How per-subject PNGs are arranged under ``figure_dir``.
#: ``flat`` writes ``<figure_dir>/<stem>.png`` (historical default).
#: ``by_subject`` writes ``<figure_dir>/<subject_id>/<kind>.png``.
FIGURE_LAYOUTS = ("flat", "by_subject")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SubjectContext:
    """
    Parent-process view of one completed subject.

    Attributes:
        subject: The cohort subject (images still addressable).
        habitat_map: Habitat label image, still in memory at consume time.
        model: This subject's habitat definition (one-step) or the cohort
            model being applied.
    """

    subject: Subject
    habitat_map: HabitatMap
    model: HabitatModel


@runtime_checkable
class FigureAtom(Protocol):
    """
    One drawable, persistable figure.

    Implement this protocol to plug a custom figure into ``Report.figures``
    without subclassing HABIT types. ``draw`` must return a matplotlib
    Figure or ``None`` (skip). It must not write files.
    """

    at: str

    def stem(self, subject_id: str) -> str:
        """Return the destination filename stem (no directory, no suffix).

        Built-in atoms use ``<subject_id>_<kind>``. ``Report`` writes that
        stem as-is when ``figure_layout="flat"``, and strips the leading
        ``<subject_id>_`` when ``figure_layout="by_subject"``.
        """
        ...

    def draw(self, ctx: SubjectContext) -> Optional[Any]:
        """Return a matplotlib Figure, or ``None`` to skip this subject."""
        ...


@dataclass
class Report:
    """
    What to leave on disk and what to keep in memory for one study run.

    Construct this in Python (notebook, script, service) and pass it as
    ``report=`` to :meth:`~habit.recipes.study.Study.fit_predict`. YAML / CLI
    may serialise a subset later; they are not the authoring surface.

    Attributes:
        persist: Artefacts written per completed subject. First-phase
            kinds: ``"habitat_map"``, ``"subject_model"``. Empty by
            default -- passing a writer without persist writes nothing.
        retain: ``"all"`` keeps every artefact in memory (historical
            default); ``"maps"`` drops voxel-level clustering units;
            ``"tables"`` additionally drops habitat maps and therefore
            requires ``writer`` plus ``"habitat_map"`` in ``persist``.
        figures: Figure atoms drawn after persist, before retention
            stripping. Empty by default.
        writer: Streaming destination. Required when ``retain="tables"``
            or when ``persist`` is non-empty.
        figure_dir: Directory for PNGs. Defaults to ``<writer.root>/figures``
            when the writer exposes a ``root``.
        figure_layout: How those PNGs are arranged. ``"flat"`` (default)
            writes ``<figure_dir>/<stem>.png``; ``"by_subject"`` writes
            ``<figure_dir>/<subject_id>/<kind>.png``, stripping a leading
            ``<subject_id>_`` from the atom stem so the filename is just
            the figure kind.
        style: :func:`~habit.viz.use_style` preset used when saving figures.
        on_subject_complete: Optional escape-hatch callback
            ``(subject, habitat_map, model)`` fired after persist and
            figures. Prefer a custom :class:`FigureAtom` when the work is
            a figure.
    """

    persist: Tuple[str, ...] = ()
    retain: str = "all"
    figures: Tuple[FigureAtom, ...] = ()
    writer: Optional[ResultWriter] = None
    figure_dir: Optional[Path] = None
    figure_layout: str = "flat"
    style: str = "radiology"
    on_subject_complete: Optional[
        Callable[[Subject, HabitatMap, HabitatModel], None]
    ] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Canonicalise sequences and reject unknown persist / retain / layout values."""
        persist = tuple(str(item) for item in self.persist)
        unknown = [item for item in persist if item not in PERSIST_KINDS]
        if unknown:
            raise HABITAPIError(
                f"Unknown Report.persist kind(s) {unknown!r}; expected a "
                f"subset of {PERSIST_KINDS}."
            )
        if self.retain not in RETAIN_MODES:
            raise HABITAPIError(
                f"Unknown Report.retain {self.retain!r}; expected one of "
                f"{RETAIN_MODES}."
            )
        if self.figure_layout not in FIGURE_LAYOUTS:
            raise HABITAPIError(
                f"Unknown Report.figure_layout {self.figure_layout!r}; "
                f"expected one of {FIGURE_LAYOUTS}."
            )
        object.__setattr__(self, "persist", persist)
        object.__setattr__(self, "figures", tuple(self.figures))
        if self.figure_dir is not None:
            object.__setattr__(self, "figure_dir", Path(self.figure_dir))
        if self.retain == "tables":
            if self.writer is None:
                raise HABITAPIError(
                    "Report(retain='tables') drops habitat maps from memory; "
                    "pass writer= so each map is persisted as its subject "
                    "completes."
                )
            if PERSIST_HABITAT_MAP not in persist:
                raise HABITAPIError(
                    "Report(retain='tables') requires persist to include "
                    "'habitat_map' so the dropped maps still land on disk."
                )
        if persist and self.writer is None:
            raise HABITAPIError(
                "Report.persist is non-empty; pass writer= (e.g. "
                "DirectoryResultWriter) so artefacts have a destination."
            )

    @property
    def streams(self) -> bool:
        """Return whether this report does any per-subject work."""
        return bool(
            self.writer is not None
            or self.retain != "all"
            or self.figures
            or self.on_subject_complete is not None
        )

    def resolve_figure_dir(self) -> Optional[Path]:
        """
        Return the PNG directory, or ``None`` when figures cannot be written.

        Returns:
            ``figure_dir`` when set; otherwise ``<writer.root>/figures``
            when the writer exposes a filesystem root.
        """
        if self.figure_dir is not None:
            return Path(self.figure_dir)
        root = getattr(self.writer, "root", None)
        if root is not None:
            return Path(root) / "figures"
        return None

    def resolve_figure_path(self, subject_id: str, stem: str) -> Path:
        """
        Return the PNG path for one atom under the current figure_layout.

        ``flat`` writes ``<figure_dir>/<stem>.png``. ``by_subject`` writes
        ``<figure_dir>/<subject_id>/<kind>.png``, where ``kind`` is
        ``stem`` with a leading ``<subject_id>_`` stripped so built-in
        atoms do not repeat the subject id in the filename.

        Args:
            subject_id: Subject identifier; used as the subdirectory name
                when ``figure_layout="by_subject"``.
            stem: Filename stem from ``FigureAtom.stem(subject_id)`` (no
                suffix). Built-in atoms return ``<subject_id>_<kind>``.

        Returns:
            Destination path including the ``.png`` suffix.

        Raises:
            HABITAPIError: If no figure directory can be resolved.
        """
        destination = self.resolve_figure_dir()
        if destination is None:
            raise HABITAPIError(
                "Report.figures requires figure_dir= or a writer that "
                "exposes a filesystem root (DirectoryResultWriter.root)."
            )
        subject_id = str(subject_id)
        if self.figure_layout == "flat":
            return Path(destination) / f"{stem}.png"
        prefix = f"{subject_id}_"
        kind = stem[len(prefix):] if stem.startswith(prefix) else stem
        if not kind:
            kind = stem
        return Path(destination) / subject_id / f"{kind}.png"

    def consume_subject(self, ctx: SubjectContext) -> None:
        """
        Persist, draw, and notify for one completed subject.

        Runs in the parent process, including for checkpoint-resumed
        subjects, so a crash between the backend's checkpoint write and
        this call is repaired on resume.

        Args:
            ctx: The completed subject's map, model, and images.

        Raises:
            HABITAPIError: If figures are declared but no figure directory
                can be resolved.
        """
        writer = self.writer
        if writer is not None:
            if PERSIST_HABITAT_MAP in self.persist:
                writer.write_habitat_map(ctx.habitat_map)
            if PERSIST_SUBJECT_MODEL in self.persist:
                write_model = getattr(writer, "write_subject_model", None)
                if not callable(write_model):
                    raise HABITAPIError(
                        "Report.persist includes 'subject_model' but this "
                        "writer has no write_subject_model method. Use "
                        "DirectoryResultWriter or implement that extra."
                    )
                write_model(ctx.model, str(ctx.subject.subject_id))
        if self.figures:
            self._write_figures(ctx)
        if self.on_subject_complete is not None:
            self.on_subject_complete(ctx.subject, ctx.habitat_map, ctx.model)

    def _write_figures(self, ctx: SubjectContext) -> None:
        """Draw each figure atom and atomically replace its PNG."""
        destination = self.resolve_figure_dir()
        if destination is None:
            raise HABITAPIError(
                "Report.figures requires figure_dir= or a writer that "
                "exposes a filesystem root (DirectoryResultWriter.root)."
            )
        destination.mkdir(parents=True, exist_ok=True)
        from habit.utils.optional_deps import require
        from habit.viz import use_style

        plt = require(
            "matplotlib.pyplot",
            extra="viz",
            purpose="per-subject figures written by habit.report.Report",
        )
        subject_id = str(ctx.subject.subject_id)
        with use_style(self.style) as style:
            dpi = int(getattr(style, "dpi", 300))
            for atom in self.figures:
                figure = atom.draw(ctx)
                if figure is None:
                    continue
                path = self.resolve_figure_path(subject_id, atom.stem(subject_id))
                path.parent.mkdir(parents=True, exist_ok=True)

                def _write(tmp_path: Path, fig: Any = figure) -> None:
                    fig.savefig(tmp_path, dpi=dpi, bbox_inches="tight")

                try:
                    write_via_temp_then_replace(path, _write)
                finally:
                    plt.close(figure)


def coerce_report(
    report: Optional[Report],
    *,
    writer: Optional[ResultWriter] = None,
    retain: str = "all",
    on_subject_complete: Optional[
        Callable[[Subject, HabitatMap, HabitatModel], None]
    ] = None,
    persist_subject_models: bool = True,
) -> Optional[Report]:
    """
    Build or complete a :class:`Report` from the Study call-site kwargs.

    ``report=`` is the primary API. The older ``writer=`` / ``retain=`` /
    ``on_subject_complete=`` kwargs still work: they construct an implicit
    Report, or fill a field the explicit Report left empty.

    Args:
        report: Explicit report, or ``None``.
        writer: Streaming writer from the call site.
        retain: Retention mode from the call site (``"all"`` means
            "not specified" when an explicit report is also given).
        on_subject_complete: Optional callback from the call site.
        persist_subject_models: When building an implicit report with a
            writer, also persist ``<subject_id>.habitatmodel``.

    Returns:
        A completed Report, or ``None`` when nothing streams (historical
        in-memory default).

    Raises:
        HABITAPIError: If an explicit report and a kwarg disagree.
    """
    if report is None:
        if writer is None and retain == "all" and on_subject_complete is None:
            return None
        persist: Tuple[str, ...]
        if writer is None:
            persist = ()
        elif persist_subject_models:
            persist = (PERSIST_HABITAT_MAP, PERSIST_SUBJECT_MODEL)
        else:
            persist = (PERSIST_HABITAT_MAP,)
        return Report(
            persist=persist,
            retain=retain,
            writer=writer,
            on_subject_complete=on_subject_complete,
        )
    updated = report
    if writer is not None:
        if report.writer is not None and report.writer is not writer:
            raise HABITAPIError(
                "Pass writer on Report(...) or as writer=, not both."
            )
        if report.writer is None:
            updated = replace(updated, writer=writer)
    if retain != "all" and report.retain == "all":
        updated = replace(updated, retain=retain)
    elif retain != "all" and report.retain != retain:
        raise HABITAPIError(
            f"retain={retain!r} disagrees with Report.retain={report.retain!r}."
        )
    if on_subject_complete is not None:
        if (
            report.on_subject_complete is not None
            and report.on_subject_complete is not on_subject_complete
        ):
            raise HABITAPIError(
                "Pass on_subject_complete on Report(...) or as a keyword, "
                "not both."
            )
        if report.on_subject_complete is None:
            updated = replace(updated, on_subject_complete=on_subject_complete)
    if updated is not report:
        # Re-run __post_init__ invariants after fills (retain='tables'
        # may now have a writer that the original Report lacked).
        return Report(
            persist=updated.persist,
            retain=updated.retain,
            figures=updated.figures,
            writer=updated.writer,
            figure_dir=updated.figure_dir,
            figure_layout=updated.figure_layout,
            style=updated.style,
            on_subject_complete=updated.on_subject_complete,
        )
    return updated
