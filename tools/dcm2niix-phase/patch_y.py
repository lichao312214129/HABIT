"""Point dcm2niix filename code %y at Temporal Position Identifier.

The stock v1.0.20260724 code stores GE private tag (0019,10A2) in
rawDataRunNumber, then prints that field twice for %y. On SIGNA that tag
increments per slice, so a rename puts every slice in its own folder.
%y is changed to a separate field filled only from (0020,0100).
rawDataRunNumber itself is left unchanged because NIfTI volume grouping
still uses it.
"""

from pathlib import Path
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    """Replace one exact occurrence.

    text: full file contents.
    old: exact substring that must occur once.
    new: replacement substring.
    label: short name used in the error if the match count is wrong.
    """
    found: int = text.count(old)
    if found != 1:
        raise SystemExit(f"{label}: expected 1 match, found {found}")
    return text.replace(old, new, 1)


def add_header_field(text: str) -> str:
    """Add temporalPositionIdentifier next to the rawDataRunNumber field.

    text: contents of nii_dicom.h.
    """
    if "temporalPositionIdentifier" in text:
        return text
    lines: list[str] = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped: str = line.strip()
        if stripped.startswith("int rawDataRunNumber") and stripped.endswith(";"):
            indent: str = line[: len(line) - len(line.lstrip(" \t"))]
            lines.insert(
                index + 1,
                indent
                + "int temporalPositionIdentifier; // (0020,0100) only; filename %y\n",
            )
            return "".join(lines)
    raise SystemExit("nii_dicom.h: int rawDataRunNumber field not found")


def patch_reader(text: str) -> str:
    """Initialize the new field and fill it from (0020,0100) only.

    text: contents of console/nii_dicom.cpp.
    """
    text = replace_once(
        text,
        "\td.rawDataRunNumber = -1;\n",
        "\td.rawDataRunNumber = -1;\n\td.temporalPositionIdentifier = -1;\n",
        "init",
    )
    text = replace_once(
        text,
        "\t\tcase kTemporalPosition: // fall through, both kSliceNumberMrPhilips (2001,100A) and kTemporalPosition are is\n"
        "\t\t\tvolumeNumber = dcmStrInt(lLength, &buffer[lPos]);\n"
        "\t\t\t// temporalPositionIdentifier = volumeNumber;\n",
        "\t\tcase kTemporalPosition: // (0020,0100) Temporal Position Identifier\n"
        "\t\t\tvolumeNumber = dcmStrInt(lLength, &buffer[lPos]);\n"
        "\t\t\td.temporalPositionIdentifier = volumeNumber;\n",
        "temporal-tag",
    )
    return text


def patch_filename(text: str) -> str:
    """Print (0020,0100) once for %y, and drop the duplicated block.

    text: contents of console/nii_dicom_batch.cpp.
    """
    duplicated: str = (
        "\t\t\tif ((f == 'Y') && (dcm.rawDataRunNumber >= 0)) {\n"
        '\t\t\t\tsnprintf(newstr, PATH_MAX, "%d", dcm.rawDataRunNumber); // GE (0019,10A2) else (0020,0100)\n'
        "\t\t\t\tstrcat(outname, newstr);\n"
        "\t\t\t}\n"
        "\t\t\tif (f == 'X')\n"
        "\t\t\t\tstrcat(outname, dcm.studyID);\n"
        "\t\t\tif ((f == 'Y') && (dcm.rawDataRunNumber >= 0)) {\n"
        '\t\t\t\tsnprintf(newstr, PATH_MAX, "%d", dcm.rawDataRunNumber); // GE (0019,10A2) else (0020,0100)\n'
        "\t\t\t\tstrcat(outname, newstr);\n"
        "\t\t\t}\n"
    )
    once: str = (
        "\t\t\tif ((f == 'Y') && (dcm.temporalPositionIdentifier >= 0)) {\n"
        '\t\t\t\tsnprintf(newstr, PATH_MAX, "%d", dcm.temporalPositionIdentifier); // (0020,0100) only\n'
        "\t\t\t\tstrcat(outname, newstr);\n"
        "\t\t\t}\n"
        "\t\t\tif (f == 'X')\n"
        "\t\t\t\tstrcat(outname, dcm.studyID);\n"
    )
    text = replace_once(text, duplicated, once, "percent-y")
    text = replace_once(
        text,
        "\t\t\t\tif ((pos < strlen(inname)) && (toupper(inname[pos + 1]) == 'Y') && (dcm.rawDataRunNumber >= 0)) {\n"
        '\t\t\t\t\tchar zeroPad[128] = {""};\n'
        '\t\t\t\t\tsnprintf(zeroPad, 128, "%%0%dd", f - \'0\');\n'
        "\t\t\t\t\tsnprintf(newstr, PATH_MAX, zeroPad, dcm.rawDataRunNumber); // GE (0019,10A2) else (0020,0100)\n",
        "\t\t\t\tif ((pos < strlen(inname)) && (toupper(inname[pos + 1]) == 'Y') && (dcm.temporalPositionIdentifier >= 0)) {\n"
        '\t\t\t\t\tchar zeroPad[128] = {""};\n'
        '\t\t\t\t\tsnprintf(zeroPad, 128, "%%0%dd", f - \'0\');\n'
        "\t\t\t\t\tsnprintf(newstr, PATH_MAX, zeroPad, dcm.temporalPositionIdentifier); // (0020,0100) only\n",
        "percent-y-pad",
    )
    return text


def main() -> None:
    """Patch a dcm2niix checkout given as the first argument."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_y.py <dcm2niix-source-dir>")
    root: Path = Path(sys.argv[1])
    header: Path = root / "console" / "nii_dicom.h"
    reader: Path = root / "console" / "nii_dicom.cpp"
    batch: Path = root / "console" / "nii_dicom_batch.cpp"
    header.write_text(add_header_field(header.read_text(encoding="utf-8")), encoding="utf-8")
    reader.write_text(patch_reader(reader.read_text(encoding="utf-8")), encoding="utf-8")
    batch.write_text(patch_filename(batch.read_text(encoding="utf-8")), encoding="utf-8")
    print("patched", root)


if __name__ == "__main__":
    main()
