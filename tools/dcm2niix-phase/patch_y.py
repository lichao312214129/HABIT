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
    # Several replacements keep the old text as a prefix of the new text
    # (for example the rawDataRunNumber initializer). After one successful
    # apply, `old` is still found inside `new`, so a second build.sh run
    # must stop once `new` is already present.
    if new in text:
        return text
    found: int = text.count(old)
    if found != 1:
        raise SystemExit(f"{label}: expected 1 match, found {found}")
    return text.replace(old, new, 1)


def add_header_field(text: str) -> str:
    """Add temporalPositionIdentifier next to the rawDataRunNumber field.

    text: contents of nii_dicom.h.

    v1.0.20260724 packs rawDataRunNumber into a multi-line comma-separated
    int declaration, so it is not its own statement. The new field is a
    separate int placed immediately after that declaration. Filename code
    %y reads it; rawDataRunNumber itself stays the volume-grouping field.
    """
    if "temporalPositionIdentifier" in text:
        return text
    needle: str = "rawDataRunNumber"
    start: int = text.find(needle)
    if start < 0 or text.find(needle, start + len(needle)) >= 0:
        raise SystemExit(
            f"nii_dicom.h: expected 1 rawDataRunNumber, found {text.count(needle)}"
        )
    semi: int = text.find(";", start)
    if semi < 0:
        raise SystemExit("nii_dicom.h: rawDataRunNumber declaration has no semicolon")
    line_start: int = text.rfind("\n", 0, start) + 1
    indent: str = text[line_start:start]
    indent = indent[: len(indent) - len(indent.lstrip(" \t"))]
    # The semicolon may sit mid-line, in front of a trailing comment.
    # Insert the new field after that whole line so the comment stays
    # attached to the original declaration.
    line_end: int = text.find("\n", semi)
    if line_end < 0:
        insert_at = len(text)
        prefix = "\n"
    else:
        insert_at = line_end + 1
        prefix = ""
    field: str = (
        prefix
        + indent
        + "int temporalPositionIdentifier; // (0020,0100) only; filename %y\n"
    )
    return text[:insert_at] + field + text[insert_at:]


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


def patch_superbuild(text: str) -> str:
    """Forward CMAKE_SYSTEM_NAME into the console ExternalProject.

    text: contents of SuperBuild/SuperBuild.cmake.

    v1.0.20260724 copies the C and C++ compilers into the nested console
    build, but not the target system name. mingw-w64 still writes
    dcm2niix.exe, while the generated install() rule looks for an
    unsuffixed dcm2niix and stops the build after the link succeeded.
    """
    old: str = (
        "        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}\n"
        "        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}\n"
    )
    new: str = (
        "        -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}\n"
        "        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}\n"
        "        # Windows cross builds need the target system name so the\n"
        "        # nested console project installs dcm2niix.exe, not dcm2niix.\n"
        "        -DCMAKE_SYSTEM_NAME:STRING=${CMAKE_SYSTEM_NAME}\n"
    )
    return replace_once(text, old, new, "superbuild-system-name")


def main() -> None:
    """Patch a dcm2niix checkout given as the first argument."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_y.py <dcm2niix-source-dir>")
    root: Path = Path(sys.argv[1])
    header: Path = root / "console" / "nii_dicom.h"
    reader: Path = root / "console" / "nii_dicom.cpp"
    batch: Path = root / "console" / "nii_dicom_batch.cpp"
    superbuild: Path = root / "SuperBuild" / "SuperBuild.cmake"
    header.write_text(add_header_field(header.read_text(encoding="utf-8")), encoding="utf-8")
    reader.write_text(patch_reader(reader.read_text(encoding="utf-8")), encoding="utf-8")
    batch.write_text(patch_filename(batch.read_text(encoding="utf-8")), encoding="utf-8")
    superbuild.write_text(
        patch_superbuild(superbuild.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    print("patched", root)


if __name__ == "__main__":
    main()
