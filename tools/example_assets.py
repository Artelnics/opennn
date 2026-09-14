"""Read indexed example files, including the logical contents of bundled ZIPs."""

import io
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def expand_file(name, data):
    if not name.endswith(".zip"):
        yield name, data
        return
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            original = member.orig_filename
            path = PurePosixPath(original)
            if (path.is_absolute() or ".." in path.parts or "\\" in original
                    or ":" in original or "\0" in original or not path.parts):
                raise ValueError(f"Invalid asset archive path: {original}")
            yield (PurePosixPath(name).parent / path).as_posix(), archive.read(member)


def indexed_files(prefix="examples"):
    """Use Git blobs so validation is independent of OneDrive and CRLF checkouts."""
    entries = subprocess.check_output(
        ["git", "ls-files", "-s", "-z", "--", prefix], cwd=ROOT
    ).decode("utf-8").split("\0")
    items = []
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        _, oid, stage = metadata.split()
        if stage != "0":
            raise ValueError(f"Unresolved index entry: {name}")
        items.append((name, oid))
    blobs = subprocess.check_output(
        ["git", "cat-file", "--batch"], cwd=ROOT,
        input="".join(oid + "\n" for _, oid in items).encode("ascii"),
    )
    offset, seen = 0, set()
    for name, _ in items:
        end = blobs.index(b"\n", offset)
        size = int(blobs[offset:end].split()[-1])
        data = blobs[end + 1:end + 1 + size]
        offset = end + size + 2
        for logical_name, content in expand_file(name, data):
            if logical_name in seen:
                raise ValueError(f"Duplicate example file: {logical_name}")
            seen.add(logical_name)
            yield logical_name, content
