"""Bundled archives preserve file identities and the example data staging layout."""

import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from example_assets import expand_file


def packed(members):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            member = zipfile.ZipInfo()
            member.filename = name  # Keep deliberately invalid names on Windows too.
            archive.writestr(member, content)
    return data.getvalue()


class ExampleAssetsTest(unittest.TestCase):
    def test_archive_preserves_logical_paths_and_binary_bytes(self):
        payload = b"\x00\xff\r\n"
        self.assertEqual(dict(expand_file("examples/mnist/data/images.zip",
                                         packed({"zero/0.bmp": payload}))),
                         {"examples/mnist/data/zero/0.bmp": payload})
        self.assertEqual(dict(expand_file("examples/legacy_8/reference.zip",
                                         packed({"forecasting/data/model.bin": payload}))),
                         {"examples/legacy_8/forecasting/data/model.bin": payload})

    def test_invalid_member_names_cannot_change_asset_identity(self):
        for name in ("../outside.bmp", "/outside.bmp", "C:/outside.bmp", "zero\\image.bmp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                list(expand_file("examples/mnist/data/images.zip", packed({name: b"x"})))

    def test_cmake_stages_loose_and_packed_data_in_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="opennn data ") as folder:
            source, output = Path(folder) / "source", Path(folder) / "build data"
            source.mkdir()
            (source / "SOURCE.md").write_text("Dataset attribution\n")
            for content in (b"original pixels", b"updated pixels"):
                (source / "images.zip").write_bytes(packed({"zero/0.bmp": content}))
                subprocess.run([
                    "cmake", f"-DOPENNN_DATA_SOURCE={source}",
                    f"-DOPENNN_DATA_DESTINATION={output}",
                    "-P", str(ROOT / "examples/prepare_data.cmake"),
                ], check=True, capture_output=True)
                self.assertEqual((output / "zero/0.bmp").read_bytes(), content)
                # Fixed archive timestamps must not leave an old image cache valid.
                self.assertGreaterEqual((output / "zero/0.bmp").stat().st_mtime,
                                        (source / "images.zip").stat().st_mtime - 2)
                self.assertEqual((output / "SOURCE.md").read_text(), "Dataset attribution\n")
                self.assertFalse((output / "images.zip").exists())


if __name__ == "__main__":
    unittest.main()
