"""Tests for image_analyzer.scanner and image_analyzer.utils.image_utils."""

import shutil
from pathlib import Path

import pytest

from image_analyzer.config import ScanConfig
from image_analyzer.scanner import ImageScanner, ImageInfo
from image_analyzer.utils.image_utils import sanitize_filename


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scan_config(sample_config: dict, **overrides) -> ScanConfig:
    """Build a ScanConfig from the sample_config fixture dict with optional overrides."""
    data = {**sample_config["scan"], **overrides}
    return ScanConfig(**data)


# ---------------------------------------------------------------------------
# scan_folder tests
# ---------------------------------------------------------------------------


class TestScanFolder:
    def test_scan_folder_finds_images(self, create_test_image, sample_config):
        """Scanner should discover .jpg and .png files in the image folder."""
        create_test_image("photo1.jpg")
        create_test_image("photo2.png")

        config = _make_scan_config(sample_config)
        scanner = ImageScanner(config)
        results = scanner.scan_folder()

        filenames = {img.filename for img in results}
        assert "photo1.jpg" in filenames
        assert "photo2.png" in filenames
        assert len(results) == 2

    def test_scan_folder_skips_non_images(self, create_test_image, sample_config, tmp_image_dir):
        """A .txt file in the folder must not appear in scan results."""
        create_test_image("real_image.jpg")
        txt_path = tmp_image_dir / "Images" / "Original" / "notes.txt"
        txt_path.write_text("this is not an image")

        config = _make_scan_config(sample_config)
        scanner = ImageScanner(config)
        results = scanner.scan_folder()

        filenames = {img.filename for img in results}
        assert "real_image.jpg" in filenames
        assert "notes.txt" not in filenames

    def test_scan_folder_recursive(self, create_test_image, sample_config):
        """With recursive=True, images in sub-folders should be found."""
        create_test_image("top.jpg")
        create_test_image("deep.jpg", subdir="sub1/sub2")

        config = _make_scan_config(sample_config, recursive=True)
        scanner = ImageScanner(config)
        results = scanner.scan_folder()

        filenames = {img.filename for img in results}
        assert "top.jpg" in filenames
        assert "deep.jpg" in filenames

    def test_scan_folder_non_recursive(self, create_test_image, sample_config):
        """With recursive=False, images in sub-folders should be skipped."""
        create_test_image("top.jpg")
        create_test_image("nested.jpg", subdir="child")

        config = _make_scan_config(sample_config, recursive=False)
        scanner = ImageScanner(config)
        results = scanner.scan_folder()

        filenames = {img.filename for img in results}
        assert "top.jpg" in filenames
        assert "nested.jpg" not in filenames


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


class TestDetectDuplicates:
    def test_detect_duplicates(self, create_test_image, sample_config):
        """Two identical images should result in one unique + one duplicate pair."""
        path1 = create_test_image("original.jpg", size=(50, 50), color="blue")
        # Copy the file so bytes (and therefore SHA-256) are identical
        dup_path = path1.parent / "copy.jpg"
        shutil.copy2(path1, dup_path)

        config = _make_scan_config(sample_config)
        scanner = ImageScanner(config)
        images = scanner.scan_folder()

        assert len(images) == 2  # both discovered

        unique, dup_pairs = scanner.detect_duplicates(images)

        assert len(unique) == 1
        assert len(dup_pairs) == 1
        kept, removed = dup_pairs[0]
        assert kept != removed


# ---------------------------------------------------------------------------
# iPhone edit pair detection
# ---------------------------------------------------------------------------


class TestDetectIphoneEditPairs:
    def test_detect_iphone_edit_pairs(self, create_test_image, sample_config):
        """IMG_E3021.jpg should be detected as an edit of IMG_3021.jpg."""
        create_test_image("IMG_3021.jpg", color="green")
        create_test_image("IMG_E3021.jpg", color="green")

        config = _make_scan_config(sample_config)
        scanner = ImageScanner(config)
        images = scanner.scan_folder()

        pairs = scanner.detect_iphone_edit_pairs(images)

        assert "IMG_3021.jpg" in pairs
        variant_names = [Path(p).name for p in pairs["IMG_3021.jpg"]]
        assert "IMG_E3021.jpg" in variant_names

        # Verify the edit ImageInfo was mutated
        edit_img = next(i for i in images if i.filename == "IMG_E3021.jpg")
        assert edit_img.is_iphone_edit is True
        assert edit_img.paired_with == "IMG_3021.jpg"


# ---------------------------------------------------------------------------
# Corrupted image handling
# ---------------------------------------------------------------------------


class TestCorruptedImages:
    def test_corrupted_image_skipped(self, sample_config, tmp_image_dir):
        """A file with .jpg extension but garbage content should be skipped."""
        bad_file = tmp_image_dir / "Images" / "Original" / "corrupt.jpg"
        bad_file.write_bytes(b"this is definitely not a jpeg file at all")

        config = _make_scan_config(sample_config)
        scanner = ImageScanner(config)
        results = scanner.scan_folder()

        filenames = {img.filename for img in results}
        assert "corrupt.jpg" not in filenames


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "input_name, expected",
        [
            ("Hello World.jpg", "hello_world.jpg"),
            ("My Photo (1).png", "my_photo_1.png"),
            ("café_résumé.jpg", "caf_rsum.jpg"),
            ("  leading-trailing  .jpg", "leading-trailing.jpg"),
            ("UPPER_CASE_FILE.PNG", "upper_case_file.png"),
            ("file---name___test.jpg", "file-name_test.jpg"),
            ("special!@#$%^&*chars.jpg", "specialchars.jpg"),
            ("already_clean.jpg", "already_clean.jpg"),
            ("spaces   between.jpg", "spaces_between.jpg"),
        ],
    )
    def test_sanitize_filename(self, input_name, expected):
        assert sanitize_filename(input_name) == expected
