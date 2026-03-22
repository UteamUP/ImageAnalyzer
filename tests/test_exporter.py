"""Tests for CSVExporter — CSV generation, image renaming, summary report."""

import csv
from pathlib import Path

import pytest
from PIL import Image

from image_analyzer.exporter import CSVExporter
from image_analyzer.models import (
    ASSET_CSV_COLUMNS,
    ClassificationResult,
    EntityType,
    ExtractedAssetData,
    ExtractedChemicalData,
    ExtractedToolData,
    ImageAnalysisResult,
    ImageGroup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    image_path: str = "/tmp/img.jpg",
    original_filename: str = "img.jpg",
    entity_type: EntityType = EntityType.ASSET,
    confidence: float = 0.85,
    name: str = "Widget",
    serial_number: str | None = None,
    extracted_data=None,
    flagged_for_review: bool = False,
    review_reason: str | None = None,
) -> ImageAnalysisResult:
    if extracted_data is None:
        extracted_data = ExtractedAssetData(
            name=name,
            serial_number=serial_number,
        )
    return ImageAnalysisResult(
        image_path=image_path,
        original_filename=original_filename,
        file_hash_sha256="deadbeef",
        perceptual_hash="ff00ff00",
        classification=ClassificationResult(
            primary_type=entity_type,
            confidence=confidence,
        ),
        extracted_data=extracted_data,
        flagged_for_review=flagged_for_review,
        review_reason=review_reason,
    )


def _make_group(
    result: ImageAnalysisResult,
    members: list[ImageAnalysisResult] | None = None,
) -> ImageGroup:
    return ImageGroup(
        primary=result,
        members=members or [],
        group_confidence=result.classification.confidence,
    )


def _create_test_image(path: Path) -> None:
    """Create a minimal JPEG at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (10, 10), "blue")
    img.save(str(path), "JPEG")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCSVExporter:
    def test_export_asset_csv_columns(self, tmp_path):
        """CSV should have the correct ASSET_CSV_COLUMNS headers."""
        result = _make_result(name="Pump A")
        group = _make_group(result)

        exporter = CSVExporter(output_folder=str(tmp_path), rename_images=False)
        paths = exporter.export_csvs([group], [])

        assert "asset" in paths
        csv_path = Path(paths["asset"])
        assert csv_path.exists()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == ASSET_CSV_COLUMNS

    def test_export_chemical_csv_joins_lists(self, tmp_path):
        """Chemical hazard/precautionary statements should be semicolon-joined."""
        chem_data = ExtractedChemicalData(
            name="Acetone",
            hazard_statements=["H225", "H319", "H336"],
            precautionary_statements=["P210", "P261"],
        )
        result = _make_result(
            entity_type=EntityType.CHEMICAL,
            name="Acetone",
            extracted_data=chem_data,
        )
        group = _make_group(result)

        exporter = CSVExporter(output_folder=str(tmp_path), rename_images=False)
        paths = exporter.export_csvs([group], [])

        csv_path = Path(paths["chemical"])
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["hazard_statements"] == "H225; H319; H336"
        assert row["precautionary_statements"] == "P210; P261"

    def test_export_skips_empty_types(self, tmp_path):
        """If no tools are found, tools.csv should not be created."""
        result = _make_result(entity_type=EntityType.ASSET, name="Asset Only")
        group = _make_group(result)

        exporter = CSVExporter(output_folder=str(tmp_path), rename_images=False)
        paths = exporter.export_csvs([group], [])

        assert "tool" not in paths
        assert not (tmp_path / "tools.csv").exists()

    def test_rename_images_copies_not_moves(self, tmp_path):
        """After renaming, the original file must still exist."""
        src_dir = tmp_path / "originals"
        src_path = src_dir / "photo.jpg"
        _create_test_image(src_path)

        result = _make_result(
            image_path=str(src_path),
            original_filename="photo.jpg",
            name="Pump",
        )
        group = _make_group(result)

        out_dir = tmp_path / "output"
        exporter = CSVExporter(output_folder=str(out_dir), rename_images=True)
        mapping = exporter.rename_images([group])

        # Original still exists
        assert src_path.exists()
        # New file was created
        assert len(mapping) == 1
        new_path = Path(list(mapping.values())[0])
        assert new_path.exists()

    def test_rename_pattern_format(self, tmp_path):
        """Renamed file should follow the pattern: asset_name_001_YYYYMMDD.jpg"""
        src_dir = tmp_path / "originals"
        src_path = src_dir / "IMG_001.jpg"
        _create_test_image(src_path)

        result = _make_result(
            image_path=str(src_path),
            original_filename="IMG_001.jpg",
            name="Test Pump",
        )
        group = _make_group(result)

        out_dir = tmp_path / "output"
        exporter = CSVExporter(output_folder=str(out_dir), rename_images=True)
        mapping = exporter.rename_images([group])

        new_path = Path(list(mapping.values())[0])
        filename = new_path.name
        # Should be like: asset_test_pump_001_20260322.jpg
        assert filename.startswith("asset_test_pump_")
        assert filename.endswith(".jpg")
        # Should contain 3-digit sequence
        parts = filename.rsplit(".", 1)[0].split("_")
        # Find the sequence part (3 digits)
        seq_parts = [p for p in parts if p.isdigit() and len(p) == 3]
        assert len(seq_parts) == 1
        assert seq_parts[0] == "001"

    def test_no_rename_flag(self, tmp_path):
        """When rename_images=False, no copies should be made."""
        src_dir = tmp_path / "originals"
        src_path = src_dir / "photo.jpg"
        _create_test_image(src_path)

        result = _make_result(
            image_path=str(src_path),
            original_filename="photo.jpg",
            name="Pump",
        )
        group = _make_group(result)

        out_dir = tmp_path / "output"
        exporter = CSVExporter(output_folder=str(out_dir), rename_images=False)
        mapping = exporter.rename_images([group])

        assert mapping == {}

    def test_summary_report_content(self, tmp_path):
        """Summary report should include type counts and duration."""
        r1 = _make_result(name="Asset 1", entity_type=EntityType.ASSET)
        r2 = _make_result(name="Asset 2", entity_type=EntityType.ASSET)
        g1 = _make_group(r1)
        g2 = _make_group(r2)

        unclassified = _make_result(
            entity_type=EntityType.UNCLASSIFIED,
            name="Unknown",
            extracted_data=None,
        )

        exporter = CSVExporter(output_folder=str(tmp_path), rename_images=False)
        report = exporter.generate_summary_report(
            groups=[g1, g2],
            unclassified=[unclassified],
            duration_seconds=123.4,
            duplicates_found=3,
        )

        assert "Image Analysis Summary Report" in report
        assert "asset" in report
        assert "unclassified" in report
        assert "123" in report  # duration in seconds
        assert "3" in report  # duplicates
        assert (tmp_path / "summary_report.md").exists()
