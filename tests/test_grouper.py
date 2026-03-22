"""Tests for ImageGrouper — image clustering by physical item."""

import pytest

from image_analyzer.grouper import ImageGrouper
from image_analyzer.models import (
    ClassificationResult,
    EntityType,
    ExtractedAssetData,
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
    model_number: str | None = None,
    manufacturer_brand: str | None = None,
    perceptual_hash: str = "ff00ff00",
    paired_images: list[str] | None = None,
    extracted_data=None,
) -> ImageAnalysisResult:
    """Convenience factory for building test results."""
    if extracted_data is None:
        if entity_type == EntityType.TOOL:
            extracted_data = ExtractedToolData(
                name=name,
                serial_number=serial_number,
                model_number=model_number,
                manufacturer_brand=manufacturer_brand,
            )
        elif entity_type in (EntityType.ASSET, EntityType.UNCLASSIFIED):
            if entity_type == EntityType.UNCLASSIFIED:
                extracted_data = None
            else:
                extracted_data = ExtractedAssetData(
                    name=name,
                    serial_number=serial_number,
                    model_number=model_number,
                    manufacturer_brand=manufacturer_brand,
                )
        else:
            extracted_data = ExtractedAssetData(
                name=name,
                serial_number=serial_number,
                model_number=model_number,
                manufacturer_brand=manufacturer_brand,
            )

    return ImageAnalysisResult(
        image_path=image_path,
        original_filename=original_filename,
        file_hash_sha256="deadbeef",
        perceptual_hash=perceptual_hash,
        classification=ClassificationResult(
            primary_type=entity_type,
            confidence=confidence,
        ),
        extracted_data=extracted_data,
        paired_images=paired_images or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestImageGrouper:
    def test_group_by_serial_number(self):
        """Two results with the same serial_number should be grouped together."""
        r1 = _make_result(
            image_path="/tmp/a.jpg",
            serial_number="SN-001",
            name="Pump A",
        )
        r2 = _make_result(
            image_path="/tmp/b.jpg",
            serial_number="SN-001",
            name="Pump B",
        )
        grouper = ImageGrouper()
        groups = grouper.group_images([r1, r2])

        assert len(groups) == 1
        assert len(groups[0].all_image_paths) == 2

    def test_no_cross_type_grouping(self):
        """An asset and a tool with the same name must NOT be grouped."""
        r_asset = _make_result(
            image_path="/tmp/a.jpg",
            entity_type=EntityType.ASSET,
            name="Multi-tool X",
            serial_number="SN-SAME",
        )
        r_tool = _make_result(
            image_path="/tmp/b.jpg",
            entity_type=EntityType.TOOL,
            name="Multi-tool X",
            serial_number="SN-SAME",
        )
        grouper = ImageGrouper()
        groups = grouper.group_images([r_asset, r_tool])

        assert len(groups) == 2

    def test_fuzzy_name_grouping(self):
        """Very similar names should cause grouping via fuzzy match."""
        r1 = _make_result(
            image_path="/tmp/a.jpg",
            name="Hydraulic Press XR-500",
            model_number="XR-500",
            manufacturer_brand="Acme",
            perceptual_hash="ff00ff00",
        )
        r2 = _make_result(
            image_path="/tmp/b.jpg",
            name="Hydraulic Press XR500",
            model_number="XR-500",
            manufacturer_brand="Acme",
            perceptual_hash="ff00ff00",
        )
        # With model_number match (0.20) + name fuzzy (~0.19) + phash (0.15) + brand (0.05) = ~0.59
        # Need to lower threshold for this to work without serial
        grouper = ImageGrouper(similarity_threshold=0.55)
        groups = grouper.group_images([r1, r2])

        assert len(groups) == 1
        assert len(groups[0].all_image_paths) == 2

    def test_unclassified_never_grouped(self):
        """Unclassified results must each remain in their own group."""
        r1 = _make_result(
            image_path="/tmp/a.jpg",
            entity_type=EntityType.UNCLASSIFIED,
            name="Unknown 1",
        )
        r2 = _make_result(
            image_path="/tmp/b.jpg",
            entity_type=EntityType.UNCLASSIFIED,
            name="Unknown 2",
        )
        grouper = ImageGrouper()
        groups = grouper.group_images([r1, r2])

        assert len(groups) == 2
        for g in groups:
            assert len(g.members) == 0

    def test_iphone_edit_pairs_preserved(self):
        """A result with paired_images referencing another should absorb it."""
        r1 = _make_result(
            image_path="/tmp/IMG_001.jpg",
            name="Pump",
            paired_images=["/tmp/IMG_E001.jpg"],
        )
        r2 = _make_result(
            image_path="/tmp/IMG_E001.jpg",
            name="Pump",
        )
        grouper = ImageGrouper()
        groups = grouper.group_images([r1, r2])

        # The pair should be merged — only 1 group
        assert len(groups) == 1
        # The primary should be the one with paired_images or the higher confidence
        assert "/tmp/IMG_001.jpg" in groups[0].all_image_paths

    def test_representative_selection(self):
        """The representative should be the result with the highest confidence."""
        r_low = _make_result(
            image_path="/tmp/low.jpg",
            serial_number="SN-X",
            confidence=0.60,
            name="Item",
        )
        r_mid = _make_result(
            image_path="/tmp/mid.jpg",
            serial_number="SN-X",
            confidence=0.75,
            name="Item",
        )
        r_high = _make_result(
            image_path="/tmp/high.jpg",
            serial_number="SN-X",
            confidence=0.95,
            name="Item",
        )
        grouper = ImageGrouper()
        groups = grouper.group_images([r_low, r_mid, r_high])

        assert len(groups) == 1
        assert groups[0].primary.classification.confidence == 0.95
        assert groups[0].primary.image_path == "/tmp/high.jpg"

    def test_below_threshold_not_grouped(self):
        """Two results with low similarity should stay in separate groups."""
        r1 = _make_result(
            image_path="/tmp/a.jpg",
            name="Forklift Model A",
            perceptual_hash="0000ffff",
        )
        r2 = _make_result(
            image_path="/tmp/b.jpg",
            name="Welding Station Z",
            perceptual_hash="ffff0000",
        )
        grouper = ImageGrouper(similarity_threshold=0.75)
        groups = grouper.group_images([r1, r2])

        assert len(groups) == 2
