"""Shared test fixtures for image analyzer tests."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_image_dir(tmp_path):
    """Create a temporary directory structure mimicking Images/Original and Images/Updated."""
    original = tmp_path / "Images" / "Original"
    updated = tmp_path / "Images" / "Updated"
    original.mkdir(parents=True)
    updated.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_config(tmp_image_dir):
    """Return a config dict with temp paths."""
    return {
        "gemini": {
            "model": "gemini-2.0-flash",
            "max_output_tokens": 4096,
            "temperature": 0.1,
            "requests_per_minute": 15,
            "max_retries": 3,
            "timeout_seconds": 60,
        },
        "scan": {
            "image_folder": str(tmp_image_dir / "Images" / "Original"),
            "output_folder": str(tmp_image_dir / "Images" / "Updated"),
            "recursive": True,
            "supported_formats": [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tiff", ".bmp"],
            "max_image_dimension": 2048,
            "max_file_size_mb": 20,
        },
        "processing": {
            "dry_run": False,
            "rename_images": True,
            "rename_pattern": "{entity_type}_{name}_{seq}_{date}.{ext}",
            "grouping_similarity_threshold": 0.75,
            "confidence_threshold": 0.5,
            "checkpoint_file": str(tmp_image_dir / ".checkpoint.json"),
        },
    }


@pytest.fixture
def create_test_image(tmp_image_dir):
    """Factory fixture to create minimal valid test images."""
    from PIL import Image

    def _create(filename="test.jpg", size=(100, 100), color="red", subdir=None):
        folder = tmp_image_dir / "Images" / "Original"
        if subdir:
            folder = folder / subdir
            folder.mkdir(parents=True, exist_ok=True)
        filepath = folder / filename
        img = Image.new("RGB", size, color)
        if filename.lower().endswith(".png"):
            img.save(filepath, "PNG")
        else:
            img.save(filepath, "JPEG")
        return filepath

    return _create


@pytest.fixture
def sample_analysis_result():
    """Return a sample ImageAnalysisResult dict for testing exporters/groupers."""
    return {
        "image_path": "/tmp/test/IMG_001.jpg",
        "original_filename": "IMG_001.jpg",
        "file_hash_sha256": "abc123def456",
        "perceptual_hash": "f0f0f0f0",
        "classification": {
            "primary_type": "asset",
            "confidence": 0.85,
            "secondary_type": None,
            "reasoning": "Industrial equipment visible",
        },
        "extracted_data": {
            "name": "Hydraulic Press",
            "description": "Industrial hydraulic press with nameplate",
            "serial_number": "SN-12345",
            "model_number": "HP-500",
            "manufacturer_brand": "Acme Corp",
            "suggested_category": "Manufacturing",
            "suggested_vendor": "Acme Corp",
        },
        "exif_metadata": {},
        "flagged_for_review": False,
        "review_reason": None,
    }
