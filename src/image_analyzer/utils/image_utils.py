"""Image utility functions — resize, convert, validate, sanitize."""

import io
import re
import warnings
from pathlib import Path

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)

# Register HEIC/HEIF support globally so Pillow can open these formats
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIF_AVAILABLE = True
except ImportError:
    _HEIF_AVAILABLE = False
    logger.debug("pillow_heif_not_available", msg="HEIC/HEIF files will be skipped")


def resize_image(image_bytes: bytes, max_dimension: int) -> bytes:
    """Resize image if it exceeds max_dimension, preserving aspect ratio.

    Returns JPEG bytes. If the image is already within bounds, it is
    re-encoded as JPEG without up-scaling.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")

    width, height = img.size
    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        logger.debug(
            "image_resized",
            original_size=(width, height),
            new_size=new_size,
            max_dimension=max_dimension,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def convert_heic_to_jpeg(file_path: str) -> bytes:
    """Convert a HEIC/HEIF file to JPEG bytes.

    Requires pillow-heif (registered at module import time).
    """
    if not _HEIF_AVAILABLE:
        raise ImportError(
            "pillow-heif is not installed — HEIC/HEIF images cannot be converted. "
            "Install with: pip install pillow-heif"
        )

    img = Image.open(file_path)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    logger.debug("heic_converted", file_path=file_path, jpeg_size=buf.tell())
    return buf.getvalue()


def is_valid_image(file_path: str) -> bool:
    """Verify that ``file_path`` is a readable, non-corrupted image."""
    ext = Path(file_path).suffix.lower()

    # For HEIC/HEIF, check if pillow-heif is available
    if ext in (".heic", ".heif") and not _HEIF_AVAILABLE:
        logger.warning("heic_no_library", file_path=file_path)
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        # Some formats (including HEIC) may fail verify() but still be loadable.
        # Try actually loading the image as a fallback.
        try:
            with Image.open(file_path) as img:
                img.load()
            return True
        except Exception:
            logger.warning("invalid_image", file_path=file_path)
            return False


def sanitize_filename(name: str) -> str:
    """Sanitize a filename to contain only alphanumeric, underscore, and dash.

    - Lowercased
    - Spaces replaced with underscores
    - All other special / unicode characters stripped
    """
    name = name.lower().strip()
    # Separate stem from extension so we can clean each independently
    dot_idx = name.rfind(".")
    if dot_idx > 0:
        stem = name[:dot_idx]
        ext = name[dot_idx:]  # includes the dot
    else:
        stem = name
        ext = ""

    stem = stem.replace(" ", "_")
    # Keep only alphanumeric, underscore, and dash
    stem = re.sub(r"[^a-z0-9_\-]", "", stem)
    # Collapse multiple underscores / dashes
    stem = re.sub(r"[_]{2,}", "_", stem)
    stem = re.sub(r"[-]{2,}", "-", stem)
    # Strip leading/trailing underscores and dashes from the stem
    stem = stem.strip("_-")

    # Clean the extension (keep only alphanumeric + dot)
    ext = re.sub(r"[^a-z0-9.]", "", ext)

    return stem + ext


def load_image_bytes(file_path: str, max_dimension: int = 2048) -> bytes:
    """Load any supported image (including HEIC), resize if needed, return JPEG bytes.

    This is the main entry point used by downstream consumers that need
    raw JPEG bytes ready for API submission.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension in (".heic", ".heif"):
        raw_bytes = convert_heic_to_jpeg(file_path)
    else:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

    return resize_image(raw_bytes, max_dimension)
