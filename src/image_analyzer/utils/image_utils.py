"""Image utility functions — resize, convert, validate, sanitize."""

import io
import re
import warnings
from pathlib import Path

import structlog
from PIL import Image

logger = structlog.get_logger(__name__)


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

    Uses pillow-heif if available. If the library is missing, logs a
    warning and raises ``ImportError`` so callers can decide how to
    proceed.
    """
    try:
        import pillow_heif  # noqa: F811

        pillow_heif.register_heif_opener()
    except ImportError:
        warnings.warn(
            "pillow-heif is not installed — HEIC/HEIF images cannot be converted. "
            "Install with: pip install pillow-heif",
            stacklevel=2,
        )
        raise

    img = Image.open(file_path)
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    logger.debug("heic_converted", file_path=file_path, jpeg_size=buf.tell())
    return buf.getvalue()


def is_valid_image(file_path: str) -> bool:
    """Verify that ``file_path`` is a readable, non-corrupted image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
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
    name = name.lower()
    name = name.replace(" ", "_")
    # Keep only alphanumeric, underscore, dash, and dot (for extensions)
    name = re.sub(r"[^a-z0-9_\-.]", "", name)
    # Collapse multiple underscores / dashes
    name = re.sub(r"[_]{2,}", "_", name)
    name = re.sub(r"[-]{2,}", "-", name)
    # Strip leading/trailing underscores and dashes
    name = name.strip("_-")
    return name


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
