"""Image scanner — walk folders, extract metadata, detect duplicates and iPhone edit pairs."""

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import imagehash
import structlog
from PIL import Image

from image_analyzer.config import ScanConfig
from image_analyzer.utils.image_utils import is_valid_image

logger = structlog.get_logger(__name__)


@dataclass
class ImageInfo:
    """Metadata for a single discovered image file."""

    path: str
    filename: str
    extension: str
    file_size_bytes: int
    sha256_hash: str = ""
    perceptual_hash: str = ""
    exif_metadata: dict = field(default_factory=dict)
    is_iphone_edit: bool = False
    paired_with: Optional[str] = None


# Regex to detect iPhone edit variants: IMG_EXXXX or IMG_E XXXX
_IPHONE_EDIT_RE = re.compile(r"^IMG_E(\d{4})", re.IGNORECASE)
_IPHONE_ORIGINAL_RE = re.compile(r"^IMG_(\d{4})", re.IGNORECASE)


class ImageScanner:
    """Scan a folder for images, extract hashes/EXIF, and detect duplicates."""

    def __init__(self, config: ScanConfig) -> None:
        self._config = config
        self._supported = set(fmt.lower() for fmt in config.supported_formats)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_folder(self) -> list[ImageInfo]:
        """Walk the configured image folder and return metadata for every valid image.

        Respects the ``recursive`` flag in the config. Skips files that
        are not valid images or whose extension is not in
        ``supported_formats``.
        """
        root = Path(self._config.image_folder)
        if not root.is_dir():
            logger.error("image_folder_not_found", path=str(root))
            return []

        images: list[ImageInfo] = []

        if self._config.recursive:
            walker = root.rglob("*")
        else:
            walker = root.glob("*")

        for entry in sorted(walker):
            if not entry.is_file():
                continue

            ext = entry.suffix.lower()
            if ext not in self._supported:
                logger.debug("skipped_unsupported_format", file=str(entry), ext=ext)
                continue

            if not is_valid_image(str(entry)):
                logger.warning("skipped_invalid_image", file=str(entry))
                continue

            sha256, phash = self.compute_hashes(str(entry))

            info = ImageInfo(
                path=str(entry),
                filename=entry.name,
                extension=ext,
                file_size_bytes=entry.stat().st_size,
                sha256_hash=sha256,
                perceptual_hash=phash,
                exif_metadata=self.extract_exif(str(entry)),
            )
            images.append(info)
            logger.debug("image_found", filename=entry.name, size=info.file_size_bytes)

        logger.info("scan_complete", total_images=len(images), folder=str(root))
        return images

    def extract_exif(self, file_path: str) -> dict:
        """Extract useful EXIF fields: date_taken, gps, camera_model.

        Returns an empty dict when EXIF is unavailable or the file
        cannot be read.
        """
        result: dict = {}
        try:
            with Image.open(file_path) as img:
                exif_data = img.getexif()
                if not exif_data:
                    return result

                # Tag IDs: 36867=DateTimeOriginal, 271=Make, 272=Model
                date_taken = exif_data.get(36867) or exif_data.get(306)  # 306=DateTime
                if date_taken:
                    result["date_taken"] = str(date_taken)

                make = exif_data.get(271)
                model = exif_data.get(272)
                if make or model:
                    result["camera_model"] = f"{make or ''} {model or ''}".strip()

                # GPS info lives in IFD 0x8825
                gps_ifd = exif_data.get_ifd(0x8825)
                if gps_ifd:
                    result["gps"] = {
                        str(k): str(v) for k, v in gps_ifd.items()
                    }

        except Exception:
            logger.debug("exif_extraction_failed", file_path=file_path)

        return result

    def compute_hashes(self, file_path: str) -> tuple[str, str]:
        """Return ``(sha256_hash, perceptual_hash)`` for the file at *file_path*.

        The SHA-256 is computed over the raw file bytes. The perceptual
        hash uses the ``imagehash`` average-hash algorithm so that
        visually similar images produce close hashes.
        """
        # SHA-256
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        sha256 = sha.hexdigest()

        # Perceptual hash
        phash = ""
        try:
            img = Image.open(file_path)
            phash = str(imagehash.average_hash(img))
        except Exception:
            logger.debug("perceptual_hash_failed", file_path=file_path)

        return sha256, phash

    def detect_duplicates(
        self, images: list[ImageInfo]
    ) -> tuple[list[ImageInfo], list[tuple[str, str]]]:
        """Detect duplicate images by SHA-256 hash.

        Returns a tuple of ``(unique_images, duplicate_pairs_log)``
        where each duplicate pair is ``(kept_path, removed_path)``.
        """
        seen: dict[str, ImageInfo] = {}
        unique: list[ImageInfo] = []
        duplicates: list[tuple[str, str]] = []

        for img in images:
            if img.sha256_hash in seen:
                kept = seen[img.sha256_hash]
                duplicates.append((kept.path, img.path))
                logger.info(
                    "duplicate_detected",
                    kept=kept.filename,
                    duplicate=img.filename,
                    hash=img.sha256_hash[:12],
                )
            else:
                seen[img.sha256_hash] = img
                unique.append(img)

        logger.info(
            "duplicate_detection_complete",
            total=len(images),
            unique=len(unique),
            duplicates=len(duplicates),
        )
        return unique, duplicates

    def detect_iphone_edit_pairs(
        self, images: list[ImageInfo]
    ) -> dict[str, list[str]]:
        """Detect iPhone edited image variants.

        iPhones save the original as ``IMG_XXXX.jpg`` and the edited
        version as ``IMG_EXXXX.jpg``. This method finds those pairs and
        returns a dict mapping the primary (original) filename to a list
        of edit-variant file paths.

        Side-effect: sets ``is_iphone_edit`` and ``paired_with`` on the
        matching ``ImageInfo`` objects.
        """
        # Build a lookup: number -> list of ImageInfo for originals
        originals: dict[str, ImageInfo] = {}
        edits: dict[str, list[ImageInfo]] = {}

        for img in images:
            stem = Path(img.filename).stem

            edit_match = _IPHONE_EDIT_RE.match(stem)
            if edit_match:
                number = edit_match.group(1)
                edits.setdefault(number, []).append(img)
                continue

            orig_match = _IPHONE_ORIGINAL_RE.match(stem)
            if orig_match:
                number = orig_match.group(1)
                originals[number] = img

        pairs: dict[str, list[str]] = {}

        for number, edit_list in edits.items():
            if number in originals:
                primary = originals[number]
                variant_paths = []
                for edit_img in edit_list:
                    edit_img.is_iphone_edit = True
                    edit_img.paired_with = primary.filename
                    variant_paths.append(edit_img.path)
                pairs[primary.filename] = variant_paths
                logger.info(
                    "iphone_edit_pair",
                    primary=primary.filename,
                    variants=[Path(p).name for p in variant_paths],
                )

        logger.info("iphone_edit_detection_complete", pairs_found=len(pairs))
        return pairs
