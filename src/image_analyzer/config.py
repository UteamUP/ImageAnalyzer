"""Configuration loader — YAML + .env with CLI overrides."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "gemini-3.1-flash"
    max_output_tokens: int = 4096
    temperature: float = 0.1
    requests_per_minute: int = 15
    max_retries: int = 3
    timeout_seconds: int = 60


@dataclass
class ScanConfig:
    image_folder: str = "./Images/Original"
    output_folder: str = "./Output"
    renamed_images_folder: str = "./Images/Updated"
    recursive: bool = True
    supported_formats: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tiff", ".bmp"]
    )
    max_image_dimension: int = 2048
    max_file_size_mb: int = 20


@dataclass
class ProcessingConfig:
    dry_run: bool = False
    rename_images: bool = True
    rename_pattern: str = "{entity_type}_{name}_{seq}_{date}.{ext}"
    grouping_similarity_threshold: float = 0.75
    confidence_threshold: float = 0.5
    checkpoint_file: str = ".checkpoint.json"
    max_cost: Optional[float] = None


@dataclass
class AppConfig:
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    def validate(self) -> list[str]:
        """Validate config, return list of errors."""
        errors = []
        if not self.gemini.api_key and not self.processing.dry_run:
            errors.append("GEMINI_API_KEY is required (set in .env or environment)")

        image_folder = Path(self.scan.image_folder).resolve()
        if not image_folder.is_dir():
            errors.append(f"Image folder does not exist: {image_folder}")

        output_folder = Path(self.scan.output_folder).resolve()
        output_folder.mkdir(parents=True, exist_ok=True)

        renamed_folder = Path(self.scan.renamed_images_folder).resolve()
        renamed_folder.mkdir(parents=True, exist_ok=True)

        if self.gemini.temperature < 0 or self.gemini.temperature > 2:
            errors.append(f"Temperature must be 0-2, got {self.gemini.temperature}")

        if self.gemini.requests_per_minute < 1 or self.gemini.requests_per_minute > 1000:
            errors.append(f"requests_per_minute must be 1-1000, got {self.gemini.requests_per_minute}")

        return errors


def load_config(
    config_path: str = "config.yaml",
    folder_override: Optional[str] = None,
    output_override: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = False,
    no_rename: bool = False,
    max_cost: Optional[float] = None,
    verbose: bool = False,
) -> AppConfig:
    """Load config from YAML + .env, apply CLI overrides."""
    load_dotenv()

    config_data = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f) or {}

    gemini_data = config_data.get("gemini", {})
    scan_data = config_data.get("scan", {})
    processing_data = config_data.get("processing", {})

    # Build GeminiConfig — env vars override YAML
    gemini = GeminiConfig(
        api_key=os.environ.get("GEMINI_API_KEY", gemini_data.get("api_key", "")),
        model=os.environ.get("GEMINI_MODEL", gemini_data.get("model", "gemini-3.1-flash")),
        max_output_tokens=int(
            os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", gemini_data.get("max_output_tokens", 4096))
        ),
        temperature=float(
            os.environ.get("GEMINI_TEMPERATURE", gemini_data.get("temperature", 0.1))
        ),
        requests_per_minute=int(
            os.environ.get("GEMINI_REQUESTS_PER_MINUTE", gemini_data.get("requests_per_minute", 15))
        ),
        max_retries=int(gemini_data.get("max_retries", 3)),
        timeout_seconds=int(gemini_data.get("timeout_seconds", 60)),
    )

    # Build ScanConfig — CLI overrides > env vars > YAML
    scan = ScanConfig(
        image_folder=str(Path(
            folder_override
            or os.environ.get("IMAGE_FOLDER", scan_data.get("image_folder", "./Images/Original"))
        ).resolve()),
        output_folder=str(Path(
            output_override
            or os.environ.get("OUTPUT_FOLDER", scan_data.get("output_folder", "./Output"))
        ).resolve()),
        renamed_images_folder=str(Path(
            os.environ.get("RENAMED_IMAGES_FOLDER", scan_data.get("renamed_images_folder", "./Images/Updated"))
        ).resolve()),
        recursive=scan_data.get("recursive", True),
        supported_formats=scan_data.get(
            "supported_formats",
            [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tiff", ".bmp"],
        ),
        max_image_dimension=int(scan_data.get("max_image_dimension", 2048)),
        max_file_size_mb=int(scan_data.get("max_file_size_mb", 20)),
    )

    # Build ProcessingConfig — CLI overrides
    processing = ProcessingConfig(
        dry_run=dry_run or processing_data.get("dry_run", False),
        rename_images=not no_rename and processing_data.get("rename_images", True),
        rename_pattern=processing_data.get("rename_pattern", "{entity_type}_{name}_{seq}_{date}.{ext}"),
        grouping_similarity_threshold=float(
            processing_data.get("grouping_similarity_threshold", 0.75)
        ),
        confidence_threshold=float(processing_data.get("confidence_threshold", 0.5)),
        checkpoint_file=str(Path(processing_data.get("checkpoint_file", ".checkpoint.json")).resolve()),
        max_cost=max_cost,
    )

    return AppConfig(gemini=gemini, scan=scan, processing=processing)
