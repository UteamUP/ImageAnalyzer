"""CSV export and image renaming for grouped analysis results."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import structlog

from image_analyzer.models import (
    ASSET_CSV_COLUMNS,
    CHEMICAL_CSV_COLUMNS,
    CSV_COLUMNS_BY_TYPE,
    PART_CSV_COLUMNS,
    TOOL_CSV_COLUMNS,
    UNCLASSIFIED_CSV_COLUMNS,
    EntityType,
    ExtractedChemicalData,
    ImageAnalysisResult,
    ImageGroup,
)
from image_analyzer.utils.image_utils import sanitize_filename

logger = structlog.get_logger(__name__)


class CSVExporter:
    """Export grouped image analysis results to per-entity-type CSV files."""

    def __init__(
        self,
        output_folder: str,
        renamed_images_folder: str = "",
        rename_images: bool = True,
        rename_pattern: str = "{entity_type}_{name}_{seq}_{date}.{ext}",
    ) -> None:
        self.output_folder = Path(output_folder)
        self.renamed_images_folder = Path(renamed_images_folder) if renamed_images_folder else self.output_folder
        self._rename_images_flag = rename_images
        self.rename_pattern = rename_pattern
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.renamed_images_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_csvs(
        self,
        groups: list[ImageGroup],
        unclassified: list[ImageAnalysisResult],
    ) -> dict[str, str]:
        """Write one CSV file per entity type. Returns ``{entity_type: csv_path}``."""
        # Bucket groups by entity type
        by_type: dict[EntityType, list[ImageGroup]] = {}
        for g in groups:
            etype = g.primary.classification.primary_type
            by_type.setdefault(etype, []).append(g)

        # Add unclassified as individual pseudo-groups
        if unclassified:
            by_type.setdefault(EntityType.UNCLASSIFIED, [])
            for r in unclassified:
                by_type[EntityType.UNCLASSIFIED].append(
                    ImageGroup(primary=r, members=[], group_confidence=r.classification.confidence)
                )

        result: dict[str, str] = {}
        for etype, etype_groups in by_type.items():
            if not etype_groups:
                continue
            columns = CSV_COLUMNS_BY_TYPE.get(etype)
            if columns is None:
                continue
            csv_path = self.output_folder / f"{etype.value}s.csv"
            self._write_csv(csv_path, columns, etype_groups, etype)
            result[etype.value] = str(csv_path)
            logger.info(
                "csv_exported",
                entity_type=etype.value,
                path=str(csv_path),
                row_count=len(etype_groups),
            )

        return result

    def rename_images(self, groups: list[ImageGroup]) -> dict[str, str]:
        """Copy images to ``output_folder`` with descriptive filenames.

        Returns a mapping of ``{original_path: new_path}``.
        Images are **copied** (not moved) so originals remain intact.
        """
        if not self._rename_images_flag:
            return {}

        mapping: dict[str, str] = {}
        seq_counters: dict[str, int] = {}
        today = datetime.now().strftime("%Y%m%d")

        for group in groups:
            etype = group.primary.classification.primary_type.value
            name = ""
            if group.primary.extracted_data and hasattr(group.primary.extracted_data, "name"):
                name = sanitize_filename(group.primary.extracted_data.name)
            if not name:
                name = "unnamed"

            for image_path in group.all_image_paths:
                src = Path(image_path)
                if not src.exists():
                    logger.warning("source_image_missing", path=image_path)
                    continue

                ext = src.suffix.lstrip(".")
                key = f"{etype}_{name}"
                seq = seq_counters.get(key, 1)

                # Build new filename and handle collisions
                new_name = f"{etype}_{name}_{seq:03d}_{today}.{ext}"
                dest = self.renamed_images_folder / new_name
                while dest.exists():
                    seq += 1
                    new_name = f"{etype}_{name}_{seq:03d}_{today}.{ext}"
                    dest = self.renamed_images_folder / new_name

                seq_counters[key] = seq + 1
                shutil.copy2(str(src), str(dest))
                mapping[str(src)] = str(dest)
                logger.debug("image_renamed", old=str(src), new=str(dest))

        return mapping

    def generate_summary_report(
        self,
        groups: list[ImageGroup],
        unclassified: list[ImageAnalysisResult],
        duration_seconds: float,
        duplicates_found: int,
    ) -> str:
        """Generate a Markdown summary report and write it to ``output_folder``.

        Returns the report text.
        """
        total_images = sum(len(g.all_image_paths) for g in groups) + len(unclassified)
        flagged = sum(
            1 for g in groups if g.primary.flagged_for_review
        ) + sum(1 for r in unclassified if r.flagged_for_review)

        # Per-type counts
        type_counts: dict[str, int] = {}
        for g in groups:
            t = g.primary.classification.primary_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        if unclassified:
            type_counts["unclassified"] = len(unclassified)

        minutes = duration_seconds / 60.0

        lines = [
            "# Image Analysis Summary Report",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total images processed | {total_images} |",
            f"| Groups formed | {len(groups)} |",
            f"| Flagged for review | {flagged} |",
            f"| Duplicates found | {duplicates_found} |",
            f"| Processing duration | {minutes:.1f} min ({duration_seconds:.0f}s) |",
            "",
            "## Results by Type",
            "",
            "| Entity Type | Count |",
            "|-------------|-------|",
        ]
        for t, count in sorted(type_counts.items()):
            lines.append(f"| {t} | {count} |")

        lines.append("")
        lines.append("---")
        lines.append("*Generated by UteamUP Image Analyzer*")
        lines.append("")

        report = "\n".join(lines)

        report_path = self.output_folder / "summary_report.md"
        report_path.write_text(report, encoding="utf-8")
        logger.info("summary_report_written", path=str(report_path))

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_csv(
        self,
        path: Path,
        columns: list[str],
        groups: list[ImageGroup],
        etype: EntityType,
    ) -> None:
        """Write a single CSV file for a given entity type."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()

            for group in groups:
                row = self._build_row(group, etype)
                writer.writerow(row)

    def _build_row(self, group: ImageGroup, etype: EntityType) -> dict:
        """Build a CSV row dict from a group's primary result."""
        primary = group.primary
        row: dict = {}

        if etype == EntityType.UNCLASSIFIED:
            row["original_filename"] = primary.original_filename
            row["image_path"] = primary.image_path
            row["confidence_score"] = primary.classification.confidence
            row["flagged_for_review"] = primary.flagged_for_review
            row["review_reason"] = primary.review_reason or ""
            row["classification_reasoning"] = primary.classification.reasoning
            row["related_to"] = primary.related_to or ""
            return row

        # Extract data fields from the typed extracted_data model
        if primary.extracted_data:
            data_dict = primary.extracted_data.model_dump()
            # Handle list fields for chemicals — join with semicolons
            if isinstance(primary.extracted_data, ExtractedChemicalData):
                if "hazard_statements" in data_dict:
                    data_dict["hazard_statements"] = "; ".join(
                        data_dict.get("hazard_statements") or []
                    )
                if "precautionary_statements" in data_dict:
                    data_dict["precautionary_statements"] = "; ".join(
                        data_dict.get("precautionary_statements") or []
                    )
            row.update(data_dict)

        # Relationship to parent entity
        row["related_to"] = primary.related_to or ""

        # Append common trailing columns
        row["image_paths"] = "; ".join(group.all_image_paths)
        row["original_filenames"] = "; ".join(group.all_original_filenames)
        row["confidence_score"] = primary.classification.confidence
        row["flagged_for_review"] = primary.flagged_for_review
        row["review_reason"] = primary.review_reason or ""

        return row
