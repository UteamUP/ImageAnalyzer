"""Pipeline orchestrator — wires scan, analyze, group, export phases."""

import time
from pathlib import Path

import structlog
from tqdm import tqdm

from image_analyzer.config import AppConfig
from image_analyzer.models import EntityType, ImageAnalysisResult, ImageGroup

logger = structlog.get_logger()


class Pipeline:
    """Orchestrates the 4-phase image analysis pipeline."""

    def __init__(self, config: AppConfig):
        self.config = config

    def run(self) -> None:
        """Execute the full pipeline: scan → analyze → group → export."""
        start_time = time.time()

        # Phase 1: Scan
        logger.info("Phase 1: Scanning images", folder=self.config.scan.image_folder)
        from image_analyzer.scanner import ImageScanner

        scanner = ImageScanner(self.config.scan)
        all_images = scanner.scan_folder()

        if not all_images:
            logger.warning("No images found in folder", folder=self.config.scan.image_folder)
            return

        # Detect duplicates
        unique_images, duplicate_log = scanner.detect_duplicates(all_images)
        duplicates_found = len(duplicate_log)
        if duplicates_found:
            logger.info("Duplicates detected", count=duplicates_found)

        # Detect iPhone edit pairs
        edit_pairs = scanner.detect_iphone_edit_pairs(unique_images)
        if edit_pairs:
            logger.info("iPhone edit pairs detected", count=len(edit_pairs))

        # Build set of images to analyze (skip edit variants)
        edit_variant_paths = set()
        for variants in edit_pairs.values():
            edit_variant_paths.update(variants)
        images_to_analyze = [img for img in unique_images if img.path not in edit_variant_paths]

        logger.info(
            "Scan complete",
            total_found=len(all_images),
            unique=len(unique_images),
            to_analyze=len(images_to_analyze),
            duplicates=duplicates_found,
            edit_pairs=len(edit_pairs),
        )

        # Dry-run: estimate cost and stop
        if self.config.processing.dry_run:
            self._print_dry_run(images_to_analyze, duplicates_found, edit_pairs)
            return

        # Phase 2: Analyze
        logger.info("Phase 2: Analyzing images with Gemini")
        from image_analyzer.analyzer import GeminiAnalyzer
        from image_analyzer.utils.checkpoint import Checkpoint, CheckpointLockError
        from image_analyzer.utils.image_utils import load_image_bytes

        analyzer = GeminiAnalyzer(self.config.gemini)
        checkpoint = Checkpoint.load(self.config.processing.checkpoint_file)

        try:
            checkpoint.acquire_lock()
        except CheckpointLockError as e:
            logger.error(str(e))
            return

        results: list[ImageAnalysisResult] = []

        # Restore previously processed results
        for result_dict in checkpoint.get_results():
            try:
                results.append(ImageAnalysisResult.model_validate(result_dict))
            except Exception:
                pass  # Skip corrupted checkpoint entries

        try:
            progress = tqdm(
                images_to_analyze,
                desc="Analyzing",
                unit="img",
                disable=not logger.isEnabledFor("INFO"),
            )
            for image_info in progress:
                progress.set_postfix_str(image_info.filename[:30])

                # Skip already processed
                if checkpoint.is_processed(image_info.sha256_hash):
                    continue

                # Check budget
                if self.config.processing.max_cost is not None:
                    estimated = analyzer.estimate_cost(1)
                    if analyzer.total_cost + estimated["total_cost"] > self.config.processing.max_cost:
                        logger.warning(
                            "Budget limit reached",
                            spent=f"${analyzer.total_cost:.4f}",
                            limit=f"${self.config.processing.max_cost:.2f}",
                        )
                        break

                try:
                    image_bytes = load_image_bytes(
                        image_info.path,
                        max_dimension=self.config.scan.max_image_dimension,
                    )
                    result = analyzer.analyze_image(image_info.path, image_bytes)

                    # Attach iPhone edit pair paths
                    if image_info.filename in edit_pairs:
                        result.paired_images = edit_pairs[image_info.filename]

                    # Apply confidence threshold
                    if result.classification.confidence < self.config.processing.confidence_threshold:
                        result.classification.primary_type = EntityType.UNCLASSIFIED
                        result.flagged_for_review = True
                        result.review_reason = (
                            f"Low confidence: {result.classification.confidence:.2f}"
                        )

                    results.append(result)
                    checkpoint.add_result(
                        image_info.sha256_hash,
                        result.model_dump(mode="json"),
                    )
                except Exception as e:
                    logger.error("Failed to analyze image", path=image_info.path, error=str(e))
                    # Create unclassified result for failed images
                    from image_analyzer.models import ClassificationResult

                    fail_result = ImageAnalysisResult(
                        image_path=image_info.path,
                        original_filename=image_info.filename,
                        file_hash_sha256=image_info.sha256_hash,
                        perceptual_hash=image_info.perceptual_hash,
                        classification=ClassificationResult(
                            primary_type=EntityType.UNCLASSIFIED,
                            confidence=0.0,
                            reasoning=f"Analysis failed: {str(e)}",
                        ),
                        flagged_for_review=True,
                        review_reason=f"Analysis error: {str(e)}",
                    )
                    results.append(fail_result)
                    checkpoint.add_result(
                        image_info.sha256_hash,
                        fail_result.model_dump(mode="json"),
                    )

        finally:
            checkpoint.release_lock()

        logger.info("Analysis complete", analyzed=len(results))

        # Phase 3: Group
        logger.info("Phase 3: Grouping images")
        from image_analyzer.grouper import ImageGrouper

        grouper = ImageGrouper(self.config.processing.grouping_similarity_threshold)

        classified = [r for r in results if r.classification.primary_type != EntityType.UNCLASSIFIED]
        unclassified = [r for r in results if r.classification.primary_type == EntityType.UNCLASSIFIED]

        groups = grouper.group_images(classified)
        logger.info(
            "Grouping complete",
            groups=len(groups),
            unclassified=len(unclassified),
        )

        # Phase 4: Export
        logger.info("Phase 4: Exporting CSVs")
        from image_analyzer.exporter import CSVExporter

        exporter = CSVExporter(
            output_folder=self.config.scan.output_folder,
            rename_images=self.config.processing.rename_images,
            rename_pattern=self.config.processing.rename_pattern,
        )

        csv_files = exporter.export_csvs(groups, unclassified)
        for entity_type, csv_path in csv_files.items():
            logger.info("CSV written", type=entity_type, path=csv_path)

        if self.config.processing.rename_images:
            rename_map = exporter.rename_images(groups)
            logger.info("Images renamed", count=len(rename_map))

        duration = time.time() - start_time
        report = exporter.generate_summary_report(
            groups, unclassified, duration, duplicates_found
        )
        logger.info("Pipeline complete", duration=f"{duration:.1f}s")

        # Clean up checkpoint on success
        checkpoint.delete()

    def _print_dry_run(self, images, duplicates_found, edit_pairs):
        """Print cost estimate without making API calls."""
        from image_analyzer.analyzer import GeminiAnalyzer

        analyzer = GeminiAnalyzer.__new__(GeminiAnalyzer)
        estimate = GeminiAnalyzer.estimate_cost(analyzer, len(images))

        print("\n=== DRY RUN — Cost Estimate ===")
        print(f"Images to analyze:  {len(images)}")
        print(f"Duplicates skipped: {duplicates_found}")
        print(f"iPhone edit pairs:  {len(edit_pairs)} (variants skipped)")
        print(f"Model:              {self.config.gemini.model}")
        print(f"Est. input tokens:  {estimate['input_tokens']:,}")
        print(f"Est. output tokens: {estimate['output_tokens']:,}")
        print(f"Est. total cost:    ${estimate['total_cost']:.4f}")
        print(f"Est. time:          {estimate['estimated_minutes']:.1f} minutes")
        print(f"                    (at {self.config.gemini.requests_per_minute} req/min)")
        if self.config.processing.max_cost:
            print(f"Budget cap:         ${self.config.processing.max_cost:.2f}")
        print("================================\n")
