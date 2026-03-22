"""Gemini Vision AI analyzer for inventory image classification and data extraction."""

import json
import re
from pathlib import Path

import structlog
from pydantic import ValidationError

from image_analyzer.config import GeminiConfig
from image_analyzer.models import (
    ClassificationResult,
    EntityType,
    ExtractedAssetData,
    ExtractedChemicalData,
    ExtractedPartData,
    ExtractedToolData,
    ImageAnalysisResult,
)
from image_analyzer.prompts import JSON_FIX_PROMPT, UNIFIED_ANALYSIS_PROMPT
from image_analyzer.utils.rate_limiter import RetryHandler, TokenBucketRateLimiter

logger = structlog.get_logger(__name__)

# Type alias for extracted data union
ExtractedData = ExtractedAssetData | ExtractedToolData | ExtractedPartData | ExtractedChemicalData

# Mapping from EntityType to the corresponding Pydantic model
_ENTITY_MODEL_MAP: dict[EntityType, type[ExtractedData]] = {
    EntityType.ASSET: ExtractedAssetData,
    EntityType.TOOL: ExtractedToolData,
    EntityType.PART: ExtractedPartData,
    EntityType.CHEMICAL: ExtractedChemicalData,
}

# Gemini 2.0 Flash pricing (per 1M tokens, as of 2025)
_INPUT_COST_PER_1M_TOKENS = 0.10   # USD
_OUTPUT_COST_PER_1M_TOKENS = 0.40  # USD
_ESTIMATED_INPUT_TOKENS_PER_IMAGE = 258   # ~258 tokens for image encoding
_ESTIMATED_PROMPT_TOKENS = 1500           # prompt template overhead
_ESTIMATED_OUTPUT_TOKENS = 500            # average JSON response


class GeminiAnalyzer:
    """Analyze inventory images using Google Gemini Vision AI."""

    def __init__(self, config: GeminiConfig) -> None:
        import google.generativeai as genai

        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(
            model_name=config.model,
            generation_config=genai.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            ),
        )
        self._config = config
        self._rate_limiter = TokenBucketRateLimiter(config.requests_per_minute)
        self._retry_handler = RetryHandler(max_retries=config.max_retries)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._images_analyzed = 0

        logger.info(
            "analyzer.initialized",
            model=config.model,
            temperature=config.temperature,
            rpm=config.requests_per_minute,
        )

    def analyze_image(self, image_path: str, image_bytes: bytes) -> ImageAnalysisResult:
        """Send an image to Gemini for classification and data extraction.

        Args:
            image_path: Path to the image file (for metadata in the result).
            image_bytes: Raw image bytes to send to Gemini.

        Returns:
            ImageAnalysisResult with classification and extracted data.
        """
        original_filename = Path(image_path).name

        logger.info("analyzer.analyzing", image=original_filename)

        # Wait for rate limiter
        self._rate_limiter.acquire()

        # Call Gemini with retry handling
        try:
            response = self._retry_handler.execute(
                self._call_gemini, image_bytes
            )
        except Exception as exc:
            logger.error(
                "analyzer.api_error",
                image=original_filename,
                error=str(exc),
            )
            return ImageAnalysisResult(
                image_path=image_path,
                original_filename=original_filename,
                file_hash_sha256="",
                classification=ClassificationResult(
                    primary_type=EntityType.UNCLASSIFIED,
                    confidence=0.0,
                    reasoning=f"API error: {exc}",
                ),
                extracted_data=None,
                flagged_for_review=True,
                review_reason=f"API error: {exc}",
            )

        # Parse the response
        response_text = response.text if hasattr(response, "text") else str(response)
        classification, extracted_data = self._parse_response(response_text, image_path)

        # Track costs
        self._images_analyzed += 1
        self._total_input_tokens += _ESTIMATED_INPUT_TOKENS_PER_IMAGE + _ESTIMATED_PROMPT_TOKENS
        self._total_output_tokens += _ESTIMATED_OUTPUT_TOKENS

        # Flag low confidence for review
        flagged = classification.confidence < 0.5
        review_reason = None
        if flagged:
            review_reason = f"Low confidence: {classification.confidence:.2f}"

        result = ImageAnalysisResult(
            image_path=image_path,
            original_filename=original_filename,
            file_hash_sha256="",
            classification=classification,
            extracted_data=extracted_data,
            flagged_for_review=flagged,
            review_reason=review_reason,
        )

        logger.info(
            "analyzer.result",
            image=original_filename,
            entity_type=classification.primary_type.value,
            confidence=classification.confidence,
            flagged=flagged,
        )

        return result

    def _call_gemini(self, image_bytes: bytes):
        """Make the actual API call to Gemini. Separated for retry handling."""
        import PIL.Image
        import io

        image = PIL.Image.open(io.BytesIO(image_bytes))
        response = self._model.generate_content(
            [UNIFIED_ANALYSIS_PROMPT, image],
        )
        return response

    def _parse_response(
        self, response_text: str, image_path: str
    ) -> tuple[ClassificationResult, ExtractedData | None]:
        """Parse the JSON response from Gemini.

        Attempts to parse the JSON directly. If parsing fails, sends the
        broken JSON back to Gemini with JSON_FIX_PROMPT for correction.

        Args:
            response_text: Raw text response from Gemini.
            image_path: Image path for error context.

        Returns:
            Tuple of (ClassificationResult, extracted data or None).
        """
        original_filename = Path(image_path).name

        # Try to parse JSON from the response
        parsed = self._try_parse_json(response_text)

        if parsed is None:
            # Attempt to fix the JSON via a follow-up Gemini call
            logger.warning(
                "analyzer.invalid_json",
                image=original_filename,
                response_preview=response_text[:200],
            )
            parsed = self._attempt_json_fix(response_text)

        if parsed is None:
            logger.error(
                "analyzer.json_parse_failed",
                image=original_filename,
            )
            return (
                ClassificationResult(
                    primary_type=EntityType.UNCLASSIFIED,
                    confidence=0.0,
                    reasoning="Failed to parse Gemini response as JSON",
                ),
                None,
            )

        # Extract classification
        try:
            classification_data = parsed.get("classification", {})
            classification = ClassificationResult(**classification_data)
        except (ValidationError, TypeError) as exc:
            logger.error(
                "analyzer.classification_validation_error",
                image=original_filename,
                error=str(exc),
            )
            return (
                ClassificationResult(
                    primary_type=EntityType.UNCLASSIFIED,
                    confidence=0.0,
                    reasoning=f"Classification validation error: {exc}",
                ),
                None,
            )

        # Extract entity-specific data
        extracted_data = None
        raw_data = parsed.get("extracted_data")

        if raw_data is not None and classification.primary_type != EntityType.UNCLASSIFIED:
            model_class = _ENTITY_MODEL_MAP.get(classification.primary_type)
            if model_class is not None:
                try:
                    extracted_data = model_class(**raw_data)
                except (ValidationError, TypeError) as exc:
                    logger.warning(
                        "analyzer.extraction_validation_error",
                        image=original_filename,
                        entity_type=classification.primary_type.value,
                        error=str(exc),
                    )
                    # Keep the classification but flag for review
                    classification = ClassificationResult(
                        primary_type=classification.primary_type,
                        confidence=classification.confidence,
                        secondary_type=classification.secondary_type,
                        reasoning=classification.reasoning,
                    )
                    # Return with no extracted_data -- will be flagged
                    return classification, None

        return classification, extracted_data

    def _try_parse_json(self, text: str) -> dict | None:
        """Attempt to parse JSON from text, stripping markdown fences if present."""
        cleaned = text.strip()

        # Remove markdown JSON fences if present
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def _attempt_json_fix(self, broken_text: str) -> dict | None:
        """Send broken JSON back to Gemini for correction."""
        fix_prompt = JSON_FIX_PROMPT.format(broken_json=broken_text)

        try:
            self._rate_limiter.acquire()
            response = self._retry_handler.execute(
                self._model.generate_content, fix_prompt
            )
            fix_text = response.text if hasattr(response, "text") else str(response)
            return self._try_parse_json(fix_text)
        except Exception as exc:
            logger.error("analyzer.json_fix_failed", error=str(exc))
            return None

    def _estimate_cost(self, image_count: int) -> dict:
        """Estimate cost for processing a batch of images.

        Based on Gemini 2.0 Flash pricing:
        - Input: $0.10 per 1M tokens (~258 tokens per image + ~1500 prompt tokens)
        - Output: $0.40 per 1M tokens (~500 tokens per response)

        Args:
            image_count: Number of images to estimate for.

        Returns:
            Dict with token counts and estimated costs in USD.
        """
        total_input = image_count * (_ESTIMATED_INPUT_TOKENS_PER_IMAGE + _ESTIMATED_PROMPT_TOKENS)
        total_output = image_count * _ESTIMATED_OUTPUT_TOKENS

        input_cost = (total_input / 1_000_000) * _INPUT_COST_PER_1M_TOKENS
        output_cost = (total_output / 1_000_000) * _OUTPUT_COST_PER_1M_TOKENS

        return {
            "image_count": image_count,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_input_cost_usd": round(input_cost, 6),
            "estimated_output_cost_usd": round(output_cost, 6),
            "estimated_total_cost_usd": round(input_cost + output_cost, 6),
        }

    @property
    def total_cost(self) -> dict:
        """Running total of estimated costs based on images analyzed so far."""
        input_cost = (self._total_input_tokens / 1_000_000) * _INPUT_COST_PER_1M_TOKENS
        output_cost = (self._total_output_tokens / 1_000_000) * _OUTPUT_COST_PER_1M_TOKENS

        return {
            "images_analyzed": self._images_analyzed,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_input_cost_usd": round(input_cost, 6),
            "total_output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }
