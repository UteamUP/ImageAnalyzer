"""Tests for the Gemini analyzer module."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from image_analyzer.analyzer import GeminiAnalyzer
from image_analyzer.config import GeminiConfig
from image_analyzer.models import (
    EntityType,
    ExtractedAssetData,
    ExtractedChemicalData,
)
from image_analyzer.utils.rate_limiter import RetryHandler, TokenBucketRateLimiter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ASSET_JSON = json.dumps(
    {
        "classification": {
            "primary_type": "asset",
            "confidence": 0.92,
            "secondary_type": None,
            "reasoning": "Industrial hydraulic press visible with nameplate",
        },
        "extracted_data": {
            "name": "Hydraulic Press HP-500",
            "description": "Large industrial hydraulic press",
            "serial_number": "SN-12345",
            "model_number": "HP-500",
            "manufacturer_brand": "Acme Corp",
            "suggested_category": "Manufacturing",
            "suggested_vendor": "Acme Corp",
            "visible_condition": "Good",
            "is_vehicle": False,
        },
    }
)

VALID_CHEMICAL_JSON = json.dumps(
    {
        "classification": {
            "primary_type": "chemical",
            "confidence": 0.88,
            "secondary_type": None,
            "reasoning": "Chemical container with GHS labels and hazard statements",
        },
        "extracted_data": {
            "name": "Isopropyl Alcohol 99%",
            "description": "Industrial grade isopropyl alcohol solvent",
            "chemical_formula": "C3H8O",
            "cas_number": "67-63-0",
            "un_number": "UN1219",
            "ghs_hazard_class": "Flammable Liquid Category 2",
            "signal_word": "Danger",
            "physical_state": "Liquid",
            "color": "Colorless",
            "flash_point": 12.0,
            "unit_of_measure": "L",
            "hazard_statements": [
                "H225: Highly flammable liquid and vapour",
                "H319: Causes serious eye irritation",
                "H336: May cause drowsiness or dizziness",
            ],
            "precautionary_statements": [
                "P210: Keep away from heat, hot surfaces, sparks, open flames",
                "P280: Wear protective gloves/eye protection",
                "P305+P351+P338: IF IN EYES: Rinse cautiously with water",
                "P403+P235: Store in well-ventilated place, keep cool",
            ],
            "manufacturer_name": "ChemSupply Inc",
            "suggested_vendor": "ChemSupply Inc",
            "suggested_category": "Solvents",
        },
    }
)

INVALID_JSON_RESPONSE = '```json\n{"classification": {"primary_type": "asset", "confidence": 0.8, BROKEN'

LOW_CONFIDENCE_JSON = json.dumps(
    {
        "classification": {
            "primary_type": "unclassified",
            "confidence": 0.3,
            "secondary_type": None,
            "reasoning": "Blurry image, cannot determine content",
        },
        "extracted_data": None,
    }
)


def _make_mock_response(text: str) -> MagicMock:
    """Create a mock Gemini response with the given text."""
    mock = MagicMock()
    mock.text = text
    return mock


def _make_config(**overrides) -> GeminiConfig:
    """Create a GeminiConfig with test defaults."""
    defaults = {
        "api_key": "test-api-key-for-unit-tests",
        "model": "gemini-2.0-flash",
        "max_output_tokens": 4096,
        "temperature": 0.1,
        "requests_per_minute": 60,
        "max_retries": 1,
        "timeout_seconds": 10,
    }
    defaults.update(overrides)
    return GeminiConfig(**defaults)


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


@patch("image_analyzer.analyzer.genai", create=True)
@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.GenerationConfig")
class TestGeminiAnalyzer:
    """Tests for GeminiAnalyzer class."""

    def _create_analyzer(self, mock_gen_config, mock_model_cls, mock_configure, mock_genai):
        """Helper to create an analyzer with mocked google.generativeai."""
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        config = _make_config()
        analyzer = GeminiAnalyzer(config)
        return analyzer, mock_model

    def test_parse_valid_asset_json(
        self, mock_gen_config, mock_model_cls, mock_configure, mock_genai
    ):
        """Verify that a valid asset JSON response is correctly parsed."""
        analyzer, mock_model = self._create_analyzer(
            mock_gen_config, mock_model_cls, mock_configure, mock_genai
        )

        mock_model.generate_content.return_value = _make_mock_response(VALID_ASSET_JSON)

        # Create a minimal 1x1 red pixel JPEG for testing
        import io
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("RGB", (1, 1), "red").save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = analyzer.analyze_image("/tmp/test_asset.jpg", image_bytes)

        assert result.classification.primary_type == EntityType.ASSET
        assert result.classification.confidence == 0.92
        assert isinstance(result.extracted_data, ExtractedAssetData)
        assert result.extracted_data.name == "Hydraulic Press HP-500"
        assert result.extracted_data.serial_number == "SN-12345"
        assert result.extracted_data.manufacturer_brand == "Acme Corp"
        assert result.flagged_for_review is False

    def test_parse_valid_chemical_json(
        self, mock_gen_config, mock_model_cls, mock_configure, mock_genai
    ):
        """Verify chemical data with H-codes and P-codes is correctly parsed."""
        analyzer, mock_model = self._create_analyzer(
            mock_gen_config, mock_model_cls, mock_configure, mock_genai
        )

        mock_model.generate_content.return_value = _make_mock_response(VALID_CHEMICAL_JSON)

        import io
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("RGB", (1, 1), "blue").save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = analyzer.analyze_image("/tmp/test_chemical.jpg", image_bytes)

        assert result.classification.primary_type == EntityType.CHEMICAL
        assert result.classification.confidence == 0.88
        assert isinstance(result.extracted_data, ExtractedChemicalData)
        assert result.extracted_data.name == "Isopropyl Alcohol 99%"
        assert result.extracted_data.cas_number == "67-63-0"
        assert result.extracted_data.un_number == "UN1219"
        assert result.extracted_data.signal_word == "Danger"
        assert len(result.extracted_data.hazard_statements) == 3
        assert result.extracted_data.hazard_statements[0].startswith("H225")
        assert len(result.extracted_data.precautionary_statements) == 4
        assert result.extracted_data.precautionary_statements[1].startswith("P280")
        assert result.flagged_for_review is False

    def test_parse_invalid_json_retries(
        self, mock_gen_config, mock_model_cls, mock_configure, mock_genai
    ):
        """First response is invalid JSON; fix prompt returns valid JSON on retry."""
        analyzer, mock_model = self._create_analyzer(
            mock_gen_config, mock_model_cls, mock_configure, mock_genai
        )

        # First call returns broken JSON, second call (fix) returns valid asset JSON
        mock_model.generate_content.side_effect = [
            _make_mock_response(INVALID_JSON_RESPONSE),
            _make_mock_response(VALID_ASSET_JSON),
        ]

        import io
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("RGB", (1, 1), "green").save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = analyzer.analyze_image("/tmp/test_retry.jpg", image_bytes)

        assert result.classification.primary_type == EntityType.ASSET
        assert result.classification.confidence == 0.92
        assert isinstance(result.extracted_data, ExtractedAssetData)
        # Verify generate_content was called twice (original + fix)
        assert mock_model.generate_content.call_count == 2

    def test_parse_unclassified_low_confidence(
        self, mock_gen_config, mock_model_cls, mock_configure, mock_genai
    ):
        """Confidence below 0.5 should set flagged_for_review=True."""
        analyzer, mock_model = self._create_analyzer(
            mock_gen_config, mock_model_cls, mock_configure, mock_genai
        )

        mock_model.generate_content.return_value = _make_mock_response(LOW_CONFIDENCE_JSON)

        import io
        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.new("RGB", (1, 1), "gray").save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        result = analyzer.analyze_image("/tmp/test_blurry.jpg", image_bytes)

        assert result.classification.primary_type == EntityType.UNCLASSIFIED
        assert result.classification.confidence == 0.3
        assert result.extracted_data is None
        assert result.flagged_for_review is True
        assert "Low confidence" in result.review_reason

    def test_cost_estimation(
        self, mock_gen_config, mock_model_cls, mock_configure, mock_genai
    ):
        """Verify cost estimate calculation for a batch of images."""
        analyzer, _ = self._create_analyzer(
            mock_gen_config, mock_model_cls, mock_configure, mock_genai
        )

        estimate = analyzer.estimate_cost(100)

        assert estimate["image_count"] == 100
        # 100 images * (258 + 1500) = 175,800 input tokens
        assert estimate["estimated_input_tokens"] == 100 * (258 + 1500)
        # 100 images * 500 = 50,000 output tokens
        assert estimate["estimated_output_tokens"] == 100 * 500
        # Input cost: 175,800 / 1M * $0.10 = $0.01758
        assert estimate["estimated_input_cost_usd"] == pytest.approx(0.01758, abs=0.0001)
        # Output cost: 50,000 / 1M * $0.40 = $0.02
        assert estimate["estimated_output_cost_usd"] == pytest.approx(0.02, abs=0.001)
        # Total: ~$0.03758
        assert estimate["estimated_total_cost_usd"] == pytest.approx(0.03758, abs=0.001)


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_rate_limiter_basic(self):
        """Tokens are consumed and refilled correctly."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        # Should be able to acquire immediately (bucket starts full)
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Should be near-instant

        # Consume remaining tokens rapidly
        for _ in range(59):
            limiter.acquire()

        # Next acquire should block briefly (bucket empty, refills at 1/sec)
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        # Should have waited approximately 1 second for a token to refill
        assert elapsed >= 0.5  # Allow some tolerance

    def test_rate_limiter_refill(self):
        """After waiting, tokens refill based on elapsed time."""
        limiter = TokenBucketRateLimiter(requests_per_minute=600)  # 10/sec

        # Drain 10 tokens (one second's worth at 600 RPM)
        for _ in range(10):
            limiter.acquire()

        # Wait 100ms -- should refill ~1 token (10/sec * 0.1s)
        time.sleep(0.15)

        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start
        # Should be near-instant since token was refilled during sleep
        assert elapsed < 0.1


# ---------------------------------------------------------------------------
# Retry handler tests
# ---------------------------------------------------------------------------


class TestRetryHandler:
    """Tests for RetryHandler."""

    def test_retry_handler_transient_error(self):
        """Simulate 503 then success -- verify retry happens."""
        handler = RetryHandler(max_retries=3)

        mock_func = MagicMock()
        error_503 = Exception("503 Service Unavailable")
        mock_func.side_effect = [error_503, "success"]

        result = handler.execute(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_retry_handler_permanent_error(self):
        """Simulate 400 Bad Request -- verify no retry, fails immediately."""
        handler = RetryHandler(max_retries=3)

        mock_func = MagicMock()
        error_400 = ValueError("400 Bad Request: invalid input")
        mock_func.side_effect = error_400

        with pytest.raises(ValueError, match="400 Bad Request"):
            handler.execute(mock_func)

        # Should have been called exactly once (no retries for 4xx)
        assert mock_func.call_count == 1

    def test_retry_handler_connection_error(self):
        """ConnectionError is always retried."""
        handler = RetryHandler(max_retries=2)

        mock_func = MagicMock()
        mock_func.side_effect = [
            ConnectionError("Connection refused"),
            ConnectionError("Connection refused"),
            "connected",
        ]

        result = handler.execute(mock_func)
        assert result == "connected"
        assert mock_func.call_count == 3

    def test_retry_handler_exhausted(self):
        """All retries exhausted raises the last error."""
        handler = RetryHandler(max_retries=1)

        mock_func = MagicMock()
        mock_func.side_effect = ConnectionError("persistent failure")

        with pytest.raises(ConnectionError, match="persistent failure"):
            handler.execute(mock_func)

        assert mock_func.call_count == 2  # initial + 1 retry
