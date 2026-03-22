"""Rate limiting and retry utilities for Gemini API calls."""

import random
import time

import structlog

logger = structlog.get_logger(__name__)


class TokenBucketRateLimiter:
    """Token bucket rate limiter to stay within Gemini API quotas."""

    def __init__(self, requests_per_minute: int) -> None:
        self._rpm = max(1, requests_per_minute)
        self._tokens = float(self._rpm)
        self._max_tokens = float(self._rpm)
        self._refill_rate = self._rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()

    def acquire(self) -> None:
        """Block until a token is available, then consume one."""
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                logger.debug(
                    "rate_limiter.token_acquired",
                    tokens_remaining=round(self._tokens, 2),
                    rpm=self._rpm,
                )
                return
            # Calculate wait time until at least one token is available
            wait_time = (1.0 - self._tokens) / self._refill_rate
            logger.debug(
                "rate_limiter.waiting",
                wait_seconds=round(wait_time, 3),
            )
            time.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill = now


class RetryHandler:
    """Retry transient errors with exponential backoff and jitter."""

    # HTTP status codes that are transient and should be retried
    _TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, max_retries: int = 3) -> None:
        self._max_retries = max(0, max_retries)

    def execute(self, func, *args, **kwargs):
        """Execute func with retries on transient errors.

        Retries on:
        - Network/connection errors (OSError, ConnectionError, TimeoutError)
        - HTTP 429 (rate limited)
        - HTTP 5xx (server errors)

        Does NOT retry:
        - HTTP 4xx errors (except 429) -- these are permanent client errors
        - ValueError, TypeError, KeyError -- programming errors
        """
        last_exception = None

        for attempt in range(self._max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (OSError, ConnectionError, TimeoutError) as exc:
                last_exception = exc
                if attempt >= self._max_retries:
                    logger.error(
                        "retry.exhausted",
                        error=str(exc),
                        attempts=attempt + 1,
                    )
                    raise
                self._backoff(attempt, exc)
            except Exception as exc:
                # Check if this is an API error with a retryable status code
                status_code = self._extract_status_code(exc)
                if status_code is not None and status_code in self._TRANSIENT_STATUS_CODES:
                    last_exception = exc
                    if attempt >= self._max_retries:
                        logger.error(
                            "retry.exhausted",
                            error=str(exc),
                            status_code=status_code,
                            attempts=attempt + 1,
                        )
                        raise
                    self._backoff(attempt, exc)
                else:
                    # Non-transient error (4xx except 429, programming errors) -- fail fast
                    raise

        # Should not reach here, but just in case
        raise last_exception  # type: ignore[misc]

    def _backoff(self, attempt: int, exc: Exception) -> None:
        """Sleep with exponential backoff + jitter."""
        base_delay = 2 ** attempt  # 1, 2, 4, 8, ...
        jitter = random.uniform(0, base_delay * 0.5)
        delay = base_delay + jitter
        logger.warning(
            "retry.backoff",
            attempt=attempt + 1,
            max_retries=self._max_retries,
            delay_seconds=round(delay, 2),
            error=str(exc),
        )
        time.sleep(delay)

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        """Try to extract an HTTP status code from an exception.

        Works with google-generativeai exceptions and standard HTTP errors.
        """
        # google.api_core.exceptions have a .code or .grpc_status_code attribute
        if hasattr(exc, "code"):
            code = exc.code
            if callable(code):
                code = code()
            if isinstance(code, int):
                return code
        # Some exceptions store it as status_code
        if hasattr(exc, "status_code"):
            return int(exc.status_code)
        # Check the string representation for common patterns
        exc_str = str(exc)
        if "429" in exc_str:
            return 429
        if "503" in exc_str:
            return 503
        return None
