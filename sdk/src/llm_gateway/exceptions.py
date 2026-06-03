"""Typed exception hierarchy + a classifier mapping HTTP responses to errors.

The gateway returns FastAPI-style error envelopes ``{"detail": ...}`` where
``detail`` may be a string or, for Pydantic validation errors, a list. The
classifier normalises this and disambiguates a *gateway* auth failure (401)
from an *upstream provider* failure that the gateway surfaces with the
provider's own status code (e.g. a real OpenAI 401).
"""

from __future__ import annotations

from typing import Any, Optional


class GatewayError(Exception):
    """Base class for every error raised by the SDK."""


class APIError(GatewayError):
    """A non-2xx HTTP response from the gateway."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        detail: Any = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail  # raw parsed "detail": str | list | None
        self.request_id = request_id

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.status_code is not None:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthError(APIError):
    """401 — the gateway api key is missing or invalid."""


class BudgetExceededError(APIError):
    """402 — the project or organisation daily/monthly budget is exhausted."""


class ProviderError(APIError):
    """An upstream provider failure surfaced by the gateway (502 or proxied)."""


class RateLimitError(APIError):
    """429 — rate limit exceeded. ``retry_after`` is parsed from the header."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[float] = None,
        status_code: Optional[int] = None,
        detail: Any = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            detail=detail,
            request_id=request_id,
        )
        self.retry_after = retry_after


class ConnectionError(GatewayError):
    """Could not reach the gateway (network/timeout) after exhausting retries."""


def normalise_detail(detail: Any) -> str:
    """Turn a ``detail`` value (str | list | dict | None) into a message."""
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        # Pydantic validation errors: a list of {"loc","msg","type",...}.
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict) and "msg" in item:
                loc = item.get("loc")
                if loc:
                    parts.append(f"{'.'.join(str(x) for x in loc)}: {item['msg']}")
                else:
                    parts.append(str(item["msg"]))
            else:
                parts.append(str(item))
        return "; ".join(parts) if parts else str(detail)
    return str(detail)


def _looks_like_provider_failure(detail_msg: str) -> bool:
    """Heuristic: does this detail describe an upstream provider failure?"""
    lowered = detail_msg.lower()
    if not lowered:
        return False
    prefixes = ("openai", "anthropic", "gemini", "azure", "no provider", "no ")
    if lowered.startswith(prefixes):
        return True
    # Patterns like "OpenAI error 401: ..." or "... error 502 ...".
    if "error 4" in lowered or "error 5" in lowered:
        return True
    return False


def classify(
    status_code: int,
    detail: Any,
    *,
    retry_after: Optional[float] = None,
    request_id: Optional[str] = None,
) -> APIError:
    """Map an HTTP status + parsed detail to a typed exception.

    Order matters: 429 and 402 are unambiguous; provider failures are detected
    by detail substring (so a provider-surfaced 401 is not mistaken for a
    gateway auth failure) before falling back to 401 -> AuthError.
    """
    msg = normalise_detail(detail) or f"HTTP {status_code}"

    if status_code == 429:
        return RateLimitError(
            msg,
            retry_after=retry_after,
            status_code=status_code,
            detail=detail,
            request_id=request_id,
        )
    if status_code == 402:
        return BudgetExceededError(
            msg, status_code=status_code, detail=detail, request_id=request_id
        )
    if status_code == 502 or _looks_like_provider_failure(msg):
        return ProviderError(
            msg, status_code=status_code, detail=detail, request_id=request_id
        )
    if status_code == 401:
        return AuthError(
            msg, status_code=status_code, detail=detail, request_id=request_id
        )
    return APIError(
        msg, status_code=status_code, detail=detail, request_id=request_id
    )
