"""Pure, dependency-free retry/backoff helpers (shared by sync + async)."""

from __future__ import annotations

import email.utils
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for hand-rolled exponential backoff with jitter."""

    max_retries: int = 2  # total attempts = max_retries + 1
    backoff_base: float = 0.5  # seconds
    backoff_max: float = 8.0
    jitter: float = 0.2  # +/- fraction of the computed delay
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )


def should_retry(status: Optional[int], attempt: int, cfg: RetryConfig) -> bool:
    """Return True if a request that produced ``status`` should be retried.

    ``attempt`` is the zero-based index of the attempt that just ran. A
    ``status`` of ``None`` denotes a transport-level error (connection/timeout),
    which is retryable.
    """
    if attempt >= cfg.max_retries:
        return False
    if status is None:  # transport error
        return True
    return status in cfg.retry_statuses


def compute_delay(
    attempt: int, cfg: RetryConfig, retry_after: Optional[float] = None
) -> float:
    """Seconds to wait before the next attempt.

    Honours a server-provided ``Retry-After`` (capped to ``backoff_max``) when
    present; otherwise exponential ``backoff_base * 2**attempt`` capped at
    ``backoff_max``, then +/- ``jitter`` fraction. Never negative.
    """
    if retry_after is not None and retry_after >= 0:
        return min(retry_after, cfg.backoff_max)

    raw = cfg.backoff_base * (2 ** attempt)
    capped = min(raw, cfg.backoff_max)
    if cfg.jitter:
        spread = capped * cfg.jitter
        capped += random.uniform(-spread, spread)
    return max(0.0, capped)


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse a ``Retry-After`` header into seconds-from-now.

    Accepts an integer number of seconds or an HTTP-date. Returns ``None`` if
    the value is absent or unparseable.
    """
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    # Integer seconds form.
    try:
        return float(int(value))
    except ValueError:
        pass
    # HTTP-date form.
    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = (dt - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)
