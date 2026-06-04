"""litellm-style success/failure callbacks for the in-process engine.

Register callables that receive a :class:`CallbackEvent` after each completion
succeeds or fails. This is the hosting-free home for usage/cost/latency logging.
Exceptions raised inside a callback are swallowed and logged so a misbehaving
hook can never break a request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from .models import Usage

logger = logging.getLogger("omnigate")


@dataclass
class CallbackEvent:
    """Payload handed to success/failure callbacks."""

    model: str
    provider: str
    usage: Optional[Usage]
    cost_usd: float
    latency_ms: int
    cached: bool
    fallback_used: bool
    exception: Optional[BaseException] = None


Callback = Callable[[CallbackEvent], None]

_success: list[Callback] = []
_failure: list[Callback] = []


def register(
    *,
    on_success: Optional[Callback] = None,
    on_failure: Optional[Callback] = None,
) -> None:
    """Register success and/or failure callbacks."""
    if on_success is not None:
        _success.append(on_success)
    if on_failure is not None:
        _failure.append(on_failure)


def reset() -> None:
    """Drop all registered callbacks (test helper)."""
    _success.clear()
    _failure.clear()


def _fire(callbacks: list[Callback], event: CallbackEvent) -> None:
    for cb in callbacks:
        try:
            cb(event)
        except Exception:  # noqa: BLE001 - a hook must never break the request
            logger.warning("omnigate callback raised", exc_info=True)


def fire_success(event: CallbackEvent) -> None:
    """Invoke all success callbacks with ``event`` (errors swallowed)."""
    _fire(_success, event)


def fire_failure(event: CallbackEvent) -> None:
    """Invoke all failure callbacks with ``event`` (errors swallowed)."""
    _fire(_failure, event)
