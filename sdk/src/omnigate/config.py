"""In-process engine configuration.

A plain dataclass (deliberately **not** pydantic-settings, which would
re-introduce the gateway's required ``DATABASE_URL``/``REDIS_URL`` env vars and
defeat the goal of running with zero hosting). All fields have safe defaults and
can be overridden from the environment via :meth:`EngineConfig.from_env`, from
``omnigate.configure(...)``, or per-call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_TRUTHY = {"1", "true", "yes", "on"}


def _env_str(name: str) -> Optional[str]:
    val = os.environ.get(name)
    return val if val not in (None, "") else None


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


@dataclass
class EngineConfig:
    """Tunables for the in-process engine. All optional with sane defaults."""

    timeout: float = 60.0

    # --- retry ---
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.25
    retry_max_delay: float = 8.0
    retry_jitter: float = 0.25

    # --- circuit breaker ---
    circuit_breaker_enabled: bool = True
    circuit_breaker_fail_threshold: int = 5
    circuit_breaker_cooldown: float = 30.0

    # --- response cache (opt-in) ---
    cache_enabled: bool = False
    cache_ttl: int = 300

    # --- local spend cap (off by default) ---
    max_spend_usd: Optional[float] = None

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Build a config from ``OMNIGATE_*`` environment variables."""
        max_spend_raw = _env_str("OMNIGATE_MAX_SPEND_USD")
        max_spend: Optional[float] = None
        if max_spend_raw is not None:
            try:
                max_spend = float(max_spend_raw)
            except ValueError:
                max_spend = None
        return cls(
            timeout=_env_float("OMNIGATE_TIMEOUT_SECONDS", 60.0),
            retry_max_attempts=_env_int("OMNIGATE_RETRY_MAX_ATTEMPTS", 3),
            retry_base_delay=_env_float("OMNIGATE_RETRY_BASE_DELAY_SECONDS", 0.25),
            retry_max_delay=_env_float("OMNIGATE_RETRY_MAX_DELAY_SECONDS", 8.0),
            retry_jitter=_env_float("OMNIGATE_RETRY_JITTER_SECONDS", 0.25),
            circuit_breaker_enabled=_env_bool(
                "OMNIGATE_CIRCUIT_BREAKER_ENABLED", True
            ),
            circuit_breaker_fail_threshold=_env_int(
                "OMNIGATE_CIRCUIT_BREAKER_FAIL_THRESHOLD", 5
            ),
            circuit_breaker_cooldown=_env_float(
                "OMNIGATE_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 30.0
            ),
            cache_enabled=_env_bool("OMNIGATE_CACHE_ENABLED", False),
            cache_ttl=_env_int("OMNIGATE_CACHE_TTL_SECONDS", 300),
            max_spend_usd=max_spend,
        )


_config: Optional[EngineConfig] = None


def get_config() -> EngineConfig:
    """Return the process-wide config, building it from env on first use."""
    global _config
    if _config is None:
        _config = EngineConfig.from_env()
    return _config


def set_config(cfg: EngineConfig) -> None:
    """Replace the process-wide config (used by ``configure()`` and tests)."""
    global _config
    _config = cfg
