"""Application configuration loaded from environment variables.

Uses pydantic-settings (Pydantic v2). Field names map case-insensitively to
environment variables, so ``database_url`` reads ``DATABASE_URL``.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Required infrastructure ---
    database_url: str
    redis_url: str
    secret_key: str
    admin_api_key: str

    # --- Provider credentials (optional so the app can boot without both) ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # --- Tunables ---
    default_rate_limit_per_min: int = 60
    request_timeout_seconds: float = 60.0
    log_level: str = "INFO"
    # "text" (human) or "json" (structured) application logs.
    log_format: str = "text"

    # --- Resilience: retry ---
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 0.25
    retry_max_delay_seconds: float = 8.0
    retry_jitter_seconds: float = 0.25

    # --- Resilience: circuit breaker ---
    circuit_breaker_enabled: bool = True
    circuit_breaker_fail_threshold: int = 5
    circuit_breaker_cooldown_seconds: float = 30.0
    circuit_breaker_backend: str = "memory"  # "memory" | "redis"

    # --- Response cache (opt-in; also requires a per-request flag) ---
    response_cache_enabled: bool = False
    response_cache_ttl_seconds: int = 300
    response_cache_prefix: str = "respcache:"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once per process)."""
    return Settings()  # type: ignore[call-arg]  # values come from env
