"""Provider key + Azure target resolution for the in-process engine.

Precedence for keys: an explicit ``api_key=`` argument wins, then a programmatic
override set via ``omnigate.configure(...)``, then environment variables. Azure
additionally needs an endpoint, deployment, and api-version (its
:class:`Target`); the deployment comes from the ``azure/<deployment>`` model id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .exceptions import APIError

_DEFAULT_AZURE_API_VERSION = "2024-10-21"

#: provider -> ordered env var names to check for its key.
_ENV: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "azure": ["AZURE_OPENAI_API_KEY"],
}

# Programmatic overrides (set by omnigate.configure(...)).
_overrides: dict[str, str] = {}
_azure: dict[str, str] = {}  # keys: "endpoint", "api_version"


@dataclass
class Target:
    """Azure-only call target (endpoint/deployment/api-version)."""

    endpoint: str
    deployment: str
    api_version: str


def reset() -> None:
    """Clear all programmatic overrides (test helper)."""
    _overrides.clear()
    _azure.clear()


def set_override(provider: str, api_key: str) -> None:
    """Store a programmatic key override for ``provider``."""
    _overrides[provider] = api_key


def set_azure(*, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> None:
    """Store programmatic Azure endpoint/api-version overrides."""
    if endpoint is not None:
        _azure["endpoint"] = endpoint
    if api_version is not None:
        _azure["api_version"] = api_version


def _from_env(provider: str) -> Optional[str]:
    for name in _ENV.get(provider, []):
        val = os.environ.get(name)
        if val:
            return val
    return None


def resolve_key(provider: str, explicit: Optional[str]) -> str:
    """Resolve the API key for ``provider`` (explicit > override > env).

    Raises :class:`APIError` (400) with an actionable message when none is found.
    """
    if explicit:
        return explicit
    if provider in _overrides:
        return _overrides[provider]
    env_key = _from_env(provider)
    if env_key:
        return env_key
    env_names = " or ".join(_ENV.get(provider, [])) or f"{provider.upper()}_API_KEY"
    raise APIError(
        f"No API key for provider {provider!r}. Set {env_names} or pass api_key=.",
        status_code=400,
    )


def resolve_target(
    provider: str,
    model: str,
    *,
    api_base: Optional[str],
    api_version: Optional[str],
) -> Optional[Target]:
    """Resolve the Azure call target, or ``None`` for non-Azure providers."""
    if provider != "azure":
        return None

    endpoint = api_base or _azure.get("endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise APIError(
            "Azure endpoint missing. Pass api_base= or set AZURE_OPENAI_ENDPOINT.",
            status_code=400,
        )

    deployment: Optional[str] = None
    if model.lower().startswith("azure/"):
        deployment = model.split("/", 1)[1] or None
    if not deployment:
        raise APIError(
            "Azure deployment not specified. Use model 'azure/<deployment>'.",
            status_code=400,
        )

    version = (
        api_version
        or _azure.get("api_version")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or _DEFAULT_AZURE_API_VERSION
    )
    return Target(endpoint=endpoint.rstrip("/"), deployment=deployment, api_version=version)
