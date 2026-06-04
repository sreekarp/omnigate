"""Model-name -> provider spec routing (by prefix).

Mirrors the gateway's ``provider_name_for_model`` selection rules. Spec instances
are created once and cached (they are stateless and pure).
"""

from __future__ import annotations

from functools import lru_cache

from ..exceptions import APIError
from .base import ProviderSpec


def provider_name_for_model(model: str) -> str:
    """Return the provider name responsible for ``model``.

    Raises :class:`APIError` (400) for unrecognised models.
    """
    name = model.lower()
    if name.startswith("azure/") or name.startswith("azure-"):
        return "azure"
    if (
        name.startswith("gpt")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
        or name.startswith("chatgpt")
    ):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gemini") or name.startswith("models/gemini"):
        return "gemini"
    raise APIError(f"No provider registered for model {model!r}", status_code=400)


@lru_cache
def _spec(name: str) -> ProviderSpec:
    if name == "openai":
        from .openai import OpenAISpec

        return OpenAISpec()
    if name == "anthropic":
        from .anthropic import AnthropicSpec

        return AnthropicSpec()
    if name == "gemini":
        from .gemini import GeminiSpec

        return GeminiSpec()
    if name == "azure":
        from .azure import AzureSpec

        return AzureSpec()
    raise APIError(f"Unknown provider {name!r}", status_code=400)


def spec_for_model(model: str) -> tuple[ProviderSpec, str]:
    """Return ``(spec, provider_name)`` for ``model``."""
    name = provider_name_for_model(model)
    return _spec(name), name
