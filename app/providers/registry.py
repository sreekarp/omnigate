"""Model -> provider routing.

Selection is by model-name prefix. Provider instances are created lazily and
cached (one per process) — except Azure, whose adapter is constructed per
request from the project's stored endpoint/deployment/api-version, so the
router builds it itself (see ``app/routers/chat.py``).
"""

from functools import lru_cache

from app.providers.base import AbstractProvider, ProviderError


def provider_name_for_model(model: str) -> str:
    """Return the provider name responsible for ``model``.

    Raises :class:`ProviderError` (400) for unrecognised models.
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
    raise ProviderError(f"No provider registered for model {model!r}", status_code=400)


@lru_cache
def _openai() -> AbstractProvider:
    from app.providers.openai import OpenAIProvider

    return OpenAIProvider()


@lru_cache
def _anthropic() -> AbstractProvider:
    from app.providers.anthropic import AnthropicProvider

    return AnthropicProvider()


@lru_cache
def _gemini() -> AbstractProvider:
    from app.providers.gemini import GeminiProvider

    return GeminiProvider()


def get_provider(name: str) -> AbstractProvider:
    """Return the cached adapter for a provider name.

    Azure is intentionally not constructed here — it requires per-request
    credential metadata, so the router builds it directly.
    """
    if name == "openai":
        return _openai()
    if name == "anthropic":
        return _anthropic()
    if name == "gemini":
        return _gemini()
    if name == "azure":
        raise ProviderError(
            "Azure provider is constructed per-request from stored credentials",
            status_code=500,
        )
    raise ProviderError(f"Unknown provider {name!r}", status_code=400)


def get_provider_for_model(model: str) -> AbstractProvider:
    """Return the provider adapter responsible for ``model`` (non-Azure)."""
    return get_provider(provider_name_for_model(model))
