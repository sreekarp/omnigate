"""Model -> provider routing.

Selection is by model-name prefix. Provider instances are created lazily and
cached (one per process).
"""

from functools import lru_cache

from app.providers.anthropic import AnthropicProvider
from app.providers.base import AbstractProvider, ProviderError
from app.providers.openai import OpenAIProvider


@lru_cache
def _openai() -> OpenAIProvider:
    return OpenAIProvider()


@lru_cache
def _anthropic() -> AnthropicProvider:
    return AnthropicProvider()


def get_provider_for_model(model: str) -> AbstractProvider:
    """Return the provider adapter responsible for ``model``.

    Raises :class:`ProviderError` (400) for unrecognised models.
    """
    name = model.lower()
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3"):
        return _openai()
    if name.startswith("claude"):
        return _anthropic()
    raise ProviderError(
        f"No provider registered for model {model!r}", status_code=400
    )
