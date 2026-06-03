"""Offline unit tests for model -> provider routing.

Covers ``provider_name_for_model`` prefix selection across all providers, the
400 ``ProviderError`` for unknown models, and that ``get_provider('azure')``
raises a 500 (Azure is built per-request, never cached here).
"""

import pytest

from app.providers.base import ProviderError
from app.providers.registry import (
    get_provider,
    get_provider_for_model,
    provider_name_for_model,
)


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "GPT-4",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
        "chatgpt-4o-latest",
    ],
)
def test_openai_models(model):
    assert provider_name_for_model(model) == "openai"


@pytest.mark.parametrize(
    "model",
    ["claude-3-5-sonnet-latest", "claude-opus-4", "CLAUDE-3-haiku"],
)
def test_anthropic_models(model):
    assert provider_name_for_model(model) == "anthropic"


@pytest.mark.parametrize(
    "model",
    ["gemini-1.5-flash", "gemini-2.0-flash", "models/gemini-1.5-pro"],
)
def test_gemini_models(model):
    assert provider_name_for_model(model) == "gemini"


@pytest.mark.parametrize("model", ["azure/gpt-4o", "azure-gpt4o", "AZURE/my-deploy"])
def test_azure_models(model):
    assert provider_name_for_model(model) == "azure"


def test_unknown_model_raises_400():
    with pytest.raises(ProviderError) as exc:
        provider_name_for_model("mistral-large")
    assert exc.value.status_code == 400


def test_get_provider_azure_raises_500():
    with pytest.raises(ProviderError) as exc:
        get_provider("azure")
    assert exc.value.status_code == 500


def test_get_provider_unknown_raises_400():
    with pytest.raises(ProviderError) as exc:
        get_provider("not-a-provider")
    assert exc.value.status_code == 400


def test_get_provider_known_names_have_matching_name_attr():
    assert get_provider("openai").name == "openai"
    assert get_provider("anthropic").name == "anthropic"
    assert get_provider("gemini").name == "gemini"


def test_get_provider_for_model_routes_and_caches():
    p1 = get_provider_for_model("gpt-4o")
    p2 = get_provider_for_model("gpt-4o-mini")
    assert p1.name == "openai"
    assert p1 is p2  # lru_cache returns the same singleton
