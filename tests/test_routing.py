import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.providers.openai import OpenAIProvider
from app.providers.registry import get_provider_for_model


def test_openai_models_route_to_openai():
    assert isinstance(get_provider_for_model("gpt-4o-mini"), OpenAIProvider)


def test_claude_models_route_to_anthropic():
    assert isinstance(
        get_provider_for_model("claude-3-5-sonnet-latest"), AnthropicProvider
    )


def test_unknown_model_raises():
    with pytest.raises(ProviderError) as exc:
        get_provider_for_model("mistral-large")
    assert exc.value.status_code == 400
