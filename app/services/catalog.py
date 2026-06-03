"""Model catalog: combines the pricing table with provider routing into
OpenAI-style model cards for ``GET /v1/models`` (and the dashboard/CLI).
"""

from typing import Any

from app.providers.base import ProviderError
from app.providers.registry import provider_name_for_model
from app.services.pricing import get_price, known_models

# Maps internal provider name -> OpenAI-style "owned_by".
_OWNED_BY = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "azure": "azure",
}


def _card(model: str) -> dict[str, Any]:
    try:
        provider = provider_name_for_model(model)
    except ProviderError:
        provider = "unknown"
    price = get_price(model)
    return {
        "id": model,
        "object": "model",
        "created": 0,
        "owned_by": _OWNED_BY.get(provider, provider),
        "provider": provider,
        "pricing": (
            {
                "input_per_1k_usd": float(price[0]),
                "output_per_1k_usd": float(price[1]),
            }
            if price is not None
            else None
        ),
    }


def model_cards() -> list[dict[str, Any]]:
    """All catalogued (priced + routable) models as OpenAI-style cards."""
    return [_card(model) for model in known_models()]


def model_card(model_id: str) -> dict[str, Any] | None:
    """A single model card by id, or None if unknown."""
    if get_price(model_id) is None and model_id not in known_models():
        return None
    return _card(model_id)
