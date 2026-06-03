"""Offline unit tests for the model catalog.

``model_cards`` returns OpenAI-style cards carrying provider + pricing;
``model_card`` returns a single card or ``None`` for an unknown id. No I/O.
"""

from app.services.catalog import model_card, model_cards
from app.services.pricing import known_models


def test_model_cards_cover_all_known_models():
    cards = model_cards()
    ids = {c["id"] for c in cards}
    assert ids == set(known_models())


def test_card_shape():
    cards = model_cards()
    card = next(c for c in cards if c["id"] == "gpt-4o")
    assert card["object"] == "model"
    assert "created" in card
    assert card["provider"] == "openai"
    assert card["owned_by"] == "openai"
    assert card["pricing"] == {
        "input_per_1k_usd": 0.0025,
        "output_per_1k_usd": 0.010,
    }


def test_provider_and_owned_by_mapping():
    by_id = {c["id"]: c for c in model_cards()}
    assert by_id["claude-3-5-sonnet-latest"]["provider"] == "anthropic"
    assert by_id["claude-3-5-sonnet-latest"]["owned_by"] == "anthropic"
    assert by_id["gemini-1.5-flash"]["provider"] == "gemini"
    # gemini provider maps to owned_by "google".
    assert by_id["gemini-1.5-flash"]["owned_by"] == "google"


def test_model_card_single_known():
    card = model_card("gpt-4o-mini")
    assert card is not None
    assert card["id"] == "gpt-4o-mini"
    assert card["pricing"]["input_per_1k_usd"] == 0.00015


def test_model_card_unknown_is_none():
    assert model_card("totally-made-up-model") is None


def test_every_card_has_pricing():
    # known_models() are all priced, so every card carries pricing.
    for card in model_cards():
        assert card["pricing"] is not None
        assert card["pricing"]["input_per_1k_usd"] >= 0.0
        assert card["pricing"]["output_per_1k_usd"] >= 0.0
