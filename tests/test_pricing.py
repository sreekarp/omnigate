from decimal import Decimal

from app.services.pricing import compute_cost


def test_known_model_cost():
    # gpt-4o-mini: 0.00015/1k in, 0.0006/1k out
    cost = compute_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
    assert cost == Decimal("0.000750")


def test_zero_tokens_is_zero():
    assert compute_cost("gpt-4o", 0, 0) == Decimal("0")


def test_unknown_model_is_zero():
    assert compute_cost("totally-made-up-model", 1000, 1000) == Decimal("0")
