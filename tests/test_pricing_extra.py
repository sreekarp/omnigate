"""Extra offline unit tests for pricing resolution + cost computation.

Covers ``get_price`` normalisation (``azure/`` prefix, dated/version suffixes
like ``-2024-08-06``, ``-20241022``, ``-002``), ``known_models``, and
``compute_cost`` for gemini and azure-prefixed models. Pure functions, no I/O.
"""

from decimal import Decimal

from app.services.pricing import compute_cost, get_price, known_models


def test_exact_match():
    assert get_price("gpt-4o") == (Decimal("0.0025"), Decimal("0.010"))


def test_azure_prefix_stripped():
    assert get_price("azure/gpt-4o") == get_price("gpt-4o")
    assert get_price("azure/gpt-4o-mini") == get_price("gpt-4o-mini")


def test_dated_suffix_yyyy_mm_dd_trimmed():
    # gpt-4o-2024-08-06 -> gpt-4o
    assert get_price("gpt-4o-2024-08-06") == get_price("gpt-4o")


def test_dated_suffix_yyyymmdd_trimmed():
    # claude-3-5-sonnet-20241022 is itself in the table; also a generic compact
    # date trims. Use a model with only the base priced.
    assert get_price("claude-3-5-sonnet-20241022") == (
        Decimal("0.003"),
        Decimal("0.015"),
    )
    # gemini-2.0-flash-20241022 -> gemini-2.0-flash
    assert get_price("gemini-2.0-flash-20241022") == get_price("gemini-2.0-flash")


def test_numeric_002_suffix_trimmed():
    # gemini-1.5-flash-002 -> gemini-1.5-flash
    assert get_price("gemini-1.5-flash-002") == get_price("gemini-1.5-flash")


def test_azure_prefix_and_dated_suffix_combined():
    assert get_price("azure/gpt-4o-2024-08-06") == get_price("gpt-4o")


def test_unknown_model_returns_none():
    assert get_price("totally-made-up-model") is None


def test_known_models_sorted_and_nonempty():
    models = known_models()
    assert models == sorted(models)
    assert "gpt-4o" in models
    assert "gemini-1.5-flash" in models
    assert len(models) > 10


def test_compute_cost_gemini():
    # gemini-1.5-flash: 0.000075/1k in, 0.0003/1k out
    cost = compute_cost("gemini-1.5-flash", prompt_tokens=2000, completion_tokens=1000)
    expected = (Decimal("2000") / 1000) * Decimal("0.000075") + (
        Decimal("1000") / 1000
    ) * Decimal("0.0003")
    assert cost == expected.quantize(Decimal("0.000001"))


def test_compute_cost_azure_prefixed_matches_base():
    azure_cost = compute_cost("azure/gpt-4o", 1000, 1000)
    base_cost = compute_cost("gpt-4o", 1000, 1000)
    assert azure_cost == base_cost
    assert azure_cost > Decimal("0")


def test_compute_cost_rounds_to_six_places():
    cost = compute_cost("gpt-4o-mini", 1, 1)
    assert cost == cost.quantize(Decimal("0.000001"))


def test_compute_cost_unknown_is_zero():
    assert compute_cost("no-such-model", 1000, 1000) == Decimal("0")
