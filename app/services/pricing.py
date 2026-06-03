"""Token pricing and cost computation.

Prices are USD per 1,000 tokens, split into input (prompt) and output
(completion). Update this table as provider pricing changes. Unknown models
fall back to a zero price (cost 0) and log a warning.
"""

from decimal import Decimal

from app.logging_config import get_logger

logger = get_logger(__name__)

# (input_per_1k, output_per_1k) in USD.
_PRICING: dict[str, tuple[Decimal, Decimal]] = {
    # --- OpenAI ---
    "gpt-4o": (Decimal("0.0025"), Decimal("0.010")),
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "gpt-4-turbo": (Decimal("0.010"), Decimal("0.030")),
    "gpt-3.5-turbo": (Decimal("0.0005"), Decimal("0.0015")),
    # --- Anthropic ---
    "claude-3-5-sonnet-latest": (Decimal("0.003"), Decimal("0.015")),
    "claude-3-5-sonnet-20241022": (Decimal("0.003"), Decimal("0.015")),
    "claude-3-5-haiku-latest": (Decimal("0.0008"), Decimal("0.004")),
    "claude-3-opus-latest": (Decimal("0.015"), Decimal("0.075")),
}

_ZERO = Decimal("0")


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    """Return the USD cost for a request, rounded to 6 decimal places."""
    price = _PRICING.get(model)
    if price is None:
        logger.warning("No pricing for model %r; recording cost as 0", model)
        return _ZERO

    input_per_1k, output_per_1k = price
    cost = (
        (Decimal(prompt_tokens) / Decimal(1000)) * input_per_1k
        + (Decimal(completion_tokens) / Decimal(1000)) * output_per_1k
    )
    return cost.quantize(Decimal("0.000001"))
