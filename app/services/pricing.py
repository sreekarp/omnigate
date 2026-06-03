"""Token pricing and cost computation.

Prices are USD per 1,000 tokens, split into input (prompt) and output
(completion). Update this table as provider pricing changes. Unknown models
fall back to a zero price (cost 0) and log a warning.

Model-name resolution is forgiving: an ``azure/`` prefix is stripped and dated
suffixes (e.g. ``-2024-08-06``, ``-20241022``, ``-002``) are progressively
trimmed so versioned model ids still price correctly.

NOTE: the figures below are illustrative list prices and should be verified
against each provider's current pricing page before relying on them for
billing.
"""

import re
from decimal import Decimal

from app.logging_config import get_logger

logger = get_logger(__name__)

# (input_per_1k, output_per_1k) in USD.
_PRICING: dict[str, tuple[Decimal, Decimal]] = {
    # --- OpenAI ---
    "gpt-4o": (Decimal("0.0025"), Decimal("0.010")),
    "gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "gpt-4.1": (Decimal("0.002"), Decimal("0.008")),
    "gpt-4.1-mini": (Decimal("0.0004"), Decimal("0.0016")),
    "gpt-4.1-nano": (Decimal("0.0001"), Decimal("0.0004")),
    "gpt-4-turbo": (Decimal("0.010"), Decimal("0.030")),
    "gpt-4": (Decimal("0.030"), Decimal("0.060")),
    "gpt-3.5-turbo": (Decimal("0.0005"), Decimal("0.0015")),
    "o1": (Decimal("0.015"), Decimal("0.060")),
    "o1-mini": (Decimal("0.0011"), Decimal("0.0044")),
    "o3": (Decimal("0.010"), Decimal("0.040")),
    "o3-mini": (Decimal("0.0011"), Decimal("0.0044")),
    "o4-mini": (Decimal("0.0011"), Decimal("0.0044")),
    # --- Anthropic ---
    "claude-3-5-sonnet-latest": (Decimal("0.003"), Decimal("0.015")),
    "claude-3-5-sonnet-20241022": (Decimal("0.003"), Decimal("0.015")),
    "claude-3-5-haiku-latest": (Decimal("0.0008"), Decimal("0.004")),
    "claude-3-7-sonnet-latest": (Decimal("0.003"), Decimal("0.015")),
    "claude-sonnet-4": (Decimal("0.003"), Decimal("0.015")),
    "claude-opus-4": (Decimal("0.015"), Decimal("0.075")),
    "claude-3-opus-latest": (Decimal("0.015"), Decimal("0.075")),
    "claude-3-sonnet": (Decimal("0.003"), Decimal("0.015")),
    "claude-3-haiku": (Decimal("0.00025"), Decimal("0.00125")),
    # --- Google Gemini (per 1k; small-context tier) ---
    "gemini-1.5-flash": (Decimal("0.000075"), Decimal("0.0003")),
    "gemini-1.5-flash-8b": (Decimal("0.0000375"), Decimal("0.00015")),
    "gemini-1.5-pro": (Decimal("0.00125"), Decimal("0.005")),
    "gemini-2.0-flash": (Decimal("0.0001"), Decimal("0.0004")),
    "gemini-2.0-flash-lite": (Decimal("0.000075"), Decimal("0.0003")),
    "gemini-1.0-pro": (Decimal("0.0005"), Decimal("0.0015")),
}

_ZERO = Decimal("0")

# Trailing version/date suffixes to trim when resolving a model id, e.g.
# "gpt-4o-2024-08-06" -> "gpt-4o", "gemini-1.5-flash-002" -> "gemini-1.5-flash".
_DATE_SUFFIX = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{6,8}|\d{3})$")


def _normalise(model: str) -> str:
    name = model.strip()
    if name.startswith("azure/"):
        name = name[len("azure/") :]
    return name


def get_price(model: str) -> tuple[Decimal, Decimal] | None:
    """Resolve a model id to its (input, output) per-1k price, or None.

    Tries an exact match, then strips an ``azure/`` prefix, then progressively
    trims dated/version suffixes.
    """
    if model in _PRICING:
        return _PRICING[model]

    name = _normalise(model)
    if name in _PRICING:
        return _PRICING[name]

    # Progressively trim trailing version/date segments.
    prev = None
    while name != prev:
        prev = name
        trimmed = _DATE_SUFFIX.sub("", name)
        if trimmed in _PRICING:
            return _PRICING[trimmed]
        name = trimmed
    return None


def known_models() -> list[str]:
    """Sorted list of model ids with a configured price."""
    return sorted(_PRICING)


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    """Return the USD cost for a request, rounded to 6 decimal places."""
    price = get_price(model)
    if price is None:
        logger.warning("No pricing for model %r; recording cost as 0", model)
        return _ZERO

    input_per_1k, output_per_1k = price
    cost = (
        (Decimal(prompt_tokens) / Decimal(1000)) * input_per_1k
        + (Decimal(completion_tokens) / Decimal(1000)) * output_per_1k
    )
    return cost.quantize(Decimal("0.000001"))
