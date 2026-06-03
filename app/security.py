"""API key generation, hashing, and constant-time comparison helpers.

Project API keys are never stored in plaintext. We store a SHA-256 hash and a
short non-secret prefix (for display in the dashboard). Lookups hash the
incoming key and match on the indexed ``key_hash`` column.
"""

import hashlib
import secrets

API_KEY_PREFIX = "llmg_"
_PREFIX_DISPLAY_LEN = 12  # how many leading chars we keep for display


def generate_api_key() -> str:
    """Generate a new project API key, e.g. ``llmg_AbC123...``."""
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Return the SHA-256 hex digest of an API key for storage/lookup."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def key_display_prefix(key: str) -> str:
    """Return a non-secret prefix of the key for UI display."""
    return key[:_PREFIX_DISPLAY_LEN]


def constant_time_compare(a: str, b: str) -> bool:
    """Timing-attack-resistant string comparison."""
    return secrets.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
