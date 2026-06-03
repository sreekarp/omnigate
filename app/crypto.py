"""Symmetric encryption for provider keys stored at rest (BYOK vault).

Developers' OpenAI/Anthropic keys must never be stored in plaintext. We encrypt
them with Fernet (AES-128-CBC + HMAC), using a key derived from ``SECRET_KEY``.

NOTE: rotating SECRET_KEY invalidates all stored provider keys (they'd need to
be re-entered). For production, consider a dedicated, separately-managed key.
"""

import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings


@lru_cache
def _fernet() -> Fernet:
    # Fernet needs a 32-byte urlsafe-base64 key; derive one deterministically
    # from SECRET_KEY so the same secret always yields the same cipher.
    digest = hashlib.sha256(get_settings().secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret, returning a urlsafe token string for DB storage."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(token: str) -> str:
    """Decrypt a stored token back to plaintext.

    Raises ValueError if the token is invalid (e.g. SECRET_KEY changed).
    """
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:  # pragma: no cover - defensive
        raise ValueError("Could not decrypt stored secret") from exc
