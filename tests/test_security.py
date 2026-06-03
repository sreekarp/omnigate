from app.security import (
    API_KEY_PREFIX,
    constant_time_compare,
    generate_api_key,
    hash_api_key,
    key_display_prefix,
)


def test_generated_key_has_prefix_and_is_unique():
    k1 = generate_api_key()
    k2 = generate_api_key()
    assert k1.startswith(API_KEY_PREFIX)
    assert k1 != k2


def test_hash_is_stable_and_64_hex():
    key = generate_api_key()
    h = hash_api_key(key)
    assert h == hash_api_key(key)
    assert len(h) == 64


def test_display_prefix_is_not_secret():
    key = generate_api_key()
    assert key_display_prefix(key) == key[:12]


def test_constant_time_compare():
    assert constant_time_compare("abc", "abc")
    assert not constant_time_compare("abc", "abd")
