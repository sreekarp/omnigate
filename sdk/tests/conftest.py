"""Pytest configuration for the SDK test suite.

``asyncio_mode = auto`` is also set in pyproject; this file additionally makes
the suite runnable even if pytest-asyncio is configured via ini discovery from
a different rootdir.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Best-effort: ensure async tests run without an explicit marker.
    try:
        config.inicfg.setdefault("asyncio_mode", "auto")  # type: ignore[attr-defined]
    except Exception:
        pass
