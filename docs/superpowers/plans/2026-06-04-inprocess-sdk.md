# In-Process `omnigate` SDK — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, recommended for this tightly-coupled engine) or superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a litellm-style in-process engine to the `omnigate` package so `omnigate.completion(...)` / `acompletion(...)` call providers directly (routing, retry, fallback, circuit breaker, cost, opt-in cache, callbacks, spend cap) with no hosting — while keeping the hosted `Client`/`AsyncClient` unchanged.

**Architecture:** Providers are reduced to pure I/O-free **specs** (build payload / parse response / parse stream events). Two thin executors in `engine.py` (sync `httpx.Client`, async `httpx.AsyncClient`) perform the network call and feed the specs. An orchestrator runs `cache → breaker → retry → fallback → cost → callbacks`. Everything lives in `sdk/src/omnigate/`, importing nothing from `app/`. All new code is ported/decoupled from the proven `app/` logic.

**Tech Stack:** Python ≥3.10, `httpx`, `pydantic` v2, `pytest` + `httpx.MockTransport` (offline). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-04-omnigate-inprocess-sdk-design.md`

---

## File structure

New (all under `sdk/src/omnigate/`):

| File | Responsibility |
|---|---|
| `pricing.py` | per-model USD/1k pricing + `compute_cost` (port of `app/services/pricing.py`, pure) |
| `resilience.py` | `RetryPolicy`, `is_retryable`, `retry_async`, `retry_sync` |
| `circuit_breaker.py` | `InMemoryCircuitBreaker` (sync methods; port, Redis dropped) |
| `cache.py` | `ResponseCache` (TTL dict), `should_cache`, `cache_key` |
| `config.py` | `EngineConfig` dataclass + env loader + global getter/setter |
| `keys.py` | `Target`, `resolve_key`, `resolve_target`, programmatic key overrides |
| `callbacks.py` | `CallbackEvent`, `register`, `fire_success`, `fire_failure` |
| `providers/__init__.py` | package marker |
| `providers/base.py` | `ProviderSpec` ABC + `StreamState`; `Target` import |
| `providers/openai.py` | `OpenAISpec` |
| `providers/anthropic.py` | `AnthropicSpec` |
| `providers/gemini.py` | `GeminiSpec` |
| `providers/azure.py` | `AzureSpec` |
| `providers/registry.py` | `spec_for_model(model) -> (ProviderSpec, provider_name)` |
| `engine.py` | sync/async executors + `completion`/`acompletion`/`configure`/`register_callback` |

Modified:

| File | Change |
|---|---|
| `models.py` | extend `ChatRequest` with `top_p, stop, presence_penalty, frequency_penalty, seed, fallback_models, cache`; add `stop_sequences()` |
| `exceptions.py` | add optional `retry_after` to `APIError`; add `classify_http_error(...)` engine helper |
| `__init__.py` | export `completion, acompletion, configure, register_callback, EngineConfig` |
| `_version.py` | `0.1.0 → 0.2.0` |
| `pyproject.toml` | `version → 0.2.0`; fix `Homepage` URL |
| `README.md` (sdk) | lead with in-process usage |

Tests: `sdk/tests/test_engine.py` (new), plus a small addition to `sdk/tests/test_client.py` is NOT needed (kept untouched).

---

## Task 1: Extend `ChatRequest` + add `retry_after` to `APIError`

**Files:** Modify `sdk/src/omnigate/models.py`, `sdk/src/omnigate/exceptions.py`; Test `sdk/tests/test_engine.py`

- [ ] **Step 1: Failing test** — create `sdk/tests/test_engine.py`:

```python
from __future__ import annotations
import httpx, pytest
from omnigate.models import ChatRequest, Message

def test_chatrequest_has_engine_fields():
    r = ChatRequest(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="hi")],
        top_p=0.9, stop=["X"], presence_penalty=0.1, frequency_penalty=0.2,
        seed=7, fallback_models=["gpt-4o"], cache=True,
    )
    assert r.top_p == 0.9 and r.seed == 7
    assert r.stop_sequences() == ["X"]
    assert r.fallback_models == ["gpt-4o"] and r.cache is True

def test_apierror_has_retry_after():
    from omnigate.exceptions import APIError
    e = APIError("boom", status_code=503, retry_after=2.5)
    assert e.retry_after == 2.5
```

- [ ] **Step 2: Run, expect FAIL** — `python -m pytest sdk/tests/test_engine.py -q` → fails (unexpected kwargs).

- [ ] **Step 3: Implement** — in `models.py`, add to `ChatRequest` (after `stream`):

```python
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stop: "str | list[str] | None" = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    seed: Optional[int] = Field(default=None)
    fallback_models: list[str] = Field(default_factory=list, max_length=5)
    cache: Optional[bool] = Field(default=None)

    def stop_sequences(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)
```

In `exceptions.py`, change `APIError.__init__` to accept and store `retry_after`:

```python
    def __init__(self, message, *, status_code=None, detail=None,
                 request_id=None, retry_after=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id
        self.retry_after = retry_after
```

(Leave `RateLimitError.__init__` as-is; it sets `self.retry_after` after `super().__init__`, still correct.)

- [ ] **Step 4: Run, expect PASS** — `python -m pytest sdk/tests/test_engine.py -q` and `python -m pytest sdk/tests -q` (32 existing still pass).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(sdk): extend ChatRequest with sampling/fallback fields; APIError.retry_after"`

---

## Task 2: Port `pricing.py`

**Files:** Create `sdk/src/omnigate/pricing.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_pricing_known_and_normalised():
    from omnigate.pricing import compute_cost, get_price
    assert get_price("gpt-4o-mini") is not None
    assert get_price("azure/gpt-4o") == get_price("gpt-4o")
    assert get_price("gpt-4o-2024-08-06") == get_price("gpt-4o")
    c = compute_cost("gpt-4o-mini", 1000, 1000)
    assert float(c) > 0
    assert compute_cost("totally-unknown", 100, 100) == 0
```

- [ ] **Step 2: Run, expect FAIL** (no module).

- [ ] **Step 3: Implement** — copy `app/services/pricing.py` verbatim, with ONE change: replace `from app.logging_config import get_logger` / `logger = get_logger(__name__)` with:

```python
import logging
logger = logging.getLogger("omnigate")
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): port pricing table + compute_cost`

---

## Task 3: Port `resilience.py` (+ sync twin)

**Files:** Create `sdk/src/omnigate/resilience.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_retry_policy_delay_and_retryable():
    from omnigate.resilience import RetryPolicy, is_retryable
    from omnigate.exceptions import APIError, RateLimitError
    p = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=8.0, jitter=0.0)
    assert p.delay_for(0) == 1.0 and p.delay_for(1) == 2.0 and p.delay_for(10) == 8.0
    assert p.delay_for(0, retry_after=5.0) == 5.0
    assert is_retryable(APIError("x", status_code=503)) is True
    assert is_retryable(APIError("x", status_code=400)) is False
    assert is_retryable(RateLimitError("x", status_code=429)) is True

@pytest.mark.anyio_skip  # placeholder marker not used; see note
async def test_retry_async_then_success():
    from omnigate.resilience import RetryPolicy, retry_async
    from omnigate.exceptions import ProviderError
    calls = {"n": 0}
    async def thunk():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("boom", status_code=503)
        return "ok"
    p = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0)
    out = await retry_async(thunk, policy=p, sleep=lambda s: _noop())
    assert out == "ok" and calls["n"] == 3

async def _noop():
    return None

def test_retry_sync_then_success():
    from omnigate.resilience import RetryPolicy, retry_sync
    from omnigate.exceptions import ProviderError
    calls = {"n": 0}
    def thunk():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ProviderError("boom", status_code=500)
        return "ok"
    p = RetryPolicy(max_attempts=3, base_delay=0.0, max_delay=0.0, jitter=0.0)
    out = retry_sync(thunk, policy=p, sleep=lambda s: None)
    assert out == "ok" and calls["n"] == 2
```

> Note: `asyncio_mode = "auto"` (sdk pyproject) means `async def test_*` run without decorators. Remove the bogus `@pytest.mark.anyio_skip` line — it was illustrative; do not include it.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — port `app/utils/resilience.py` with these changes:
  - Drop `from app.providers.base import ProviderError` and the `Settings` typing import.
  - `is_retryable(exc)` checks the SDK exception:

```python
import asyncio, logging, random, time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar
import httpx
from .exceptions import APIError

logger = logging.getLogger("omnigate")
T = TypeVar("T")
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APIError):
        return exc.status_code in _RETRYABLE_STATUS
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False
```

  - Keep `RetryPolicy` (fields `max_attempts, base_delay, max_delay, jitter`) and `delay_for` verbatim; add:

```python
    @classmethod
    def from_config(cls, cfg) -> "RetryPolicy":
        return cls(cfg.retry_max_attempts, cfg.retry_base_delay,
                   cfg.retry_max_delay, cfg.retry_jitter)
```

  - Keep `retry_async` verbatim (rename type alias imports as needed). Add a sync twin `retry_sync` that mirrors it using `time.sleep` and a sync thunk:

```python
def retry_sync(func, *, policy, on_retry=None, sleep=time.sleep):
    attempts = max(1, policy.max_attempts)
    last_exc = None
    for attempt in range(attempts):
        try:
            return func()
        except BaseException as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            delay = policy.delay_for(attempt, retry_after=getattr(exc, "retry_after", None))
            if on_retry is not None:
                try: on_retry(attempt + 1, exc)
                except Exception: logger.warning("on_retry raised", exc_info=True)
            sleep(delay)
    assert last_exc is not None
    raise last_exc
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): port retry policy (async + sync twin)`

---

## Task 4: Port `circuit_breaker.py` (in-memory, sync methods)

**Files:** Create `sdk/src/omnigate/circuit_breaker.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_breaker_opens_and_recovers():
    from omnigate.circuit_breaker import InMemoryCircuitBreaker
    clock = {"t": 0.0}
    cb = InMemoryCircuitBreaker(fail_threshold=2, cooldown_seconds=10.0, now=lambda: clock["t"])
    assert cb.allow("openai") is True
    cb.record_failure("openai"); cb.record_failure("openai")
    assert cb.allow("openai") is False           # open
    clock["t"] = 11.0
    assert cb.allow("openai") is True            # half-open trial admitted
    cb.record_success("openai")
    assert cb.allow("openai") is True            # closed
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — port ONLY `_Circuit`, `CircuitState`, `circuit_key`, `CircuitOpenError`, and `InMemoryCircuitBreaker` from `app/services/circuit_breaker.py`, converting the three `async def allow/record_success/record_failure` to plain `def` (delete `async`; bodies identical). Drop the Redis backend, the `Protocol`, and the `get_circuit_breaker` factory. Replace logging import with `import logging; logger = logging.getLogger("omnigate")`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): port in-memory circuit breaker`

---

## Task 5: `cache.py` — in-memory TTL response cache

**Files:** Create `sdk/src/omnigate/cache.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_cache_eligibility_and_roundtrip():
    from omnigate.cache import ResponseCache, should_cache, cache_key
    from omnigate.models import ChatRequest, Message, ChatResponse, Usage
    req = ChatRequest(model="gpt-4o-mini", messages=[Message(role="user", content="hi")], temperature=0)
    assert should_cache(req, enabled=True) is True
    assert should_cache(req, enabled=False) is False
    stream_req = req.model_copy(update={"stream": True})
    assert should_cache(stream_req, enabled=True) is False
    nondet = req.model_copy(update={"temperature": 0.7})
    assert should_cache(nondet, enabled=True) is False
    c = ResponseCache(now=lambda: 0.0)
    k = cache_key("openai", req)
    assert c.get(k) is None
    resp = ChatResponse(id="1", provider="openai", model="gpt-4o-mini",
                        content="x", usage=Usage())
    c.set(k, resp, ttl=300)
    hit = c.get(k)
    assert hit is not None and hit.cached is True
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement:**

```python
"""In-memory TTL response cache for deterministic, non-streaming completions."""
from __future__ import annotations
import hashlib, json, logging, time
from typing import Callable, Optional
from .models import ChatRequest, ChatResponse

logger = logging.getLogger("omnigate")

def should_cache(request: ChatRequest, *, enabled: bool) -> bool:
    if request.stream:
        return False
    if request.cache is False:
        return False
    if request.temperature != 0:
        return False
    return bool(enabled or request.cache is True)

def cache_key(provider: str, request: ChatRequest) -> str:
    payload = {
        "provider": provider,
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
        "stop": request.stop_sequences(),
        "seed": request.seed,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

class ResponseCache:
    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._store: dict[str, tuple[float, str]] = {}
        self._now = now

    def get(self, key: str) -> Optional[ChatResponse]:
        try:
            item = self._store.get(key)
            if item is None:
                return None
            expiry, raw = item
            if expiry <= self._now():
                self._store.pop(key, None)
                return None
            resp = ChatResponse.model_validate_json(raw)
            resp.cached = True
            return resp
        except Exception:
            logger.warning("response cache get failed", exc_info=True)
            return None

    def set(self, key: str, response: ChatResponse, ttl: int) -> None:
        if ttl <= 0:
            return
        try:
            self._store[key] = (self._now() + ttl, response.model_dump_json())
        except Exception:
            logger.warning("response cache set failed", exc_info=True)

    def clear(self) -> None:
        self._store.clear()

_cache = ResponseCache()
def get_cache() -> ResponseCache:
    return _cache
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): in-memory TTL response cache`

---

## Task 6: `config.py` — `EngineConfig`

**Files:** Create `sdk/src/omnigate/config.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_engine_config_env(monkeypatch):
    from omnigate.config import EngineConfig
    monkeypatch.setenv("OMNIGATE_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OMNIGATE_CACHE_ENABLED", "true")
    monkeypatch.setenv("OMNIGATE_MAX_SPEND_USD", "1.50")
    cfg = EngineConfig.from_env()
    assert cfg.timeout == 12.5 and cfg.cache_enabled is True and cfg.max_spend_usd == 1.5
    default = EngineConfig()
    assert default.timeout == 60.0 and default.cache_enabled is False and default.max_spend_usd is None
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — `EngineConfig` dataclass with the fields from spec §5 and a `from_env()` classmethod using small `_env_float/_env_int/_env_bool(name, default)` helpers (`_env_bool` truthy set = `{"1","true","yes","on"}`). Provide module globals:

```python
_config: EngineConfig | None = None
def get_config() -> EngineConfig:
    global _config
    if _config is None:
        _config = EngineConfig.from_env()
    return _config
def set_config(cfg: EngineConfig) -> None:
    global _config
    _config = cfg
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): EngineConfig (plain dataclass, env-driven)`

---

## Task 7: `keys.py` — key + Azure target resolution

**Files:** Create `sdk/src/omnigate/keys.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_resolve_key_precedence(monkeypatch):
    from omnigate import keys
    keys.reset()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from omnigate.exceptions import APIError
    with pytest.raises(APIError):
        keys.resolve_key("openai", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert keys.resolve_key("openai", None) == "sk-env"
    assert keys.resolve_key("openai", "sk-explicit") == "sk-explicit"
    keys.set_override("openai", "sk-override")
    assert keys.resolve_key("openai", None) == "sk-override"
    assert keys.resolve_key("openai", "sk-explicit") == "sk-explicit"  # explicit still wins

def test_resolve_target_azure(monkeypatch):
    from omnigate import keys
    keys.reset()
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    t = keys.resolve_target("azure", "azure/my-deploy", api_base=None, api_version=None)
    assert t.endpoint == "https://x.openai.azure.com" and t.deployment == "my-deploy"
    assert t.api_version  # defaulted
    assert keys.resolve_target("openai", "gpt-4o", api_base=None, api_version=None) is None
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — `Target` dataclass (`endpoint`, `deployment`, `api_version`) lives in `providers/base.py` (Task 8); to avoid ordering issues, define `Target` in `keys.py` and re-export from `base.py`. Implement:
  - `_ENV = {"openai":["OPENAI_API_KEY"], "anthropic":["ANTHROPIC_API_KEY"], "gemini":["GEMINI_API_KEY","GOOGLE_API_KEY"], "azure":["AZURE_OPENAI_API_KEY"]}`
  - `_overrides: dict[str,str] = {}`, `_azure: dict[str,str] = {}` (endpoint/api_version), `set_override(provider,key)`, `set_azure(endpoint=None, api_version=None)`, `reset()`.
  - `resolve_key(provider, explicit)` → explicit or override or first present env var; else `raise APIError(f"No API key for {provider}. Set {env[0]} or pass api_key=.", status_code=400)`.
  - `resolve_target(provider, model, *, api_base, api_version)`:
    - non-azure → `None`
    - azure: endpoint = `api_base` or `_azure["endpoint"]` or `AZURE_OPENAI_ENDPOINT`; missing → `APIError`. deployment = `model.split("/",1)[1]` if `model` lower-startswith `azure/` else from env? (require `azure/<deployment>`); missing → `APIError`. version = `api_version` or `_azure["api_version"]` or `AZURE_OPENAI_API_VERSION` or `"2024-10-21"`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): key + Azure target resolution`

---

## Task 8: `providers/base.py` + `callbacks.py`

**Files:** Create `sdk/src/omnigate/providers/__init__.py`, `providers/base.py`, `callbacks.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_callbacks_fire_and_swallow():
    from omnigate import callbacks
    callbacks.reset()
    seen = []
    callbacks.register(on_success=lambda e: seen.append(("ok", e.model)),
                       on_failure=lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    from omnigate.callbacks import CallbackEvent
    callbacks.fire_success(CallbackEvent(model="m", provider="openai", usage=None,
                                         cost_usd=0.1, latency_ms=5, cached=False, fallback_used=False))
    callbacks.fire_failure(CallbackEvent(model="m", provider="openai", usage=None,
                                         cost_usd=0.0, latency_ms=1, cached=False,
                                         fallback_used=False, exception=ValueError("x")))
    assert seen == [("ok", "m")]   # failure callback raised but was swallowed
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement:**
  - `providers/__init__.py`: empty.
  - `providers/base.py`:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from ..keys import Target  # re-export
from ..models import ChatRequest, ChatResponse, StreamChunk

class StreamState:
    """Mutable per-stream accumulator (provider-defined contents)."""
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

class ProviderSpec(ABC):
    name: str
    @abstractmethod
    def url(self, request: ChatRequest, target: Target | None) -> str: ...
    @abstractmethod
    def headers(self, api_key: str) -> dict[str, str]: ...
    @abstractmethod
    def build_payload(self, request: ChatRequest, *, stream: bool) -> dict: ...
    @abstractmethod
    def parse_response(self, data: dict, request_model: str) -> ChatResponse: ...
    # streaming protocol
    def stream_begin(self) -> StreamState:
        return StreamState()
    @abstractmethod
    def stream_feed(self, state: StreamState, raw: dict) -> list[StreamChunk]: ...
    def stream_end(self, state: StreamState) -> StreamChunk | None:
        return None
```

  - `callbacks.py`: `@dataclass CallbackEvent(model, provider, usage, cost_usd, latency_ms, cached, fallback_used, exception=None)`; module lists `_success`, `_failure`; `register(*, on_success=None, on_failure=None)`; `fire_success(event)`/`fire_failure(event)` iterate and wrap each call in `try/except` logging at WARNING; `reset()`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): ProviderSpec base + callbacks registry`

---

## Task 9: `providers/openai.py` + `providers/azure.py` + `registry.py`

**Files:** Create `providers/openai.py`, `providers/azure.py`, `providers/registry.py`; Test add to `test_engine.py`

- [ ] **Step 1: Failing test:**

```python
def test_openai_spec_payload_and_parse():
    from omnigate.providers.openai import OpenAISpec
    from omnigate.models import ChatRequest, Message
    s = OpenAISpec()
    req = ChatRequest(model="gpt-4o-mini", messages=[Message(role="user", content="hi")],
                      max_tokens=10, temperature=0.5)
    p = s.build_payload(req, stream=False)
    assert p["model"] == "gpt-4o-mini" and p["messages"] == [{"role": "user", "content": "hi"}]
    assert p["max_tokens"] == 10
    data = {"id": "x", "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "yo"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
    resp = s.parse_response(data, "gpt-4o-mini")
    assert resp.content == "yo" and resp.usage.total_tokens == 3 and resp.provider == "openai"

def test_registry_routing():
    from omnigate.providers.registry import spec_for_model
    assert spec_for_model("gpt-4o-mini")[1] == "openai"
    assert spec_for_model("claude-3-5-haiku-latest")[1] == "anthropic"
    assert spec_for_model("gemini-1.5-flash")[1] == "gemini"
    assert spec_for_model("azure/my-deploy")[1] == "azure"
    from omnigate.exceptions import APIError
    with pytest.raises(APIError):
        spec_for_model("mystery-model")
```

> `registry.spec_for_model` references all four specs (Task 10 adds anthropic/gemini); write the registry now importing them lazily so this task's openai/azure assertions pass once Task 10 lands. To keep Task 9 self-contained, the routing test for anthropic/gemini is added in Task 10. Keep only `gpt`/`azure`/`mystery` assertions here; move claude/gemini asserts to Task 10.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement:**
  - `providers/openai.py`: port the module helpers `build_chat_payload`, `parse_chat_response`, `parse_stream_chunk`, `_parse_retry_after` from `app/providers/openai.py` (drop `app` imports; `from ..models import ...`). Wrap them in `OpenAISpec(ProviderSpec)`:
    - `name = "openai"`, `url(...) -> "https://api.openai.com/v1/chat/completions"`, `headers(api_key)` → Bearer (raise `APIError(...,400)` if empty).
    - `build_payload` → `build_chat_payload`.
    - `parse_response(data, model)` → `parse_chat_response(data, provider_name=self.name, request_model=model)`.
    - `stream_feed(state, raw)` → `parse_stream_chunk(raw)` (returns content and the include_usage terminal chunk). `stream_end` → `None`.
  - `providers/azure.py`: `AzureSpec(ProviderSpec)` `name="azure"`, reuses `build_chat_payload`/`parse_chat_response`/`parse_stream_chunk` from `.openai`; `url(request, target)` builds `{target.endpoint}/openai/deployments/{quote(target.deployment)}/chat/completions?api-version={target.api_version}`; `headers` → `{"api-key": key}`; streaming same as OpenAI.
  - `providers/registry.py`: `spec_for_model(model)` mirrors `provider_name_for_model` prefixes; returns `(spec_instance, name)`; caches single instances; `azure/` → `AzureSpec`; unknown → `APIError(..., 400)`.

- [ ] **Step 4: Run, expect PASS** (openai/azure/mystery asserts).

- [ ] **Step 5: Commit** — `feat(sdk): OpenAI + Azure provider specs + registry`

---

## Task 10: `providers/anthropic.py` + `providers/gemini.py`

**Files:** Create `providers/anthropic.py`, `providers/gemini.py`; extend registry test

- [ ] **Step 1: Failing tests:**

```python
def test_anthropic_spec_stream_accumulates_usage():
    from omnigate.providers.anthropic import AnthropicSpec
    s = AnthropicSpec(); st = s.stream_begin()
    out = []
    out += s.stream_feed(st, {"type": "message_start", "message": {"usage": {"input_tokens": 5}}})
    out += s.stream_feed(st, {"type": "content_block_delta",
                              "delta": {"type": "text_delta", "text": "hi"}})
    out += s.stream_feed(st, {"type": "message_delta", "usage": {"output_tokens": 7},
                              "delta": {"stop_reason": "end_turn"}})
    term = s.stream_end(st)
    assert any(c.text == "hi" for c in out)
    assert term.usage.prompt_tokens == 5 and term.usage.completion_tokens == 7
    assert term.finish_reason == "end_turn"

def test_gemini_spec_parse_and_routing():
    from omnigate.providers.gemini import GeminiSpec
    s = GeminiSpec()
    data = {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5}}
    resp = s.parse_response(data, "gemini-1.5-flash")
    assert resp.content == "ok" and resp.usage.total_tokens == 5
    from omnigate.providers.registry import spec_for_model
    assert spec_for_model("claude-3-5-haiku-latest")[1] == "anthropic"
    assert spec_for_model("gemini-1.5-flash")[1] == "gemini"
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement:**
  - `providers/anthropic.py`: `AnthropicSpec` — port payload building / response parsing / `_parse_retry_after` from `app/providers/anthropic.py`. `url` → `https://api.anthropic.com/v1/messages`; `headers` → `{"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}`. Streaming uses `StreamState`:
    - `stream_feed`: on `message_start` set `state.data["in"]`; on `content_block_delta` text → return `[StreamChunk(text=...)]`; on `message_delta` update `state.data["out"]` and `state.data["finish"]`; on `error` raise `ProviderError(..., status_code=502)`; else `[]`.
    - `stream_end`: return `StreamChunk(usage=Usage(prompt=in, completion=out, total=in+out), finish_reason=finish)`.
  - `providers/gemini.py`: `GeminiSpec` — port `_normalise_model`, `_extract_text`, `_usage_from_metadata`, payload building, response parsing from `app/providers/gemini.py`. `url(request, _)` → `{BASE}/models/{quote(model)}:generateContent` (non-stream) / `:streamGenerateContent?alt=sse` (stream); `headers` → `{"x-goog-api-key": key, ...}`. Streaming: `stream_feed` returns content chunks and stores last-wins `usageMetadata` + finishReason in `state.data`; `stream_end` returns the terminal usage chunk.
  - Extend `registry.spec_for_model` already covers claude/gemini prefixes (added in Task 9).

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `feat(sdk): Anthropic + Gemini provider specs`

---

## Task 11: `engine.py` — executors + orchestrator

**Files:** Create `sdk/src/omnigate/engine.py`; Test add to `test_engine.py`

The engine exposes `acompletion`/`completion` and the orchestration. Use injectable `transport` so tests stay offline.

- [ ] **Step 1: Failing tests** (offline via `MockTransport`; one transport routes every provider URL):

```python
import json, httpx, pytest
from omnigate import keys as _keys
from omnigate.config import EngineConfig, set_config
from omnigate.cache import get_cache
from omnigate import callbacks as _cb

OPENAI_OK = {"id": "c1", "model": "gpt-4o-mini",
             "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}

@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    _keys.reset(); _cb.reset(); get_cache().clear()
    set_config(EngineConfig(retry_base_delay=0.0, retry_max_delay=0.0, retry_jitter=0.0,
                            circuit_breaker_cooldown=0.01))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    yield
    set_config(EngineConfig())

def _transport(handler):
    return httpx.MockTransport(handler)

def test_completion_happy_path_sync():
    import omnigate
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json=OPENAI_OK)
    r = omnigate.completion(model="gpt-4o-mini", messages="hi", transport=_transport(handler))
    assert r.content == "hello" and r.usage.total_tokens == 5
    assert r.provider == "openai" and r.cost_usd > 0 and r.latency_ms >= 0

async def test_acompletion_happy_path():
    import omnigate
    def handler(req): return httpx.Response(200, json=OPENAI_OK)
    r = await omnigate.acompletion(model="gpt-4o-mini", messages="hi",
                                   transport=_transport(handler))
    assert r.content == "hello"

def test_completion_retries_then_succeeds():
    import omnigate
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        if n["i"] < 2:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json=OPENAI_OK)
    r = omnigate.completion(model="gpt-4o-mini", messages="hi", transport=_transport(handler))
    assert r.content == "hello" and n["i"] == 2

def test_completion_fallback_used():
    import omnigate
    def handler(req):
        if "anthropic.com" in str(req.url):
            return httpx.Response(200, json={
                "id": "a1", "model": "claude-3-5-haiku-latest",
                "content": [{"type": "text", "text": "fallback!"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}, "stop_reason": "end_turn"})
        return httpx.Response(500, json={"error": "down"})
    r = omnigate.completion(model="gpt-4o-mini", messages="hi",
                            fallbacks=["claude-3-5-haiku-latest"],
                            transport=_transport(handler))
    assert r.content == "fallback!" and r.fallback_used is True and r.provider == "anthropic"

def test_completion_auth_error_not_retried():
    import omnigate
    from omnigate import AuthError
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        return httpx.Response(401, json={"error": "bad key"})
    with pytest.raises(AuthError):
        omnigate.completion(model="gpt-4o-mini", messages="hi", transport=_transport(handler))
    assert n["i"] == 1

def test_completion_cache_hit():
    import omnigate
    n = {"i": 0}
    def handler(req):
        n["i"] += 1
        return httpx.Response(200, json=OPENAI_OK)
    kw = dict(model="gpt-4o-mini", messages="hi", temperature=0, cache=True,
              transport=_transport(handler))
    a = omnigate.completion(**kw); b = omnigate.completion(**kw)
    assert n["i"] == 1 and b.cached is True and b.cost_usd == 0.0

def test_spend_cap_raises():
    import omnigate
    from omnigate import BudgetExceededError
    set_config(EngineConfig(max_spend_usd=0.0))
    def handler(req): return httpx.Response(200, json=OPENAI_OK)
    with pytest.raises(BudgetExceededError):
        omnigate.completion(model="gpt-4o-mini", messages="hi", transport=_transport(handler))

def test_callbacks_fire_on_success():
    import omnigate
    events = []
    omnigate.register_callback(on_success=lambda e: events.append(e))
    def handler(req): return httpx.Response(200, json=OPENAI_OK)
    omnigate.completion(model="gpt-4o-mini", messages="hi", transport=_transport(handler))
    assert events and events[0].provider == "openai" and events[0].cost_usd > 0

def test_streaming_sync_reassembles_and_usage():
    import omnigate
    sse = ("data: " + json.dumps({"choices": [{"delta": {"content": "Hel"}}]}) + "\n\n"
           "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}}]}) + "\n\n"
           "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 1,
                                  "completion_tokens": 2, "total_tokens": 3}, "model": "gpt-4o-mini"}) + "\n\n"
           "data: [DONE]\n\n")
    def handler(req):
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    chunks = list(omnigate.completion(model="gpt-4o-mini", messages="hi", stream=True,
                                      transport=_transport(handler)))
    assert "".join(c.text for c in chunks) == "Hello"
    assert any(c.usage and c.usage.total_tokens == 3 for c in chunks)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `engine.py`:**
  - Helpers: `_build_request(model, messages, **params) -> ChatRequest` (uses `coerce_messages`); `_candidates(req) -> list[str]` (primary + fallbacks, deduped, cap 6); `_sse_data(line) -> str | None`.
  - `_resolve(model, *, api_key, api_base, api_version) -> (spec, provider_name, key, target)` using `registry.spec_for_model`, `keys.resolve_key`, `keys.resolve_target`.
  - `_classify(provider, status, text, retry_after)` → typed exc (401/403→AuthError; 429→RateLimitError; else→ProviderError); see `exceptions.classify_http_error` (add it in this task or reuse Task 1; define here for clarity).
  - **Async non-stream** `_acall(client, spec, req, key, target)`: POST `spec.url(...)`, `headers`, `json=spec.build_payload(req, stream=False)`; on `status>=400` raise `_classify`; else `spec.parse_response(resp.json(), req.model)`. Wrap `httpx` errors → `ProviderError(502)`.
  - **Sync non-stream** `_call(...)`: same with `httpx.Client`.
  - **Async stream** `_astream(client, spec, req, key, target)`: `client.stream("POST", ...)`; on error raise; `state = spec.stream_begin()`; iterate `aiter_lines()`, `_sse_data`, `[DONE]`→break, json.loads → `spec.stream_feed` (yield each); after loop `spec.stream_end` (yield if not None).
  - **Sync stream** `_stream(...)`: same with `Client.stream` + `iter_lines()`.
  - **Orchestrator** (async) `_arun(req, *, config, transport, api_key, api_base, api_version)`:
    - cache lookup if `should_cache(req, enabled=config.cache_enabled)`; key uses primary provider name; hit → return.
    - breaker = module global `InMemoryCircuitBreaker` (lazily built from config); `_cb_reset_if_config_changed`.
    - for model in candidates: resolve (skip→last_error on `APIError`); if breaker enabled and not `allow(provider)` → record last_error, continue; try `retry_async(lambda: _acall(...), policy)`. On retryable exc → `breaker.record_failure`; on non-retryable → `breaker.record_success`; continue with last_error. On success: `record_success`; `cost = compute_cost(...)`; set `cost_usd`, `fallback_used`, `latency_ms`; spend-cap add (raise `BudgetExceededError`); cache set if eligible & not fallback; `callbacks.fire_success`; return.
    - exhausted → `callbacks.fire_failure`; raise last_error or `ProviderError("no candidate", 502)`.
    - **Spend-cap & BudgetExceededError propagate** (not caught by the candidate loop).
  - Sync orchestrator `_run(...)` mirrors with `_call`/`retry_sync`.
  - Streaming path: `completion(stream=True)` → returns the sync generator `_run_stream(...)` (primary only, resolve once, no retry); `acompletion(stream=True)` → returns the async generator.
  - **Public**:

```python
def completion(*, model, messages, stream=False, transport=None, api_key=None,
               api_base=None, api_version=None, timeout=None, num_retries=None,
               **params):
    req = _build_request(model, messages, stream=stream, **params)
    cfg = _effective_config(timeout, num_retries)
    if stream:
        return _run_stream(req, config=cfg, transport=transport, api_key=api_key,
                           api_base=api_base, api_version=api_version)
    return _run(req, config=cfg, transport=transport, api_key=api_key,
                api_base=api_base, api_version=api_version)

async def acompletion(*, ...same...):
    ...
    if stream:
        return _arun_stream(...)  # returns async generator
    return await _arun(...)
```

  - `configure(**kwargs)`: pop key kwargs (`openai_api_key`→`keys.set_override("openai",...)`, etc.; `azure_endpoint`/`azure_api_version`→`keys.set_azure(...)`), apply remaining to a copied `EngineConfig` via `set_config`. Unknown kwarg → `ValueError`.
  - `register_callback(*, on_success=None, on_failure=None)` → `callbacks.register(...)`.
  - `_effective_config(timeout, num_retries)`: copy `get_config()`, override `timeout`/`retry_max_attempts` when provided.

- [ ] **Step 4: Run, expect PASS** — `python -m pytest sdk/tests/test_engine.py -q`.

- [ ] **Step 5: Commit** — `feat(sdk): in-process engine (sync/async/stream orchestrator)`

---

## Task 12: Public API exports + version bump

**Files:** Modify `__init__.py`, `_version.py`, `pyproject.toml`

- [ ] **Step 1: Failing test:**

```python
def test_top_level_exports():
    import omnigate
    for name in ("completion", "acompletion", "configure", "register_callback", "EngineConfig"):
        assert hasattr(omnigate, name)
    assert omnigate.__version__ == "0.2.0"
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — in `__init__.py` add imports + `__all__` entries for `completion, acompletion, configure, register_callback` (from `.engine`) and `EngineConfig` (from `.config`). Set `_version.py` `__version__ = "0.2.0"`. In `sdk/pyproject.toml` set `version = "0.2.0"` and `Homepage = "https://github.com/sreekarp/omnigate"`.

- [ ] **Step 4: Run, expect PASS** — full `python -m pytest sdk/tests -q` (all green).

- [ ] **Step 5: Commit** — `feat(sdk): export in-process API; bump 0.2.0`

---

## Task 13: Docs

**Files:** Modify `sdk/README.md`, root `README.md`, `CLAUDE.md`

- [ ] **Step 1** — Rewrite `sdk/README.md`: install → in-process quick start (`completion`) → async → streaming → fallbacks → cost/usage → callbacks → spend cap → config & keys (env table) → then a "Talk to a hosted gateway" section retaining the `Client`/`AsyncClient` docs → errors table → license.
- [ ] **Step 2** — Add a short "SDK has two modes (in-process + hosted client)" note to root `README.md` and the `sdk/` line in `CLAUDE.md`.
- [ ] **Step 3: Commit** — `docs: document in-process SDK usage`

---

## Task 14: Full verification + push

- [ ] **Step 1** — `python -m pytest sdk/tests -q` → all pass (existing 32 + new).
- [ ] **Step 2** — `python -m pytest -q` (server suite, offline) → still green (no `app/` changes, sanity only).
- [ ] **Step 3** — `python -m build ./sdk --outdir /tmp/omnigate-dist` (or `dist/`) → builds `omnigate-0.2.0` wheel+sdist cleanly. (Confirms CI `publish.yml` will succeed.)
- [ ] **Step 4** — `git push -u origin feat/inprocess-sdk`.
- [ ] **Step 5** — Report to user: branch pushed; to publish `omnigate 0.2.0`, merge to `main` and create a GitHub release (CI `publish.yml` builds `./sdk` and Trusted-Publishes).

---

## Self-review

- **Spec coverage:** §1 API → T11/T12; §3.1 specs → T8–T10; §3.2 orchestrator → T11; §3.3 errors → T1 (+`classify` in T11); §4 ChatRequest fields → T1; §5 config/keys → T6/T7; §6 cache/callbacks/spend-cap → T5/T8/T11; §7 backward-compat → all tasks keep existing tests green (T1/T12/T14); §8 tests → T1–T12; §9 docs/packaging → T12/T13/T14. Router explicitly deferred (no task) per approval. ✓
- **Placeholder scan:** the illustrative `@pytest.mark.anyio_skip` line in Task 3 is flagged for removal in-place; no other placeholders. ✓
- **Type consistency:** `RetryPolicy(max_attempts, base_delay, max_delay, jitter)` consistent T3/T11; `ResponseCache.get/set/clear` consistent T5/T11; `spec_for_model -> (spec, name)` consistent T9/T11; `CallbackEvent` field names consistent T8/T11; `Target(endpoint, deployment, api_version)` consistent T7/T9/T10. ✓
