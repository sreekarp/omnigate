# Design: `omnigate` as a standalone, litellm-style in-process SDK

- **Date:** 2026-06-04
- **Status:** Approved (implementation pending)
- **Package affected:** `omnigate` (the published PyPI SDK, `sdk/`), bumped `0.1.0 → 0.2.0`
- **Untouched:** `omnigate-gateway` (the FastAPI server, `app/`)

## 1. Goal

Today the published `omnigate` package is **only** an HTTP client (`Client` /
`AsyncClient`) that talks to a *running* OmniGate server. You cannot use any of
the gateway's value (multi-provider routing, retry, fallback, circuit breaking,
cost tracking) without hosting the server.

Make `omnigate` work like litellm: `pip install omnigate`, then call providers
**directly in-process** with zero hosting, no Postgres, no Redis.

```python
import omnigate

# key resolved from env (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY /
# AZURE_OPENAI_*) or passed explicitly as api_key=
r = omnigate.completion(model="gpt-4o-mini", messages="Hello!", temperature=0)
print(r.content, r.usage.total_tokens, r.cost_usd, r.model)

# async
r = await omnigate.acompletion(
    model="claude-3-5-haiku-latest",
    messages=[{"role": "user", "content": "hi"}],
    fallbacks=["gpt-4o-mini"],
)

# streaming (sync iterator of StreamChunk; async mirror via acompletion)
for chunk in omnigate.completion(model="gpt-4o-mini", messages="haiku", stream=True):
    print(chunk.text, end="")
```

The hosted `Client` / `AsyncClient` and every existing model/exception/helper
stay **exactly as they are** — this change is purely additive and backward
compatible.

## 2. Non-goals (require hosting; stay in `app/`)

Gateway API-key auth, cross-client rate limiting, per-org/project budgets, the
usage Postgres database, Prometheus exposition, and the dashboard remain in the
server package. Their hosting-free analogs in the SDK are the **local spend cap**
and **callbacks** (see §6).

## 3. Architecture

The engine lives **inside the `omnigate` package** and imports nothing from
`app/` (this matches the SDK's established "standalone — mirrors the wire schema,
imports nothing from the server" philosophy). It is a careful **port + decouple**
of the proven `app/` logic, not a rewrite.

```
sdk/src/omnigate/
  # --- new in-process engine ---
  engine.py            # orchestrator: cache → breaker → retry → fallback → cost → callbacks
  config.py            # EngineConfig dataclass (plain env reads; NO pydantic-settings, NO DB/Redis)
  keys.py              # resolve_key/resolve_target: explicit args > env vars
  pricing.py           # ported from app/services/pricing.py (pure; logger only)
  resilience.py        # ported RetryPolicy + retry_async (from_settings → from_config)
  circuit_breaker.py   # ported InMemoryCircuitBreaker only (Redis backend dropped)
  cache.py             # in-memory TTL response cache (port of the Redis cache's eligibility + keying)
  callbacks.py         # success/failure hook registry
  providers/
    base.py            # ProviderSpec: pure build (url/headers/payload) + parse (response/stream-line). NO I/O.
    openai.py          # ported, reduced to a spec (reuses build_chat_payload/parse_* helpers)
    anthropic.py       # ported, reduced to a spec
    gemini.py          # ported, reduced to a spec
    azure.py           # ported, reduced to a spec (per-target: endpoint/deployment/api-version)
    registry.py        # model-prefix → provider spec routing
  # --- unchanged (HTTP client + shared types) ---
  client.py async_client.py _transport.py _retry.py
  models.py exceptions.py _version.py __init__.py py.typed
```

### 3.1 Pure provider specs + thin sync/async executors

The single most important design choice. Today's `app/providers/*` adapters each
own their I/O (`httpx.AsyncClient`) and are async-only. To get **sync + async +
streaming for all four providers without quadrupling code**, each provider is
reduced to a *pure, I/O-free descriptor* — exactly the way today's
`_transport.py` is "transport-agnostic helpers" while `Client`/`AsyncClient`
supply the actual I/O.

`ProviderSpec` (per provider) exposes only pure functions/values:

- `name: str`
- `endpoint(request, target) -> str` — the URL (Azure uses target endpoint/deployment/version)
- `headers(api_key) -> dict[str, str]`
- `build_payload(request, *, stream: bool) -> dict`
- `parse_response(data: dict, request_model: str) -> ChatResponse`
- `parse_stream_line(line: str) -> list[StreamChunk]` — translate one decoded SSE/line into 0+ chunks; the engine handles `data:`/`[DONE]` framing and the single terminal usage chunk per the existing contract
- `stream_terminal(state) -> StreamChunk | None` — for providers (Anthropic, Gemini) that accumulate usage across events and emit exactly one terminal usage chunk at the end

`engine.py` holds the two executors that perform the network call and feed these
parsers:

- **sync:** `httpx.Client(timeout=...)` → POST (or `stream=`) → parse
- **async:** `httpx.AsyncClient(timeout=...)` → POST (or `stream=`) → parse

Both accept an optional injected `transport=` / `httpx` client (from
`EngineConfig` or per-call) so every path is testable offline with
`httpx.MockTransport`, and users can supply custom transports (proxies, etc.).

### 3.2 Orchestrator flow (mirrors `app/services/routing.py`)

`acompletion` is the canonical async implementation; `completion` is its sync
twin (separate sync executor — **no** async-in-sync bridging, which keeps
streaming clean). Both run:

```
candidate_models = [model, *fallbacks]   # de-duped, capped at 6

response-cache lookup (if eligible)         → hit ⇒ return (cost 0, cached=True)
for model in candidate_models:
    spec, api_key, target = resolve(model)  # routing + key/target resolution
    if breaker.open(provider): continue     # skip, record last_error
    try:
        result = retry(lambda: call(spec, request, api_key, target))
    except RateLimit/5xx/transport (retryable): breaker.record_failure; continue
    except 4xx (provider alive): breaker.record_success; continue
    breaker.record_success
    result.cost_usd = compute_cost(...)
    result.fallback_used = (model != primary)
    spend_cap.add(result.cost_usd)          # may raise BudgetExceededError
    cache.set(...) if eligible and not fallback
    callbacks.on_success(event)
    return result
callbacks.on_failure(last_error); raise last_error
```

Streaming uses the **primary model only** (matching the server) and is not
retried once bytes flow.

### 3.3 Errors — reuse the SDK's existing typed hierarchy

The engine raises the **same exceptions users already know** from the HTTP
client, so there is one error surface:

| Provider condition | Exception |
|---|---|
| 401 (and not a "no key configured" config error) | `AuthError` |
| 429 | `RateLimitError` (with `.retry_after`) |
| 5xx / network / timeout | `ProviderError` |
| unknown model / missing key / bad Azure target | `APIError` (400-ish) — config error |
| local spend cap exceeded | `BudgetExceededError` |

Additive change: add an optional `retry_after: float | None = None` field to the
`APIError` base (today only `RateLimitError` has it) so the retry layer reads it
uniformly. `is_retryable(exc)` = `APIError` with `status_code` in
`{408,429,500,502,503,504}` **or** an `httpx` timeout/transport error.

## 4. Public API

Top-level (added to `omnigate/__init__.py`):

- `completion(model, messages, *, temperature=None, max_tokens=None, top_p=None, stop=None, presence_penalty=None, frequency_penalty=None, seed=None, stream=False, fallbacks=None, cache=None, api_key=None, api_base=None, api_version=None, timeout=None, num_retries=None, user=None) -> ChatResponse | Iterator[StreamChunk]`
- `acompletion(...) -> ChatResponse | AsyncIterator[StreamChunk]` (same signature; `stream=True` returns an async iterator)
- `configure(**overrides) -> None` — set process-global `EngineConfig` defaults and/or keys
- `register_callback(*, on_success=None, on_failure=None) -> None`
- Re-exports: `EngineConfig`, plus the existing `ChatResponse`, `Message`, `Usage`, `StreamChunk`, all exceptions, `Client`, `AsyncClient`.

`messages` keeps the existing flexible coercion (`str` | `dict` | `Message` |
list thereof) via `models.coerce_messages`.

Return type: the **existing typed `ChatResponse`** (`.content`, `.usage`,
`.cost_usd`, `.model`, `.provider`, `.finish_reason`, `.cached`,
`.fallback_used`, `.latency_ms`). `completion(stream=True)` returns an
`Iterator[StreamChunk]`; `acompletion(stream=True)` an `AsyncIterator[StreamChunk]`
(litellm idiom). Typed via `@overload`.

`models.ChatRequest` is extended (additively) with the sampling passthrough
fields it currently lacks (`top_p`, `stop`, `presence_penalty`,
`frequency_penalty`, `seed`) plus `fallback_models` and `cache`, so one request
model serves both the engine and the HTTP client. All new fields are optional →
backward compatible.

## 5. Config & key resolution

`EngineConfig` is a **plain dataclass** (deliberately not pydantic-settings,
which would re-introduce the `DATABASE_URL`/`REDIS_URL` requirement). It reads
optional env with sane defaults:

| Field | Env | Default |
|---|---|---|
| `timeout` | `OMNIGATE_TIMEOUT_SECONDS` | 60.0 |
| `retry_max_attempts` | `OMNIGATE_RETRY_MAX_ATTEMPTS` | 3 |
| `retry_base_delay` | `OMNIGATE_RETRY_BASE_DELAY_SECONDS` | 0.25 |
| `retry_max_delay` | `OMNIGATE_RETRY_MAX_DELAY_SECONDS` | 8.0 |
| `retry_jitter` | `OMNIGATE_RETRY_JITTER_SECONDS` | 0.25 |
| `circuit_breaker_enabled` | `OMNIGATE_CIRCUIT_BREAKER_ENABLED` | True |
| `circuit_breaker_fail_threshold` | `OMNIGATE_CIRCUIT_BREAKER_FAIL_THRESHOLD` | 5 |
| `circuit_breaker_cooldown` | `OMNIGATE_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | 30.0 |
| `cache_enabled` | `OMNIGATE_CACHE_ENABLED` | False |
| `cache_ttl` | `OMNIGATE_CACHE_TTL_SECONDS` | 300 |
| `max_spend_usd` | `OMNIGATE_MAX_SPEND_USD` | None (off) |

Per-call kwargs (`timeout=`, `num_retries=`, `cache=`) override the global for
that call. `omnigate.configure(...)` sets the global instance and stored keys.

Key/target resolution (`keys.py`), explicit-wins-over-env:

- OpenAI: `api_key=` else `OPENAI_API_KEY`
- Anthropic: `api_key=` else `ANTHROPIC_API_KEY`
- Gemini: `api_key=` else `GEMINI_API_KEY` else `GOOGLE_API_KEY`
- Azure: `api_key=` else `AZURE_OPENAI_API_KEY`; endpoint `api_base=` else
  `AZURE_OPENAI_ENDPOINT`; version `api_version=` else `AZURE_OPENAI_API_VERSION`
  (default `2024-10-21`); deployment from `model="azure/<deployment>"`.
- Missing key ⇒ `APIError` (config error) with a clear "set OPENAI_API_KEY or
  pass api_key=" message.

## 6. Feature scope (in-process)

**Core (always on):** sync + async + streaming; model→provider routing
(OpenAI/Anthropic/Gemini/Azure by prefix); retry + backoff + jitter (honoring
`Retry-After`); fallback chain; in-memory circuit breaker; per-call cost
tracking via the pricing table.

**Extras (in this version):**

- **In-memory response cache** — opt-in (`cache=True` or `cache_enabled`),
  deterministic only (non-streaming, `temperature == 0`). A TTL dict; key =
  SHA-256 over `(provider, model, messages, max_tokens, top_p, stop, seed)`
  (no project scoping — single tenant in-process). Non-fatal, mirrors the
  server's eligibility rules in `app/services/cache.py`.
- **Callbacks / logging hooks** — `register_callback(on_success=…, on_failure=…)`.
  Each receives an event dict: `{model, provider, usage, cost_usd, latency_ms,
  cached, fallback_used, exception?}`. Exceptions inside callbacks are swallowed
  and logged (never break a request).
- **Local spend cap** — when `max_spend_usd` is set, a process-local accumulator
  raises `BudgetExceededError` once cumulative cost would exceed it. Reuses the
  existing exception.

**Deferred (fast-follow, not in this version):** a litellm-style `Router` for
weighted/round-robin load-balancing across duplicate deployments. `fallbacks=`
already delivers the primary value (failover); weighted balancing is niche for
an in-process library and adds meaningful surface/tests.

## 7. Backward compatibility

- No public symbol removed or changed; only additions.
- `ChatRequest` gains optional fields only.
- `APIError` gains an optional `retry_after` field only.
- Existing `sdk/tests/test_client.py` and the server test suite stay green.
- No new runtime dependencies (`httpx` + `pydantic` already cover the engine).
- Version `0.1.0 → 0.2.0` (minor; additive).

## 8. Testing

New `sdk/tests/test_engine.py`, fully offline via `httpx.MockTransport`
(same approach as the existing SDK tests):

- routing: each prefix → correct provider URL/headers/payload
- key resolution: explicit arg wins; env fallback; missing key → `APIError`
- happy path sync + async; cost computed; `latency_ms` populated
- retry: 429-then-200 retries and succeeds; non-retryable 400 raises immediately
- fallback: primary persistent 5xx → second candidate succeeds; `fallback_used=True`
- circuit breaker: opens after `fail_threshold`; open ⇒ candidate skipped
- cache: identical `temperature=0` request served from cache (`cached=True`, cost 0); non-deterministic not cached
- spend cap: cumulative cost over `max_spend_usd` raises `BudgetExceededError`
- callbacks: `on_success`/`on_failure` fire with correct event fields
- streaming: text reassembly + exactly one terminal usage chunk (per provider shape); async mirror
- per-provider parser unit tests (OpenAI/Azure shared, Anthropic event accumulation, Gemini `usageMetadata` last-wins)

## 9. Docs & packaging

- Rewrite `sdk/README.md` to lead with in-process usage (install → `completion`
  → async → streaming → fallbacks → cost/callbacks → config/keys), then a
  "Pointing at a hosted gateway" section for `Client`/`AsyncClient`.
- Fix the `your-org` placeholder URL in `sdk/pyproject.toml` → `sreekarp/omnigate`.
- Update root `README.md` and `CLAUDE.md` SDK notes to mention the dual mode.
- `publish.yml` already builds `./sdk` and publishes `omnigate` via Trusted
  Publishing on a GitHub release. Delivery: I prepare + bump + commit/push;
  **the user creates a release tag and CI publishes.**

## 10. Risks / mitigations

- **Drift between `app/providers` and `omnigate/providers`.** Accepted: the SDK
  is intentionally standalone (already duplicates the wire schema). Mitigated by
  keeping the ports faithful and adding parser unit tests on the SDK side.
- **Sync streaming complexity.** Mitigated by a dedicated sync executor
  (`httpx.Client.stream`) rather than bridging the async generator.
- **Python version.** SDK targets `>=3.10`; ported code uses `X | None`
  annotations (fine on 3.10+). CI matrix already runs 3.11/3.12.
