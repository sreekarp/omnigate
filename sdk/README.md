# omnigate

A small, fully-typed, **litellm-style** multi-provider LLM SDK — sync **and**
async, streaming-aware, with typed errors. Depends only on `httpx` and
`pydantic`.

```
pip install omnigate
```

Two ways to use it:

1. **In-process** — call OpenAI / Anthropic / Gemini / Azure **directly**, no
   server to run. You get routing, retry + backoff, fallbacks, circuit
   breaking, per-call cost tracking, an opt-in response cache, callbacks and a
   local spend cap.
2. **Hosted gateway client** — point `Client` / `AsyncClient` at a running
   **OmniGate** server for centralised auth, budgets, rate limiting and metrics.

---

## In-process quick start

Set a provider key the usual way (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY` / `GOOGLE_API_KEY`, or `AZURE_OPENAI_API_KEY` +
`AZURE_OPENAI_ENDPOINT`) — or pass `api_key=` explicitly.

```python
import omnigate

r = omnigate.completion(model="gpt-4o-mini", messages="Say hi in French")
print(r.content, r.usage.total_tokens, r.cost_usd, r.model, r.provider)
```

`messages` is flexible: pass a bare string (treated as one `user` message), a
single dict/`Message`, or a list of dicts/`Message`s. The model name routes to
the provider by prefix (`gpt-*`/`o1`/`o3`/`o4` → OpenAI, `claude-*` → Anthropic,
`gemini-*` → Gemini, `azure/<deployment>` → Azure OpenAI).

### Async

```python
import asyncio, omnigate

async def main():
    r = await omnigate.acompletion(
        model="claude-3-5-haiku-latest",
        messages=[{"role": "user", "content": "hi"}],
    )
    print(r.content)

asyncio.run(main())
```

### Streaming

`completion(stream=True)` returns an iterator of `StreamChunk`; the async twin
returns an async iterator. Content chunks carry `text`; the final chunk carries
`usage`.

```python
for chunk in omnigate.completion(model="gpt-4o-mini", messages="haiku", stream=True):
    print(chunk.text, end="", flush=True)

# async
async for chunk in await omnigate.acompletion(model="gpt-4o-mini",
                                               messages="haiku", stream=True):
    print(chunk.text, end="")
```

### Fallbacks

Try models in order until one succeeds. Each may resolve to a different
provider; transient failures (429/5xx/timeout) trip the breaker, client errors
(4xx) just move on. `response.fallback_used` tells you if a fallback answered.

```python
r = omnigate.completion(
    model="gpt-4o-mini",
    messages="hi",
    fallbacks=["claude-3-5-haiku-latest", "gemini-1.5-flash"],
)
```

### Cost tracking

Every non-streamed response carries `cost_usd` computed from a built-in
per-model price table (`omnigate.pricing`). Cached hits are billed as `0.0`.

### Response cache (opt-in)

A deterministic, in-memory TTL cache for repeated `temperature=0` calls. Enable
per call with `cache=True`, or globally via `configure(cache_enabled=True)`.

```python
r1 = omnigate.completion(model="gpt-4o-mini", messages="2+2?", temperature=0, cache=True)
r2 = omnigate.completion(model="gpt-4o-mini", messages="2+2?", temperature=0, cache=True)
assert r2.cached and r2.cost_usd == 0.0   # served from cache, no second API call
```

### Callbacks

Register success/failure hooks to log usage, cost and latency to your own sink.

```python
omnigate.register_callback(
    on_success=lambda e: print(e.provider, e.model, e.cost_usd, e.latency_ms),
    on_failure=lambda e: print("failed:", e.exception),
)
```

### Local spend cap

Set a process-wide USD ceiling; once reached, further calls raise
`BudgetExceededError`.

```python
omnigate.configure(max_spend_usd=5.00)
```

### Configuration & keys

`configure(...)` sets process-global defaults and/or keys; per-call kwargs
(`timeout=`, `num_retries=`, `cache=`, `api_key=`, `api_base=`, `api_version=`)
override them. Everything also reads from the environment:

| Setting | Env var | Default |
|---|---|---|
| Request timeout (s) | `OMNIGATE_TIMEOUT_SECONDS` | 60 |
| Retry attempts | `OMNIGATE_RETRY_MAX_ATTEMPTS` | 3 |
| Retry base delay (s) | `OMNIGATE_RETRY_BASE_DELAY_SECONDS` | 0.25 |
| Retry max delay (s) | `OMNIGATE_RETRY_MAX_DELAY_SECONDS` | 8.0 |
| Retry jitter (s) | `OMNIGATE_RETRY_JITTER_SECONDS` | 0.25 |
| Circuit breaker on | `OMNIGATE_CIRCUIT_BREAKER_ENABLED` | true |
| Breaker fail threshold | `OMNIGATE_CIRCUIT_BREAKER_FAIL_THRESHOLD` | 5 |
| Breaker cooldown (s) | `OMNIGATE_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | 30 |
| Cache on | `OMNIGATE_CACHE_ENABLED` | false |
| Cache TTL (s) | `OMNIGATE_CACHE_TTL_SECONDS` | 300 |
| Local spend cap (USD) | `OMNIGATE_MAX_SPEND_USD` | (off) |

```python
import omnigate

omnigate.configure(
    openai_api_key="sk-...",
    anthropic_api_key="...",
    azure_endpoint="https://my.openai.azure.com",
    cache_enabled=True,
    num_retries=3,   # note: in configure this is EngineConfig.retry_max_attempts
)

# Azure: deployment is taken from the model id
omnigate.completion(model="azure/my-gpt4o-deployment", messages="hi",
                    api_key="...", api_base="https://my.openai.azure.com")
```

### Errors (in-process)

All errors derive from `GatewayError`.

| Exception | When |
|---|---|
| `AuthError` | provider returned 401/403 (your provider key is bad) |
| `RateLimitError` | 429 — has `.retry_after` (honored by retry) |
| `BudgetExceededError` | local spend cap reached |
| `ProviderError` | 5xx / network / timeout (retried, then surfaced) |
| `APIError` | config errors (unknown model, missing key) and other 4xx |

---

## Hosted gateway client

If you run an **OmniGate** server, point the client at it for centralised auth,
budgets, rate limiting and metrics. The client talks the gateway's HTTP surface;
it does not call providers itself.

```python
from omnigate import Client

# Public client (no key) just for signup:
with Client(base_url="https://gw.example.com") as anon:
    acct = anon.signup(email="dev@acme.com", org_name="Acme", project_name="prod")

client = Client(api_key=acct.api_key, base_url="https://gw.example.com", user_id="u-42")
client.set_provider_key(provider="openai", api_key="sk-...")  # stored encrypted by the gateway

resp = client.chat(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])
print(resp.content, resp.usage.total_tokens, resp.cost_usd)
client.close()
```

`AsyncClient` mirrors `Client` exactly (identical constructor and method names),
but every method is `async def` and `chat_stream` returns an async iterator. Use
`async with` / `await client.aclose()`.

```python
import asyncio
from omnigate import AsyncClient, BudgetExceededError, RateLimitError

async def main():
    async with AsyncClient(api_key="llmg_...", base_url="https://gw.example.com") as c:
        try:
            async for piece in c.chat_stream(model="claude-3-5-sonnet-latest", messages="hi"):
                print(piece, end="")
        except RateLimitError as e:
            print("slow down; retry after", e.retry_after)
        except BudgetExceededError as e:
            print("budget hit:", e.detail)

asyncio.run(main())
```

### Pointing the OpenAI SDK at the gateway

The gateway exposes an OpenAI-compatible `POST /v1/chat/completions`, so you can
reuse the official OpenAI SDK and just change the base URL + key:

```python
from openai import OpenAI

oai = OpenAI(
    base_url="https://gw.example.com/v1",
    api_key="llmg_...",
    default_headers={"x-api-key": "llmg_...", "x-user-id": "u-42"},
)
oai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}])
```

### Models, metrics & key management (hosted)

```python
for m in client.models():            # GET /v1/models
    print(m.id, m.owned_by, m.provider)

mx = client.metrics(range="7d")      # GET /v1/metrics (1h | 24h | 7d | 30d)
print(mx.totals.requests, mx.totals.cost_usd, mx.totals.p95_latency_ms)

key = client.create_api_key(name="ci")   # POST /v1/keys/api -> ApiKeyCreated (plaintext shown once)
client.me(); client.health()
```

Gateway-client errors map the same exception hierarchy; a provider-surfaced 401
is classified as `ProviderError` (not `AuthError`) so you can tell "my gateway
key is bad" from "my OpenAI key is bad".

## License

MIT
