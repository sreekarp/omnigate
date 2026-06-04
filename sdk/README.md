# omnillm

A small, fully-typed Python client for the **OmniLLM** — sync **and** async,
streaming-aware, with typed errors. Depends only on `httpx` and `pydantic`.

```
pip install omnillm
```

The SDK is a standalone package: it imports nothing from the gateway server, and
mirrors the gateway's wire schema with its own Pydantic models.

## Quick start (sync)

```python
from omnillm import Client

# Public client (no key) just for signup:
with Client(base_url="https://gw.example.com") as anon:
    acct = anon.signup(email="dev@acme.com", org_name="Acme", project_name="prod")

client = Client(api_key=acct.api_key, base_url="https://gw.example.com", user_id="u-42")

# Bring-your-own-key: stored encrypted by the gateway (POST /v1/keys -> 204).
client.set_provider_key(provider="openai", api_key="sk-...")

resp = client.chat(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hi in French"}],
)
print(resp.content, resp.usage.total_tokens, resp.cost_usd)
client.close()
```

`messages` is flexible: pass a bare string (treated as one `user` message), a
single dict/`Message`, or a list of dicts/`Message`s.

```python
client.chat(model="gpt-4o-mini", messages="Just a quick question")
```

## Streaming

The gateway streams **plain text** (not SSE). The SDK reassembles it for you and
raises a typed `ProviderError` if the gateway emits a mid-stream failure.

> Streaming responses carry **no usage or cost** — `cost_usd`/`usage` are only
> populated for non-streaming `chat()` calls (the server records zeros for
> streamed requests).

```python
for piece in client.chat_stream(model="gpt-4o-mini", messages="Stream me a haiku"):
    print(piece, end="", flush=True)

# Want provenance (request_id)? Ask for StreamChunk objects:
for chunk in client.chat_stream(model="gpt-4o-mini", messages="hi", as_chunks=True):
    print(chunk.text, chunk.request_id)
```

## Async

`AsyncClient` mirrors `Client` exactly: **identical constructor and method
names**, but every method is `async def` and `chat_stream` returns an async
iterator. Use `async with` / `await client.aclose()`.

```python
import asyncio
from omnillm import AsyncClient, BudgetExceededError, RateLimitError

async def main():
    async with AsyncClient(api_key="llmg_...", base_url="https://gw.example.com") as c:
        try:
            async for chunk in c.chat_stream(
                model="claude-3-5-sonnet-latest",
                messages=[{"role": "user", "content": "hi"}],
                as_chunks=True,
            ):
                print(chunk.text, end="")
        except RateLimitError as e:
            print("slow down; retry after", e.retry_after)
        except BudgetExceededError as e:
            print("budget hit:", e.detail)

asyncio.run(main())
```

## Errors

All errors derive from `GatewayError`.

| Exception | When |
|---|---|
| `AuthError` | 401 — gateway api key missing/invalid |
| `RateLimitError` | 429 — has `.retry_after` (seconds, parsed from `Retry-After`) |
| `BudgetExceededError` | 402 — daily/monthly budget exhausted |
| `ProviderError` | 502 or an upstream provider failure surfaced by the gateway |
| `APIError` | any other 4xx/5xx; carries `.status_code`, `.detail`, `.request_id` |
| `ConnectionError` | network/timeout after retries are exhausted |

A provider-surfaced 401 (e.g. a real OpenAI 401) is classified as
`ProviderError`, not `AuthError`, by inspecting the error detail — so you can
distinguish "my gateway key is bad" from "my OpenAI key is bad".

## Retries

429 and 5xx responses, plus connection/timeout errors, are retried with
exponential backoff + jitter (honoring `Retry-After`). Configure via
`retries=` or a full `RetryConfig`:

```python
from omnillm import Client, RetryConfig

Client(api_key="llmg_...", retries=3)
Client(api_key="llmg_...", retry_config=RetryConfig(max_retries=5, backoff_max=20))
```

Streaming requests are **not** retried once bytes have started flowing.

## Pointing the OpenAI SDK at the gateway

The gateway exposes an OpenAI-compatible `POST /v1/chat/completions`, so you can
reuse the official OpenAI SDK and just change the base URL + key:

```python
from openai import OpenAI

oai = OpenAI(
    base_url="https://gw.example.com/v1",
    api_key="llmg_...",          # your gateway key, sent as the bearer token
    default_headers={"x-api-key": "llmg_...", "x-user-id": "u-42"},
)
oai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi"}],
)
```

This SDK also offers a thin `completions(...)` helper returning the raw
OpenAI-shaped dict.

## Models & metrics

```python
# GET /v1/models -> list[ModelInfo] (OpenAI-style cards; pricing may be None)
for m in client.models():
    if m.pricing:
        print(m.id, m.owned_by, m.provider, m.pricing.input_per_1k_usd)
    else:
        print(m.id, m.owned_by, "(unpriced)")

# GET /v1/metrics -> MetricsResponse. range is one of 1h | 24h | 7d | 30d
# (default "24h"). The response carries totals, an optional grouped breakdown,
# and a bucketed timeseries.
mx = client.metrics(range="7d")
print(mx.totals.requests, mx.totals.cost_usd, mx.totals.cache_hit_rate)
print(mx.totals.p95_latency_ms)
for row in mx.breakdown:          # grouped by provider/model/user/status
    print(row.key, row.requests, row.cost_usd)
for pt in mx.timeseries:          # bucketed points
    print(pt.bucket, pt.requests)
```

## Gateway key management

`POST /v1/keys/api` mints an additional named gateway key; the plaintext
`api_key` is returned **once** and never recoverable afterwards.

```python
key = client.create_api_key(name="ci")   # POST /v1/keys/api -> ApiKeyCreated
print(key.api_key, key.key_prefix, key.id)  # persist key.api_key now
```

## Other methods

```python
client.me()      # GET /v1/me  -> MeResponse
client.health()  # GET /health -> dict
```

## License

MIT
