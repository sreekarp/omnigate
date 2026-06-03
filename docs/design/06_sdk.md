This is the design subtask for the SDK + packaging subsystem; I'll leave task #1 as in_progress (the parent workflow owns its lifecycle) and just return the spec. The spec is my deliverable.

---

# Implementation Spec — Python Client SDK (`llm-gateway-sdk`) + Packaging

## 0. Scope & ground rules
- New top-level dir `sdk/` = a **separate, independently installable package** (its own `pyproject.toml`). It does **not** import anything from `app/`.
- Runtime deps: **`httpx` + `pydantic` only**. No `tenacity` (hand-roll backoff), no `anyio` beyond what httpx pulls in.
- Targets the gateway's own HTTP surface (verified against the live code): `POST /v1/signup`, `POST /v1/keys`, `GET /v1/me`, `POST /v1/chat`, `GET /health`.
- `models()` and `metrics()` target endpoints **not yet present** in the gateway (only a Jinja dashboard + `/health` exist today). They are designed here against the agreed shapes for workflow tasks **#5 (metrics JSON)** and **#6 (`/v1/models`)**. **Gotcha/dependency:** the SDK must ship and pass tests *before* those endpoints exist using `MockTransport`; runtime use of `models()`/`metrics()` requires tasks #5/#6 to land. Flag this in the SDK docstring.

---

## 1. Wire contract the SDK must speak (from current code)

**Auth headers (every authenticated call):**
- `x-api-key: <gateway key>` (the `llmg_...` key from signup/admin). Resolved in `app/middleware/auth.py`.
- `x-user-id: <str>` — optional, per-request attribution → `UsageRecord.user_id`.

**`POST /v1/chat`** — body = `ChatRequest`:
```json
{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hi"}],
 "max_tokens":null,"temperature":null,"stream":false}
```
- `role` ∈ `system|user|assistant`; `messages` min length 1; `max_tokens>=1`; `0<=temperature<=2`.
- **Non-stream response** = `ChatResponse`:
```json
{"id":"...","provider":"openai","model":"gpt-4o-mini",
 "content":"...","usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8},
 "cost_usd":0.0000042}
```
- **Stream response**: `Content-Type: text/plain`, header `x-request-id: <uuid>`. Body is **raw concatenated text chunks** (NOT SSE, no `data:` framing, no `[DONE]`). On mid-stream provider failure the server appends a literal sentinel line: `\n[error] <message>` and then closes (HTTP status is already 200 by then). **Streaming responses carry no usage/cost** (server records zeros).

**`POST /v1/signup`** (public, 201) — body `SignupRequest` `{email, org_name?, project_name="Default"}` → `SignupResponse` `{org_id, project_id, email, api_key, message}`. `api_key` shown once.

**`POST /v1/keys`** (auth, **204 No Content**, empty body) — body `SetProviderKeyRequest` `{provider:"openai"|"anthropic", api_key}` (`api_key` min length 8).

**`GET /v1/me`** (auth) → `MeResponse` `{project_id, org_id, project_name, key_prefix, rate_limit_per_min, configured_providers:[...]}`.

**`GET /health`** → `{"status":"ok","version":"0.1.0"}`.

**Error envelope (FastAPI):** all errors are `{"detail": "<string>"}` (note: validation errors from Pydantic are `{"detail":[{...}]}` — a *list*; handle both). Status codes the SDK must map:

| HTTP | Source in code | SDK exception |
|---|---|---|
| 401 | auth.py (missing/invalid key) | `AuthError` |
| 429 | rate_limit.py (`Retry-After: 60` header set) | `RateLimitError` |
| 402 | budget.py ("Project/Organisation daily budget exceeded") | `BudgetExceededError` |
| 400 | no provider key configured / unknown model / bad request | `APIError` (or `ProviderError` if detail starts with provider name — see §4) |
| 502 (or provider's 4xx/5xx surfaced) | provider adapters raise `ProviderError(status_code=...)` → `HTTPException` | `ProviderError` |
| other 4xx/5xx | — | `APIError` |

**Gotcha:** the gateway surfaces upstream provider failures with the *provider's own* status code (e.g. a real `401` from OpenAI becomes a `401` from the gateway with detail `"OpenAI error 401: ..."`). So **status code alone is ambiguous** for 401. Disambiguate by detail substring (`"error 4"`/provider name) — see classification in §4.

---

## 2. File layout

```
sdk/
├── pyproject.toml
├── README.md
├── LICENSE
└── src/
    └── llm_gateway/
        ├── __init__.py          # public re-exports + __version__
        ├── _version.py          # __version__ = "0.1.0"
        ├── models.py            # Pydantic v2 mirror models (no server import)
        ├── exceptions.py        # exception hierarchy + classify()
        ├── _transport.py        # shared URL/header/retry/parse logic (sync+async)
        ├── _retry.py            # backoff math, Retry-After parsing (pure fns)
        ├── client.py            # Client (sync, httpx.Client)
        ├── async_client.py      # AsyncClient (httpx.AsyncClient)
        └── py.typed             # PEP 561 marker
tests/sdk/
        ├── conftest.py          # MockTransport fixtures, fake gateway app
        ├── test_models.py
        ├── test_retry.py
        ├── test_sync_client.py
        ├── test_async_client.py
        ├── test_streaming.py
        └── test_exceptions.py
```

Use `src/` layout so tests run against the installed package (`pip install -e sdk/`), preventing accidental imports of the server.

---

## 3. `models.py` — typed request/response (mirror, do not import server)

Pydantic v2, identical field names/types to `app/schemas/chat.py` and `app/schemas/account.py`. Keep them permissive on read (forward-compat) but validated on write.

```python
from typing import Literal, Optional
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatResponse(BaseModel):
    model_config = {"extra": "ignore"}   # forward-compat with new server fields
    id: str
    provider: str
    model: str
    content: str
    usage: Usage = Usage()
    cost_usd: float = 0.0

class StreamChunk(BaseModel):
    """A piece of streamed text + provenance (request_id from x-request-id)."""
    text: str
    request_id: str | None = None

class SignupResponse(BaseModel):
    model_config = {"extra": "ignore"}
    org_id: str            # accept str; server sends UUID-as-str in JSON
    project_id: str
    email: str
    api_key: str
    message: str = ""

class MeResponse(BaseModel):
    model_config = {"extra": "ignore"}
    project_id: str
    org_id: str
    project_name: str
    key_prefix: str
    rate_limit_per_min: int
    configured_providers: list[str] = []

# Designed against tasks #6 / #5 (endpoints not yet built):
class ModelInfo(BaseModel):
    model_config = {"extra": "ignore"}
    id: str                       # model name, e.g. "gpt-4o-mini"
    provider: str                 # "openai" | "anthropic"
    input_per_1k: float | None = None
    output_per_1k: float | None = None

class MetricsResponse(BaseModel):
    model_config = {"extra": "ignore"}
    window: str = "today"         # echo of requested window
    spend_usd: float = 0.0
    request_count: int = 0
    by_model: dict[str, float] = {}     # model -> spend
    by_status: dict[str, int] = {}      # "ok"/"error"/... -> count
```
**Gotcha:** server uuid fields serialize to JSON strings; type them `str` in the SDK (don't import `uuid` into the public surface).

---

## 4. `exceptions.py` — typed hierarchy + classifier

```python
class GatewayError(Exception):
    """Base for all SDK errors."""

class APIError(GatewayError):
    def __init__(self, message, *, status_code=None, detail=None, request_id=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail            # raw parsed "detail" (str | list | None)
        self.request_id = request_id

class AuthError(APIError): ...          # 401 (gateway key invalid/missing)
class RateLimitError(APIError):
    def __init__(self, *a, retry_after: float | None = None, **kw):
        super().__init__(*a, **kw)
        self.retry_after = retry_after  # parsed from Retry-After header
class BudgetExceededError(APIError): ...# 402
class ProviderError(APIError): ...      # upstream provider failure (502, or surfaced 4xx)

class ConnectionError(GatewayError):    # network/timeout (httpx.TransportError) after retries
    """Could not reach the gateway."""
```

**`classify(status, detail, headers) -> APIError`** rules (order matters):
1. `429` → `RateLimitError(retry_after=parse Retry-After)`.
2. `402` → `BudgetExceededError`.
3. detail (str) contains `"error "` digit pattern OR starts with `OpenAI`/`Anthropic`/`No provider`/`No <x> key` OR `status==502` → `ProviderError`. (Catches upstream-surfaced 401/403/500 so a *provider* 401 isn't mistaken for a *gateway* auth failure.)
4. `401` → `AuthError`.
5. default → `APIError`.

`detail` extraction: parse JSON body; `body.get("detail")` may be a `str` or a `list` (validation errors) — normalize to a readable message (`str(detail)` if list).

---

## 5. `_retry.py` — backoff (pure, testable, no deps)

```python
@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 2          # total attempts = max_retries + 1
    backoff_base: float = 0.5     # seconds
    backoff_max: float = 8.0
    jitter: float = 0.2           # +/- fraction
    retry_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504})

def should_retry(status: int | None, attempt: int, cfg) -> bool: ...
def compute_delay(attempt, cfg, retry_after: float | None) -> float:
    # honor Retry-After when present, else exponential: base*2**attempt, capped, +/- jitter
def parse_retry_after(value: str | None) -> float | None:
    # int seconds OR HTTP-date -> seconds-from-now; None on parse failure
```
- Retry on listed statuses **and** on `httpx.TransportError`/`httpx.TimeoutException` (connection-level).
- **Do NOT retry streaming requests** by default (a partially consumed stream can't be replayed) — only retry the *initial* connection before any bytes are yielded. Keep simple: stream retries only when the failure happens before the first chunk.
- Jitter via `random` (stdlib). Sleep: `time.sleep` (sync) / `await asyncio.sleep` (async).

---

## 6. `_transport.py` — shared logic

Hold provider-agnostic helpers used by both clients:
- `build_url(base_url, path)`, `auth_headers(api_key, user_id) -> dict`.
- `prepare_chat_body(model, messages, max_tokens, temperature, stream) -> dict` (validates via `ChatRequest`, dumps with `exclude_none=False` to match server defaults; keep `max_tokens`/`temperature` omitted when `None` — use `model_dump(exclude_none=True)` then re-add `stream`).
- `handle_response(resp) -> dict`: if `resp.status_code >= 400`: parse detail, `raise classify(...)`; else `return resp.json()`.
- `iter_text_chunks(raw_iter, request_id) -> Iterator[StreamChunk]`: pass through text; detect the `\n[error] ` sentinel — when seen, raise `ProviderError(detail=rest_of_line, request_id=...)` so streaming errors are *typed*, not silently yielded as text. **Gotcha:** the sentinel can be split across httpx chunk boundaries — buffer a small tail and scan across the join.

---

## 7. Public API surface (signatures)

Both clients share identical signatures; `AsyncClient` methods are `async` and stream helpers return async iterators.

### Constructor
```python
Client(
    api_key: str | None = None,        # gateway key; sent as x-api-key
    *,
    base_url: str = "http://localhost:8000",
    user_id: str | None = None,        # default x-user-id for all calls
    timeout: float = 60.0,
    retries: int = 2,
    retry_config: RetryConfig | None = None,
    headers: dict[str, str] | None = None,   # extra headers merged in
    transport: httpx.BaseTransport | None = None,   # for tests (MockTransport)
)
```
- Owns an `httpx.Client` (or `AsyncClient`). Supports context-manager + `.close()` / `await .aclose()`.
- `api_key` optional so `signup()` (public) works on a keyless client.

### Methods (sync `Client`)
```python
def chat(
    self, *,
    model: str,
    messages: list[Message | dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    user_id: str | None = None,          # per-call override of x-user-id
) -> ChatResponse: ...

def chat_stream(
    self, *,
    model: str,
    messages: list[Message | dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    user_id: str | None = None,
    as_chunks: bool = False,             # False -> yield str; True -> yield StreamChunk
) -> Iterator[str] | Iterator[StreamChunk]:
    """Context-managed under the hood; raises ProviderError on the [error] sentinel."""

def models(self) -> list[ModelInfo]: ...          # GET /v1/models  (task #6)
def metrics(self, *, window: str = "today") -> MetricsResponse: ...  # GET /v1/metrics (task #5)
def signup(self, *, email: str, org_name: str | None = None,
           project_name: str = "Default") -> SignupResponse: ...   # POST /v1/signup
def set_provider_key(self, *, provider: Literal["openai","anthropic"],
                     api_key: str) -> None: ...    # POST /v1/keys -> 204
def me(self) -> MeResponse: ...                    # GET /v1/me  (bonus, mirrors server)
def health(self) -> dict: ...                      # GET /health
def close(self) -> None: ...
def __enter__/__exit__: ...
```

### `AsyncClient` — same names, `async def`; `chat_stream` returns `AsyncIterator[...]`; `aclose()`, `async with`.

**Helper for ergonomics:** accept `messages` as either `list[dict]` or `list[Message]`; coerce dicts through `Message(**d)`. Also offer a convenience: `messages` may be a bare `str` → treated as `[{"role":"user","content":str}]`.

**Streaming-after-signup helper note:** `set_provider_key` returns `None` (204). Don't try to `.json()` a 204 — guard on status 204/empty body.

---

## 8. Usage examples

**Sync — onboard + chat + stream:**
```python
from llm_gateway import Client

# Public client (no key) just for signup:
with Client(base_url="https://gw.example.com") as anon:
    acct = anon.signup(email="dev@acme.com", org_name="Acme", project_name="prod")

client = Client(api_key=acct.api_key, base_url="https://gw.example.com", user_id="u-42")
client.set_provider_key(provider="openai", api_key="sk-...")   # BYOK, stored encrypted

resp = client.chat(model="gpt-4o-mini",
                   messages=[{"role": "user", "content": "Say hi in French"}])
print(resp.content, resp.usage.total_tokens, resp.cost_usd)

for piece in client.chat_stream(model="gpt-4o-mini",
                                messages="Stream me a haiku"):
    print(piece, end="", flush=True)
client.close()
```

**Async + typed errors:**
```python
import asyncio
from llm_gateway import AsyncClient, BudgetExceededError, RateLimitError

async def main():
    async with AsyncClient(api_key="llmg_...", base_url="https://gw.example.com") as c:
        try:
            async for chunk in c.chat_stream(model="claude-3-5-sonnet-latest",
                                              messages=[{"role":"user","content":"hi"}],
                                              as_chunks=True):
                print(chunk.text, end="")
        except RateLimitError as e:
            print("slow down; retry after", e.retry_after)
        except BudgetExceededError as e:
            print("budget hit:", e.detail)

asyncio.run(main())
```

---

## 9. `sdk/pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-gateway-sdk"
version = "0.1.0"
description = "Python client SDK for the LLM Gateway (sync + async, streaming, typed)."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "LLM Gateway" }]
dependencies = [
    "httpx>=0.27,<1.0",
    "pydantic>=2.7,<3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24"]

[project.urls]
Homepage = "https://github.com/your-org/llm-gateway"

[tool.hatch.build.targets.wheel]
packages = ["src/llm_gateway"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```
- `requires-python = ">=3.10"` (uses `X | Y` unions, `list[...]`). The *server* runs 3.14, but the SDK should support older client runtimes — keep it broad.
- Ship `py.typed` (declare in wheel via `include`/force-include if hatch doesn't auto-pick it: add `[tool.hatch.build] artifacts = ["src/llm_gateway/py.typed"]`).

---

## 10. Server-side `pyproject.toml` (gateway) + CLI

Recommend converting the gateway to a proper package with a console entry point. Place at repo root (`C:\Sreekar\New folder\pyproject.toml`).

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "llm-gateway"
version = "0.1.0"                      # source from app.__version__
description = "Production LLM gateway: auth, rate limit, budgets, cost tracking."
requires-python = ">=3.12"             # server uses 3.14 features; >=3.12 safe floor
dependencies = [
    "fastapi==0.115.6",
    "uvicorn[standard]==0.34.0",
    "sqlalchemy[asyncio]==2.0.36",
    "asyncpg==0.30.0",
    "alembic==1.14.0",
    "greenlet==3.1.1",
    "redis==5.2.1",
    "httpx==0.28.1",
    "pydantic==2.10.4",
    "pydantic-settings==2.7.1",
    "jinja2==3.1.5",
    "cryptography==44.0.0",
]

[project.optional-dependencies]
dev = ["pytest==8.3.4", "pytest-asyncio==0.25.0"]

[project.scripts]
llm-gateway = "app.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["app"]
```
- Mirror the pinned versions from `requirements.txt` (single source of truth — can later generate `requirements.txt` from this). Keep `requirements.txt` for Docker layer caching.

**New file `app/cli.py`** — `main()` using **stdlib `argparse`** (no Click; respects "no heavy deps"). Subcommands operate on the same DB/services the server uses (async funcs wrapped with `asyncio.run`):

| Command | Action | Backing code |
|---|---|---|
| `llm-gateway serve [--host --port --reload]` | `uvicorn.run("app.main:app", ...)` | main.py |
| `llm-gateway db upgrade [--rev head]` | shell to `alembic upgrade` (or `alembic.config.main`) | Alembic |
| `llm-gateway db revision -m MSG --autogenerate` | alembic revision | Alembic |
| `llm-gateway org create --name --daily-budget` | insert Organisation | models/services |
| `llm-gateway org list` | list orgs + today's spend | usage.org_spend_today |
| `llm-gateway project create --org-id --name [--daily-budget --rate-limit]` | create project, **print plaintext key once** | security.generate_api_key |
| `llm-gateway project list [--org-id]` | list projects (prefix only) | models |
| `llm-gateway key set --project-id --provider --api-key` | store BYOK key encrypted | services.credentials.set_provider_key |
| `llm-gateway key list --project-id` | list configured providers (no secrets) | list_configured_providers |
| `llm-gateway usage --project-id\|--org-id [--days N]` | print spend/request/token rollup | usage service |
| `llm-gateway config check` | validate env (`get_settings()`), ping DB + Redis, report | config/db/redis |
| `llm-gateway version` | print `app.__version__` | __init__ |

CLI rules: async DB work via `asyncio.run(...)`; use the existing `get_session`/services (do **not** duplicate SQL); print with `logging`-friendly stdout (CLI may `print` results to stdout since it's a user tool, but route diagnostics through the logger to honor the no-`print`-in-app convention — keep result output on stdout, logs on stderr). Exit non-zero on failure.

---

## 11. Test plan (httpx `MockTransport`, no network, no DB)

All SDK tests inject `transport=httpx.MockTransport(handler)` into the client — no live server needed.

1. **`test_models.py`** — round-trip each model from a sample server JSON; `extra="ignore"` drops unknown fields; `ChatRequest` validation rejects empty messages, `temperature>2`, `max_tokens<1`; `messages` coercion from `dict`/`str`.
2. **`test_retry.py`** (pure) — `compute_delay` monotonic + capped + jittered within bounds; `parse_retry_after` for int seconds and HTTP-date; `should_retry` honors `retry_statuses` and attempt count; transport errors retried, 4xx (non-429) not retried.
3. **`test_sync_client.py`** — `chat()` happy path returns typed `ChatResponse` with correct headers sent (`x-api-key`, `x-user-id`, per-call `user_id` override); `set_provider_key()` handles **204** (no body) without error; `me()`/`health()`/`signup()` map correctly; `signup()` works on a keyless client.
4. **`test_async_client.py`** — mirror of #3 with `AsyncClient` + `MockTransport` async handler; `async with` lifecycle closes the client.
5. **`test_streaming.py`** —
   - handler returns a streamed `text/plain` body in several chunks + `x-request-id` header; assert reassembled text == concatenation; `as_chunks=True` yields `StreamChunk` with `request_id` populated.
   - **Sentinel test:** body ends with `\n[error] OpenAI error 500: boom`; assert the iterator raises `ProviderError` with `.detail` containing the message and `.request_id` set.
   - **Split-sentinel test:** the `\n[error] ` token split across two chunks still raises (buffer/join logic).
6. **`test_exceptions.py`** — `classify()` mapping: 401→`AuthError`; 429 (+`Retry-After: 30`)→`RateLimitError(retry_after=30)`; 402→`BudgetExceededError`; 502 or detail `"OpenAI error 401: ..."`→`ProviderError` (the ambiguous-401 case); validation-error `detail` as a **list** is stringified into `APIError`; network `httpx.ConnectError` after retries → SDK `ConnectionError`.
7. **Retry integration** — `MockTransport` returns `429,429,200`; assert `chat()` succeeds after 2 retries and that `compute_delay`-derived sleeps were invoked (monkeypatch `time.sleep`/`asyncio.sleep` to record calls).
8. **`config check` / CLI** (server side, optional in this subtask) — argparse parser builds; `version` prints `__version__`; smoke-invoke `serve --help`.

**Verification gate:** `pip install -e sdk/[dev] && pytest tests/sdk -q` green; `python -c "import llm_gateway; print(llm_gateway.__version__)"`; `python -c "from llm_gateway import Client, AsyncClient, ChatResponse, AuthError, RateLimitError, BudgetExceededError, ProviderError, APIError"` (public surface importable). `mypy src/llm_gateway` clean (ship `py.typed`).

---

## 12. Key gotchas (consolidated)
- **Streaming is plain text, not SSE** — do not parse `data:`/`[DONE]`; just stream bytes and watch for the `\n[error] ` sentinel (which can straddle chunk boundaries).
- **Streaming carries no usage/cost** — `StreamChunk` has text + request_id only; document that `cost_usd`/`usage` are unavailable for streamed calls (server records zeros).
- **204 from `/v1/keys`** — never call `.json()`.
- **Ambiguous 401** — a gateway-auth 401 vs a provider-surfaced 401 differ only by `detail`; classify by substring before falling back to `AuthError`.
- **`detail` may be a list** (Pydantic validation errors) — normalize before constructing the exception message.
- **Don't retry mid-stream** — only retry the initial connect before the first byte.
- **No server imports** — SDK mirrors schemas; `models()`/`metrics()` depend on workflow tasks #6/#5 shipping the endpoints (tests use mocks until then).
- **UUIDs are JSON strings** in responses — type SDK fields as `str`.

**Relevant existing files referenced (absolute paths):**
`C:\Sreekar\New folder\app\routers\chat.py`, `C:\Sreekar\New folder\app\routers\account.py`, `C:\Sreekar\New folder\app\schemas\chat.py`, `C:\Sreekar\New folder\app\schemas\account.py`, `C:\Sreekar\New folder\app\middleware\{auth,rate_limit,budget}.py`, `C:\Sreekar\New folder\app\providers\base.py`, `C:\Sreekar\New folder\app\security.py`, `C:\Sreekar\New folder\app\main.py`, `C:\Sreekar\New folder\requirements.txt`.

**New files to create (absolute paths):** under `C:\Sreekar\New folder\sdk\` (package per §2), plus `C:\Sreekar\New folder\pyproject.toml` and `C:\Sreekar\New folder\app\cli.py` (server packaging + CLI per §10), and tests under `C:\Sreekar\New folder\tests\sdk\`.