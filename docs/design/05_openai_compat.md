The `/v1/models` shape is well-established and I have it from the search results (`object: "list"`, `data: [{id, object: "model", created, owned_by}]`). I have all verified wire details needed. Here is the implementation spec.

---

# Spec: OpenAI-Compatible Endpoint + Model Catalog

Adds `POST /v1/chat/completions` (wire-compatible with OpenAI Chat Completions, non-stream + SSE) and `GET /v1/models`. Reuses existing provider routing, BYOK, pricing, and usage recording. Backward compatible: `POST /v1/chat`, all DB rows, and `ChatRequest/ChatResponse` schemas are untouched.

## 0. Verified wire facts (sources at bottom)

- Response object: `object="chat.completion"`, top-level `id`, `created` (unix int seconds), `model`, `choices[]`, `usage{prompt_tokens,completion_tokens,total_tokens}`. Each choice: `index`, `message{role,content}`, `finish_reason`, `logprobs:null`.
- Streaming: `object="chat.completion.chunk"`, choices carry `delta` instead of `message`; SSE framed as `data: {json}\n\n`; terminates with literal `data: [DONE]\n\n`.
- Usage on stream: only present when client sends `stream_options:{"include_usage":true}`; emitted as one **extra final chunk** with `choices:[]` and `usage` populated; all earlier chunks have `usage:null`. If the stream is cancelled, this chunk may never arrive.
- `/v1/models`: `{object:"list", data:[{id, object:"model", created, owned_by}]}`. OpenAI's real fields are only those four; extra fields are tolerated by the SDK, so we add `pricing`/`capabilities` as non-standard extensions.

## 1. Auth: accept `Authorization: Bearer` in addition to `x-api-key`

The OpenAI SDK sends the key only as `Authorization: Bearer <key>`. Modify `app/middleware/auth.py` `get_auth_context` to read both headers and prefer whichever is present.

```python
async def get_auth_context(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_user_id: str | None = Header(default=None, alias="x-user-id"),
    session: AsyncSession = Depends(get_session),
) -> AuthContext:
    api_key = x_api_key
    if api_key is None and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            api_key = token.strip()
    if not api_key:
        raise HTTPException(401, "Missing API key (x-api-key or Authorization: Bearer)")
    # ...unchanged: hash_api_key(api_key) -> lookup Project...
```

**Gotchas:**
- The gateway key here is the **project gateway key** (`Project.key_hash`), NOT a provider key. BYOK provider keys are still resolved server-side from `ProviderCredential`. The SDK's `api_key=` is the gateway key.
- Precedence: if both headers present, prefer `x-api-key` (keeps `/v1/chat` behavior identical); fall back to Bearer.
- `x-user-id` remains an optional custom header; the SDK can pass it via `default_headers={"x-user-id": ...}`.
- This is a single change in `get_auth_context`, so `enforce_rate_limit`/`enforce_budget` (which chain through it) automatically gain Bearer support for `/v1/chat` too — harmless and desirable.

## 2. New schemas — `app/schemas/openai_compat.py`

Pydantic v2 models mirroring OpenAI's wire shape. **Do not** modify `chat.py`. These are request/response DTOs only; we translate to/from the internal `ChatRequest`/`ChatResponse`.

```python
class OAIMessage(BaseModel):
    role: Literal["system","user","assistant","tool","developer"]
    content: str | None = None            # tolerate null; coerce to "" internally
    name: str | None = None               # accepted, ignored

class OAIStreamOptions(BaseModel):
    include_usage: bool = False

class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")   # tolerate top_p, presence_penalty, etc.
    model: str
    messages: list[OAIMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)  # newer alias
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False
    stream_options: OAIStreamOptions | None = None
    user: str | None = None               # OpenAI end-user id -> maps to x-user-id fallback

# Response (non-stream)
class OAIRespMessage(BaseModel):
    role: str = "assistant"
    content: str
class OAIChoice(BaseModel):
    index: int = 0
    message: OAIRespMessage
    finish_reason: str | None = "stop"
    logprobs: None = None
class OAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
class OpenAIChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage
```

**Translation `OpenAIChatRequest -> ChatRequest`:**
- `messages`: map each to internal `Message{role,content}`. Internal `Role` is `Literal["system","user","assistant"]`. Map `developer` -> `system`; reject `tool` role with a 400 (`{"error":{"message":"role 'tool' not supported","type":"invalid_request_error"}}`) — we don't do tool calls. Coerce `content=None` -> `""`.
- `max_tokens`: use `max_tokens or max_completion_tokens`.
- Keep `model`, `temperature`, `stream` as-is.

**Gotcha:** roles `tool`/`developer` and content arrays (multimodal `content: [{type,text}]`) — current internal schema is string-only. For v1, accept string content only; if `content` is a list, 400 with `invalid_request_error`. Document as a known limitation.

## 3. Router — `app/routers/openai_compat.py`

New `APIRouter(prefix="/v1", tags=["openai-compat"])`, registered in the app alongside the existing chat router. Reuses `enforce_budget` dependency (auth→rate→budget), `get_provider_for_model`, `get_provider_key`, `compute_cost`, `record_usage` — i.e., the exact same pipeline as `/v1/chat`.

### 3a. `POST /v1/chat/completions`

```python
@router.post("/chat/completions")
async def chat_completions(
    body: OpenAIChatRequest,
    ctx: AuthContext = Depends(enforce_budget),
    session: AsyncSession = Depends(get_session),
):
    request_id = "chatcmpl-" + uuid.uuid4().hex   # OpenAI-style id prefix
    created = int(time.time())
    internal = to_internal_chat_request(body)     # translation from §2
    user_id = ctx.user_id or body.user            # x-user-id wins, else OpenAI `user`

    provider = resolve_provider_or_400(internal.model)
    api_key = await get_provider_key(session, project_id=ctx.project.id, provider=provider.name)
    if not api_key:
        raise HTTPException(400, detail={"error": {"message": f"No {provider.name} key configured",
                                                   "type": "invalid_request_error"}})
    if body.stream:
        return await _stream_openai(body, internal, ctx, session, provider, api_key,
                                    request_id, created, user_id)
    # non-stream: identical flow to /v1/chat, then reshape
    ...
    result: ChatResponse = await provider.chat(internal, api_key)
    cost = compute_cost(internal.model, result.usage.prompt_tokens, result.usage.completion_tokens)
    await record_usage(session, org_id=..., model=result.model, prompt_tokens=...,
                       completion_tokens=..., cost=cost, status="ok", latency_ms=...,
                       request_id=request_id, user_id=user_id)
    return OpenAIChatResponse(
        id=request_id, created=created, model=result.model,
        choices=[OAIChoice(index=0,
                           message=OAIRespMessage(role="assistant", content=result.content),
                           finish_reason="stop")],
        usage=OAIUsage(prompt_tokens=result.usage.prompt_tokens,
                       completion_tokens=result.usage.completion_tokens,
                       total_tokens=result.usage.total_tokens),
    )
```

**Non-stream details:**
- `id`: `chatcmpl-<hex>` (provider's own id is discarded for the wire response so it's gateway-stable; still fine — the SDK only echoes it).
- `finish_reason`: hardcode `"stop"` for v1 (providers' adapters discard their finish_reason today). Acceptable; note as limitation.
- Error path mirrors `/v1/chat`: on `ProviderError`, record `status="error"`, then return error **in OpenAI error envelope** with the provider's `status_code`:
  `{"error": {"message": exc.message, "type": "upstream_error", "code": null}}`.
- `usage`/`cost` recording is byte-identical to `/v1/chat` — same `record_usage` call, so dashboard/budget aggregates include these requests automatically.

### 3b. Streaming (`stream: true`) — SSE

The existing `provider.stream()` yields **plain text content pieces** (`str`), not OpenAI chunk JSON. We wrap each piece into a `chat.completion.chunk` and emit SSE. Media type **must** be `text/event-stream` (NOT the `text/plain` used by `/v1/chat`).

Frame sequence:
1. **Role primer chunk** (OpenAI always sends this first):
   `data: {"id":"chatcmpl-..","object":"chat.completion.chunk","created":C,"model":M,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n`
2. **Content chunks**, one per yielded piece:
   `data: {...,"choices":[{"index":0,"delta":{"content":PIECE},"finish_reason":null}]}\n\n`
3. **Final stop chunk**:
   `data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n`
4. **Usage chunk** — only if `body.stream_options.include_usage` is true:
   `data: {...,"choices":[],"usage":{...}}\n\n` (note `choices:[]`)
5. **Terminator**: `data: [DONE]\n\n`

```python
async def _stream_openai(...):
    async def gen() -> AsyncIterator[str]:
        status_str = "ok"
        def frame(payload: dict) -> str:
            return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"
        base = {"id": request_id, "object": "chat.completion.chunk",
                "created": created, "model": internal.model}
        yield frame({**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                          "finish_reason": None}]})
        try:
            async for piece in provider.stream(internal, api_key):
                yield frame({**base, "choices": [{"index": 0, "delta": {"content": piece},
                                                  "finish_reason": None}]})
        except ProviderError as exc:
            status_str = "error"
            # surface as an SSE error event then stop
            yield frame({**base, "choices": [{"index": 0, "delta": {},
                                              "finish_reason": "error"}],
                         "error": {"message": exc.message, "type": "upstream_error"}})
        else:
            yield frame({**base, "choices": [{"index": 0, "delta": {},
                                              "finish_reason": "stop"}]})
            if body.stream_options and body.stream_options.include_usage:
                # tokens unavailable from current provider.stream() -> zeros (see gotcha)
                yield frame({**base, "choices": [],
                             "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                       "total_tokens": 0}})
        finally:
            yield "data: [DONE]\n\n"
            latency_ms = int((time.perf_counter() - started) * 1000)
            await record_usage(session, ..., prompt_tokens=0, completion_tokens=0,
                               cost=compute_cost(internal.model, 0, 0),
                               status=status_str, latency_ms=latency_ms,
                               request_id=request_id, user_id=user_id)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"x-request-id": request_id,
                                      "Cache-Control": "no-cache",
                                      "Connection": "keep-alive"})
```

**Streaming gotchas:**
- `provider.stream()` today yields **only text content** and does NOT surface token usage (matches the existing `/v1/chat` behavior of recording zeros). So the `include_usage` chunk reports zeros and the `UsageRecord` for streamed requests stays `tokens=0, cost=0` — **consistent with current `/v1/chat` semantics**. Real per-stream token counts are out of scope here; they belong to the "refactor providers for streaming usage" workitem (#3). When that lands, this endpoint just reads the real numbers — no wire change.
- JSON must be compact (`separators=(",",":")`) and each event terminated by a blank line (`\n\n`). Content pieces are JSON-string-escaped automatically by `json.dumps`, so newlines/quotes inside tokens are safe — do not hand-build the frame.
- Set `media_type="text/event-stream"`; the SDK's stream parser requires it. Do not reuse the `/v1/chat` `text/plain` path.
- Always emit `data: [DONE]\n\n` even on error, in `finally`, so SDK clients terminate cleanly.

## 4. `GET /v1/models`

Serve from a static catalog derived from `app/services/pricing.py` `_PRICING` plus a small capability map. Public-ish but still behind auth (`enforce_budget` or at least `get_auth_context`) — OpenAI requires auth for `/v1/models`, and the SDK sends the key, so keep auth on. Use `get_auth_context` (no need to run rate/budget for a metadata read).

```python
@router.get("/models")
async def list_models(ctx: AuthContext = Depends(get_auth_context)):
    return {"object": "list", "data": [model_card(m) for m in CATALOG]}

@router.get("/models/{model_id}")
async def get_model(model_id: str, ctx: AuthContext = Depends(get_auth_context)):
    card = next((model_card(m) for m in CATALOG if m == model_id), None)
    if card is None:
        raise HTTPException(404, detail={"error": {"message": f"model '{model_id}' not found",
                                                   "type": "invalid_request_error"}})
    return card
```

**`model_card(id)` shape** (OpenAI-standard fields + non-standard extensions):
```json
{
  "id": "gpt-4o-mini",
  "object": "model",
  "created": 1700000000,
  "owned_by": "openai",
  "pricing": {"input_per_1k_usd": "0.00015", "output_per_1k_usd": "0.0006"},
  "capabilities": {"streaming": true, "max_context_tokens": 128000}
}
```
- `owned_by`: derive via `get_provider_for_model(id).name` (`"openai"`/`"anthropic"`), falling back to literal owner. Reuses existing routing logic — single source of truth.
- `pricing`: pulled directly from `_PRICING[id]`; serialize `Decimal` as strings to avoid float drift. Expose a helper `pricing.get_price(model) -> tuple[Decimal,Decimal] | None` and a `pricing.known_models() -> list[str]` rather than importing `_PRICING` directly (keeps the private table private).
- `capabilities`: hand-rolled small static dict keyed by model. Keep minimal; `streaming: true` for all current models.
- `created`: a fixed sentinel (e.g. `1700000000`) is fine — clients don't depend on it.

**Gotcha:** `data` must include only models the gateway can actually route (those with a provider prefix AND a pricing entry). A model present in `_PRICING` but unroutable should be filtered out so the catalog never advertises something `/v1/chat/completions` would 400 on.

## 5. Registration

In the app factory (where `chat.router` is included), add `app.include_router(openai_compat.router)`. Both routers share prefix `/v1` but distinct paths (`/chat` vs `/chat/completions`, `/models`), so no collision.

## 6. SDK usage (acceptance target)

```python
from openai import OpenAI
client = OpenAI(api_key="<gateway_project_key>", base_url="http://localhost:8000/v1")
client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role":"user","content":"hi"}])                       # non-stream
client.chat.completions.create(model="gpt-4o-mini", messages=[...],
    stream=True, stream_options={"include_usage": True})             # stream
client.models.list()                                                 # catalog
```
The SDK appends `/chat/completions` and `/models` to `base_url`, so `base_url` must end in `/v1`.

## 7. Test plan (httpx.MockTransport, no live providers)

Use `httpx.MockTransport` to fake the upstream OpenAI/Anthropic HTTP responses, injected into the adapters' `AsyncClient` (parametrize the adapter's client construction or patch `httpx.AsyncClient`). Drive the gateway via FastAPI `TestClient`/`httpx.ASGITransport`.

1. **Auth parity:** request with `Authorization: Bearer <key>` succeeds; with `x-api-key` succeeds; both absent → 401 with message naming both headers; bad scheme (`Basic ...`) → 401. Confirm `/v1/chat` still works with `x-api-key` (regression).
2. **Non-stream shape:** assert `object=="chat.completion"`, `id` starts `chatcmpl-`, `created` is int, `choices[0].message.role=="assistant"`, `finish_reason=="stop"`, `usage` keys present and `total==prompt+completion`.
3. **Usage recording:** after a non-stream call, query `UsageRecord` — one row, `status=="ok"`, `cost==compute_cost(...)`, `request_id` matches response `id`, tokens match mock usage. Confirms reuse of `record_usage`.
4. **Provider routing/BYOK:** `model="claude-3-5-sonnet-latest"` routes to Anthropic adapter (mock asserts request hit Anthropic mock with the project's decrypted key); missing `ProviderCredential` → 400 OpenAI error envelope.
5. **Streaming frames:** parse SSE; assert first chunk `delta=={"role":"assistant"}`, ≥1 `delta.content` chunk, a `finish_reason=="stop"` chunk, final `data: [DONE]`. With `stream_options.include_usage=true` assert exactly one extra chunk with `choices==[]` and a `usage` object; without it, assert no such chunk. Assert `Content-Type: text/event-stream`.
6. **Stream error path:** mock upstream 500 mid-stream → an error frame is emitted, `[DONE]` still sent, and a `UsageRecord` with `status=="error"` is written.
7. **Compat tolerance:** request with extra OpenAI fields (`top_p`, `presence_penalty`, `n`, `logit_bias`) → ignored, 200. `max_completion_tokens` honored when `max_tokens` absent. `content:null` coerced to `""`. `role:"tool"` → 400 invalid_request_error.
8. **`/v1/models`:** `object=="list"`; every `data[i].object=="model"`, has `id/owned_by/pricing`; `owned_by` matches routing; every advertised `id` is routable + priced. `GET /v1/models/{id}` returns the card; unknown id → 404 envelope. Unauthed → 401.
9. **Real SDK smoke (optional, offline):** point `openai.OpenAI(base_url=...)` at the ASGI app via a custom `http_client=httpx.Client(transport=ASGITransport(app))`; run `.create()` and `.create(stream=True)` and assert no SDK parse errors.

## 8. Files touched / added

- Edit: `app/middleware/auth.py` (Bearer support).
- Add: `app/schemas/openai_compat.py` (DTOs + translation helpers).
- Add: `app/routers/openai_compat.py` (`/v1/chat/completions`, `/v1/models`, `/v1/models/{id}`).
- Edit: `app/services/pricing.py` (add `get_price()` + `known_models()` public helpers; `_PRICING` stays private).
- Edit: app factory to `include_router(openai_compat.router)`.
- Add tests under existing test dir using `httpx.MockTransport` + ASGI transport.
- No new runtime deps; no DB schema/migration changes (UsageRecord reused as-is).

## Sources
- [OpenAI Chat Completions reference (object/choices/usage shape)](https://platform.openai.com/docs/api-reference/chat/create)
- [Chat completion chunk object (streaming delta shape)](https://platform.openai.com/docs/api-reference/chat-streaming/streaming)
- [Usage stats in streaming via stream_options.include_usage](https://community.openai.com/t/usage-stats-now-available-when-using-streaming-with-the-chat-completions-api-or-completions-api/738156)
- [OpenAIDevs announcement: include_usage final chunk with empty choices](https://x.com/OpenAIDevs/status/1787573348496773423)