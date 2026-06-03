I have everything I need. Returning the implementation spec.

---

# Spec: Accurate Streaming Token-Usage Capture + Enriched Provider Interface

## Goal
Make streamed `/v1/chat` requests record **real** prompt/completion tokens and cost (currently hard-coded to 0 in `chat.py:_stream`), uniformly across OpenAI, Anthropic, Azure OpenAI, and Gemini, without breaking the existing wire contract (plain-text `StreamingResponse`) or existing `UsageRecord` rows.

## Recommended design: typed `StreamChunk` + thin back-compat shim

The cleanest option is to change the provider contract so `stream()` yields a **typed `StreamChunk`** (text + optional usage + optional finish_reason) instead of bare `str`. This is strictly more expressive than a side-channel accumulator (no shared mutable state, no ordering hazards, trivially testable). Back-compat for any caller that wants raw text is preserved by:
1. keeping `StreamChunk.text` as the only thing the router writes to the wire (so the HTTP response stays identical), and
2. providing a `stream_text()` default helper on the base class that adapts `stream()` back to a `str` iterator.

### New schema types — `app/schemas/chat.py`

Add to the existing file (do not modify `Usage`/`ChatResponse` field names — keep DB and `/v1/chat` JSON stable):

```python
from pydantic import BaseModel

class StreamChunk(BaseModel):
    """One streaming increment from a provider adapter."""
    text: str = ""                      # delta text to forward to the client (may be "")
    usage: Usage | None = None          # populated on the FINAL usage-bearing chunk only
    finish_reason: str | None = None    # provider-native stop reason, when known
    model: str | None = None            # provider-confirmed model id, when known
```

Use Pydantic for consistency with the rest of the schema module; a frozen `@dataclass(slots=True)` is also fine and slightly cheaper, but Pydantic keeps imports uniform. Make it `model_config = {"frozen": False}` default (no need to freeze).

`Usage` already has `prompt_tokens` / `completion_tokens` / `total_tokens` — reuse it verbatim so `compute_cost(model, prompt_tokens, completion_tokens)` works unchanged.

### Enriched interface — `app/providers/base.py`

```python
from collections.abc import AsyncIterator
from app.schemas.chat import ChatRequest, ChatResponse, StreamChunk

class AbstractProvider(ABC):
    name: str

    @abstractmethod
    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse: ...

    @abstractmethod
    def stream(self, request: ChatRequest, api_key: str) -> AsyncIterator[StreamChunk]:
        """Async generator yielding StreamChunk. The terminal usage chunk
        (text may be empty) carries the final Usage."""
        ...

    async def stream_text(self, request: ChatRequest, api_key: str) -> AsyncIterator[str]:
        """Back-compat adapter: yields only non-empty text deltas."""
        async for chunk in self.stream(request, api_key):
            if chunk.text:
                yield chunk.text
```

**Gotcha (do not regress this):** the docstring on the current `stream()` notes it is intentionally *not* declared `async` because implementations are `async def` generators. Keep that exactly — `stream` stays a normal `def`/abstractmethod whose concrete impls are `async def` generators. `stream_text` *is* `async def` (it awaits the inner generator). Do not make the base `stream` async or you turn every adapter into a coroutine-returning-generator.

---

## Per-provider wire details

All adapters: parse the usage-bearing event, **accumulate**, and `yield StreamChunk(usage=...)` once before the generator ends. Never assume the usage chunk also carries text.

### OpenAI / Azure OpenAI — `app/providers/openai.py`

Add to `_payload(...)` **only when `stream=True`**:
```python
payload["stream_options"] = {"include_usage": True}
```
- Verified: with `include_usage`, every normal chunk has `usage: null`; **one extra final chunk** arrives with `"choices": []` and a populated `"usage": {"prompt_tokens", "completion_tokens", "total_tokens"}`. (OpenAI announced 2024-05; behavior unchanged.)
- **Gotchas:**
  - The final chunk has `choices == []` → the current code does `chunk["choices"][0]` which will `IndexError`. Guard: `choices = chunk.get("choices") or []; if choices: delta = choices[0].get("delta", {})`.
  - Read usage on **any** chunk where `chunk.get("usage")` is truthy (don't gate on `[DONE]`). The usage chunk typically arrives *just before* `data: [DONE]`.
  - Some Azure API versions emit an early empty-`choices` chunk carrying only `prompt_filter_results`/`content_filter` and no usage — already handled by the "choices empty → skip text" guard; only set usage when `usage` key is present and non-null.
  - Azure uses the same body; only base URL/auth differ (`api-key` header, `?api-version=` query). The usage logic is identical.

Revised generator core:
```python
async for line in resp.aiter_lines():
    if not line or not line.startswith("data:"):
        continue
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        break
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed OpenAI stream chunk")
        continue
    choices = chunk.get("choices") or []
    if choices:
        piece = choices[0].get("delta", {}).get("content")
        finish = choices[0].get("finish_reason")
        if piece or finish:
            yield StreamChunk(text=piece or "", finish_reason=finish)
    u = chunk.get("usage")
    if u:
        yield StreamChunk(
            usage=Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            ),
            model=chunk.get("model"),
        )
```

### Anthropic — `app/providers/anthropic.py`

Usage is split across two events (verified against current Anthropic streaming docs, `anthropic-version: 2023-06-01`):
- `message_start` → `data.message.usage.input_tokens` (this is the prompt-token count; `output_tokens` here is a small priming value like `1`/`2` — ignore it). May also include `cache_creation_input_tokens` / `cache_read_input_tokens`.
- `content_block_delta` with `delta.type == "text_delta"` → `delta.text` (the streamed text; keep existing behavior).
- `message_delta` → `usage.output_tokens` — **cumulative**, so the last `message_delta` holds the final completion-token total. (Docs explicitly warn it is cumulative.)
- `message_stop` → emit the accumulated `StreamChunk(usage=...)`.

Accumulate `input_tokens` from `message_start`, keep overwriting `output_tokens` from each `message_delta` (cumulative → last wins), emit once at `message_stop`:
```python
input_tokens = 0
output_tokens = 0
async for line in resp.aiter_lines():
    if not line or not line.startswith("data:"):
        continue
    try:
        event = json.loads(line[len("data:"):].strip())
    except json.JSONDecodeError:
        logger.warning("Skipping malformed Anthropic stream chunk")
        continue
    etype = event.get("type")
    if etype == "message_start":
        input_tokens = event.get("message", {}).get("usage", {}).get("input_tokens", 0)
    elif etype == "content_block_delta":
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta" and delta.get("text"):
            yield StreamChunk(text=delta["text"])
    elif etype == "message_delta":
        u = event.get("usage") or {}
        if "output_tokens" in u:
            output_tokens = u["output_tokens"]          # cumulative; last wins
        fr = event.get("delta", {}).get("stop_reason")
        if fr:
            finish_reason = fr  # stash, attach to final usage chunk
    elif etype == "message_stop":
        yield StreamChunk(
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=locals().get("finish_reason"),
        )
    elif etype == "error":
        msg = event.get("error", {}).get("message", "stream error")
        raise ProviderError(f"Anthropic stream error: {msg}", status_code=502)
```
- **Gotchas:** ignore `ping` events; ignore the `output_tokens` in `message_start`; handle `error` events mid-stream (e.g. `overloaded_error` → maps to 529). `message_stop` data has no usage of its own — that's why we accumulate.

### Gemini — new adapter `app/providers/gemini.py` (Task #3 adds it; spec it here)

- Endpoint: `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={api_key}` (use `?alt=sse` so the response is line-delimited `data:` SSE — without it Gemini returns a streamed JSON array which is far messier to parse).
- Each SSE `data:` line is a `GenerateContentResponse` chunk. Text is at `candidates[0].content.parts[*].text`.
- `usageMetadata` appears on chunks with fields `promptTokenCount`, `candidatesTokenCount`, `totalTokenCount` (these are **cumulative/absolute**, not deltas). The **final** chunk carries the authoritative totals — overwrite each time you see `usageMetadata`, last wins.
- Map: `prompt_tokens = promptTokenCount`, `completion_tokens = candidatesTokenCount`, `total_tokens = totalTokenCount`. (Note: on the Gemini Developer API `candidatesTokenCount` includes thinking tokens; that's correct for billing. `totalTokenCount` may exceed prompt+candidates when `thoughtsTokenCount` is present — prefer the server's `totalTokenCount` for `total_tokens` rather than recomputing.)
- Register the prefix `gemini` in `app/providers/registry.py`.

---

## Router changes — `app/routers/chat.py`

Replace the zero-recording `_stream` with one that consumes `StreamChunk`, forwards `.text` to the wire, accumulates the final `usage`, and records real tokens+cost in `finally`. **Keep `media_type="text/plain"` and the `x-request-id` header unchanged** (back-compat).

```python
async def _stream(request, ctx, session, provider, api_key, request_id) -> StreamingResponse:
    project = ctx.project
    started = time.perf_counter()

    async def event_generator() -> AsyncIterator[str]:
        status_str = "ok"
        final_usage = Usage()              # from app.schemas.chat
        final_model = request.model
        try:
            async for chunk in provider.stream(request, api_key):
                if chunk.usage is not None:
                    final_usage = chunk.usage
                if chunk.model:
                    final_model = chunk.model
                if chunk.text:
                    yield chunk.text       # wire output unchanged: bare text
        except ProviderError as exc:
            status_str = "error"
            logger.warning("Streaming error for request %s: %s", request_id, exc.message)
            yield f"\n[error] {exc.message}"
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            cost = compute_cost(final_model, final_usage.prompt_tokens,
                                final_usage.completion_tokens)
            await record_usage(
                session,
                org_id=project.org_id, project_id=project.id, user_id=ctx.user_id,
                provider=provider.name, model=final_model,
                prompt_tokens=final_usage.prompt_tokens,
                completion_tokens=final_usage.completion_tokens,
                cost=cost, status=status_str, latency_ms=latency_ms,
                request_id=request_id,
            )

    return StreamingResponse(event_generator(), media_type="text/plain",
                             headers={"x-request-id": request_id})
```

- `record_usage` already computes `total_tokens = prompt+completion`, so no schema/DB change is needed.
- **Client-disconnect gotcha:** if the client aborts, the `async for` raises `asyncio.CancelledError` / `httpx.StreamClosed`. The `finally` still runs and records whatever usage accumulated so far (likely 0 if disconnect was early, partial if mid-stream). This matches OpenAI's documented caveat that an interrupted stream may never deliver the usage chunk. Acceptable — record `status="ok"` with whatever we have; optionally set `status="incomplete"` if `final_usage.total_tokens == 0` and `CancelledError` was seen (the `status` column is free-form `String(32)`).
- **DB session gotcha:** the `finally` runs after the StreamingResponse body is fully sent, which is *after* the FastAPI request handler returns. The `Depends(get_session)` session must remain open for the lifetime of the generator. Verify `get_session` doesn't close on handler return before the body streams; if it does, acquire a fresh session inside `event_generator()` via the sessionmaker instead of the injected one. This is the single most likely runtime bug — call it out in the test plan.

---

## Test plan (httpx.MockTransport, no live network)

Build a `MockTransport` whose handler returns `httpx.Response(200, stream=...)` with a hand-crafted SSE byte body, and inject the client (add an optional `client`/`transport` seam to each adapter, or patch `httpx.AsyncClient` in the test). Then:

1. **OpenAI usage chunk** — feed N text chunks (`usage: null`) then a final `{"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}}` then `data: [DONE]`. Assert: concatenated text correct; exactly one `StreamChunk.usage` with `(11,22,33)`; no `IndexError` on empty `choices`.
2. **OpenAI early empty-choices/filter chunk** — first chunk `choices:[]` with no `usage` → produces no text and no usage; doesn't crash.
3. **Anthropic split usage** — `message_start` (`input_tokens:25, output_tokens:1`), several `text_delta`s, two `message_delta`s (`output_tokens:7` then `15`), `message_stop`. Assert final usage `(25, 15, 40)` (cumulative last-wins, `message_start.output_tokens` ignored).
4. **Anthropic error event** — inject `event: error` mid-stream → `ProviderError(status_code=502)` raised; router writes `\n[error] ...` and records `status="error"`.
5. **Gemini** — SSE chunks with text parts and a final `usageMetadata: {promptTokenCount:5, candidatesTokenCount:9, totalTokenCount:14}` → usage `(5,9,14)`; uses server `totalTokenCount` for `total_tokens`.
6. **Router integration** — fake provider yielding `[StreamChunk(text="a"), StreamChunk(text="b"), StreamChunk(usage=Usage(10,20,30))]`; assert response body == `"ab"`, `media_type=="text/plain"`, `x-request-id` header present, and one `UsageRecord` written with `prompt_tokens=10, completion_tokens=20, total_tokens=30, cost==compute_cost(model,10,20)`, `status="ok"`.
7. **Back-compat `stream_text`** — assert it yields only non-empty text and drops the terminal usage-only chunk.
8. **Disconnect** — generator raising `asyncio.CancelledError` after first chunk still records a usage row in `finally` (assert row exists; tokens = whatever accumulated). Verifies the session-lifetime concern.

## Files touched
- `C:\Sreekar\New folder\app\schemas\chat.py` — add `StreamChunk` (new), `Usage`/`ChatResponse` unchanged.
- `C:\Sreekar\New folder\app\providers\base.py` — `stream()` returns `AsyncIterator[StreamChunk]`; add `stream_text()` default.
- `C:\Sreekar\New folder\app\providers\openai.py` — add `stream_options`, empty-`choices` guard, emit usage chunk (Azure reuses).
- `C:\Sreekar\New folder\app\providers\anthropic.py` — accumulate `message_start`/`message_delta`, emit terminal usage chunk, handle `error`.
- `C:\Sreekar\New folder\app\providers\gemini.py` — new adapter (`?alt=sse`, `usageMetadata`).
- `C:\Sreekar\New folder\app\providers\registry.py` — add `gemini` prefix.
- `C:\Sreekar\New folder\app\routers\chat.py` — rewrite `_stream` to accumulate usage and record real tokens+cost; wire format unchanged.
- No DB migration, no `pricing.py`/`usage.py` signature changes, no new runtime deps.

## Key accuracy facts (verified)
- OpenAI/Azure: `stream_options:{"include_usage":true}` → extra final chunk, `choices:[]`, populated `usage`; non-final chunks have `usage:null`; missing if stream interrupted. ([OpenAI announcement](https://x.com/OpenAIDevs/status/1787573348496773423), [API ref](https://platform.openai.com/docs/api-reference/chat-streaming/streaming))
- Anthropic: `message_start.message.usage.input_tokens` = prompt; `message_delta.usage.output_tokens` = **cumulative** completion (last wins); `message_start.output_tokens` is priming-only; emit at `message_stop`. ([Anthropic streaming docs](https://platform.claude.com/docs/en/build-with-claude/streaming))
- Gemini: `streamGenerateContent?alt=sse`; `usageMetadata.{promptTokenCount,candidatesTokenCount,totalTokenCount}` are absolute, final chunk authoritative. ([Gemini generate-content API](https://ai.google.dev/api/generate-content), [token docs](https://ai.google.dev/gemini-api/docs/tokens))