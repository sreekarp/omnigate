Both auth methods are valid (`?key=` query param and `x-goog-api-key` header). I have everything needed. Both verified against the official docs. Here is the spec.

---

# Gemini Provider Adapter — Implementation Spec

## 1. Wire format (Google Generative Language API, `v1beta`)

**Base URL:** `https://generativelanguage.googleapis.com/v1beta`

**Endpoints** (model name goes in the *path*, action after a colon):
- Non-stream: `POST /v1beta/models/{model}:generateContent`
- Stream (SSE): `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`

> The `?alt=sse` query param is **mandatory** for line-delimited `data:` SSE framing. Without it, `streamGenerateContent` returns a single streamed JSON *array* (much harder to parse incrementally). Always send `alt=sse`.

**Auth:** Use the **`x-goog-api-key: <key>`** header (cleaner than `?key=` query — keeps the BYOK secret out of URLs/logs). The `?key=` query form is also valid but avoid it. `Content-Type: application/json`.

**Gotcha — model in URL:** unlike OpenAI/Anthropic, the model is **not** in the body. Build the URL per-request. URL-safe model names (e.g. `gemini-2.0-flash`) need no escaping, but pass through `urllib.parse.quote(model, safe="")` defensively.

## 2. Request body

```json
{
  "contents": [
    { "role": "user",  "parts": [{ "text": "Hi" }] },
    { "role": "model", "parts": [{ "text": "Hello!" }] }
  ],
  "systemInstruction": { "parts": [{ "text": "You are terse." }] },
  "generationConfig": {
    "maxOutputTokens": 1024,
    "temperature": 0.7,
    "topP": 0.95,
    "stopSequences": ["\n\n"]
  }
}
```

**Role mapping (critical):** Gemini roles are only `"user"` and `"model"`.
- `user` → `user`
- `assistant` → **`model`**
- `system` → **NOT** a content role; collect all `system` messages and join with `\n\n` into top-level `systemInstruction.parts[0].text`. (Mirrors the existing Anthropic adapter's system-concat pattern.)

**generationConfig:** omit any key whose source field is `None`. Map `max_tokens → maxOutputTokens`, `temperature → temperature`. `topP`/`stopSequences` are not in the current `ChatRequest` schema — leave them out (or wire them later; do not add to `ChatRequest` for backward compat). Only emit the `generationConfig` object if at least one key is set.

**Gotcha — temperature range:** Gemini accepts `0.0–2.0`, same as the existing `ChatRequest` validator, so no clamping needed.

## 3. Non-stream response parsing

```json
{
  "candidates": [
    {
      "content": { "parts": [{ "text": "Hello!" }], "role": "model" },
      "finishReason": "STOP"
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 5,
    "candidatesTokenCount": 12,
    "totalTokenCount": 17
  }
}
```

- **content:** join *all* text parts of `candidates[0]`: `"".join(p.get("text","") for p in candidates[0]["content"]["parts"])`. Some parts (e.g. `thought`/`functionCall`) have no `text` key — `.get` guards that.
- **finishReason:** `candidates[0].get("finishReason")`. Values: `STOP`, `MAX_TOKENS`, `SAFETY`, `RECITATION`, `OTHER`.
- **Gotcha — blocked / empty candidates:** if a prompt is blocked, `candidates` may be **absent or empty** and `promptFeedback.blockReason` is set. Guard: if no candidates, raise `ProviderError(f"Gemini blocked: {block_reason}", status_code=400)`. Also handle `finishReason == "SAFETY"` with empty parts → return empty content rather than crashing.

### Usage (non-stream)
| Unified `Usage` field | Gemini field |
|---|---|
| `prompt_tokens` | `usageMetadata.promptTokenCount` |
| `completion_tokens` | `usageMetadata.candidatesTokenCount` |
| `total_tokens` | `usageMetadata.totalTokenCount` |

**Gotcha:** `candidatesTokenCount` can be **absent** when output is empty/blocked → default 0. `totalTokenCount` includes `thoughtsTokenCount`/`toolUsePromptTokenCount` if present, so `prompt+completion` may be **less** than `total` — trust `totalTokenCount` directly rather than re-summing.

**`id`:** Gemini returns **no top-level id**. There is a `responseId` on newer models — use `data.get("responseId", "")`. (The router overwrites cost; `id` is cosmetic.)

## 4. Streaming (SSE)

With `?alt=sse`, the body is OpenAI-style line framing: lines beginning `data: ` each containing a full `GenerateContentResponse` JSON object. **No `[DONE]` sentinel.** Reuse the existing `aiter_lines()` + `startswith("data:")` pattern from the OpenAI adapter.

Each chunk:
```
data: {"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"},"finishReason":null}]}

data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}]}

data: {"candidates":[{"content":{"parts":[{"text":"!"}]}, "finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":12,"totalTokenCount":17}}
```

- **Per chunk:** yield `candidates[0].content.parts[i].text` for each text part (a chunk may carry 0 parts).
- **Usage accumulation:** `usageMetadata` appears on the **final chunk** (and sometimes cumulatively on intermediate chunks). Strategy: on **every** chunk, if `usageMetadata` present, **overwrite** the stored totals (last-write-wins → final chunk values). `finishReason` likewise captured from whichever chunk carries it.
- **Gotcha — prefix matching:** match `line.startswith("data:")` then strip; Gemini SSE uses `data: ` (with space). The OpenAI adapter's `line[len("data:"):].strip()` handles both. Skip blank lines.

## 5. Model detection / registry

Add to `get_provider_for_model` (registry.py), before the final raise:
```python
if name.startswith("gemini"):
    return _gemini()
```
with an `@lru_cache def _gemini() -> GeminiProvider`. Detection prefix: **`gemini-*`** (covers `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-2.5-flash`, etc.). Lowercased compare already done.

Also add `"gemini"` to `SUPPORTED_PROVIDERS` in `credentials.py` and add Gemini rows to `_PRICING` in `pricing.py` (USD per 1k tokens), e.g.:
```python
"gemini-2.0-flash":      (Decimal("0.0001"),  Decimal("0.0004")),
"gemini-1.5-flash":      (Decimal("0.000075"),Decimal("0.0003")),
"gemini-1.5-pro":        (Decimal("0.00125"), Decimal("0.005")),
"gemini-2.5-pro":        (Decimal("0.00125"), Decimal("0.010")),
```
(Verify against current pricing at deploy time; unknown models already fall back to cost 0 with a warning.)

## 6. Enriched result (usage + finish_reason)

`stream()` must stay `AsyncIterator[str]` to keep `/v1/chat` streaming backward-compatible (`StreamingResponse(media_type="text/plain")`). To still surface usage+finish_reason without breaking the interface, add an **optional** out-param protocol that other refactored adapters will share (task #3):

```python
from dataclasses import dataclass, field

@dataclass
class StreamUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
```

`stream(request, api_key, usage_out: StreamUsage | None = None)` — populate `usage_out` in place from each chunk's `usageMetadata`/`finishReason`. The router can pass a `StreamUsage()` and, in the `finally:` block, record real tokens instead of zeros (this fixes the existing "stream records 0 tokens" limitation noted in chat.py). **Keep the param optional and keyword** so the existing `AbstractProvider.stream(request, api_key)` signature and current callers still work.

For non-stream, `ChatResponse` has no `finish_reason` field. Do **not** alter `ChatResponse` (backward compat / DB rows). Return finish_reason via the same `StreamUsage`-style internal object only if a caller needs it; otherwise it's discarded. (If the router later wants it, expose a private `chat_enriched()` returning `(ChatResponse, finish_reason)`.)

## 7. Adapter sketch

```python
"""Google Gemini (Generative Language API) adapter — async, httpx."""
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import AbstractProvider, ProviderError
from app.schemas.chat import ChatRequest, ChatResponse, Usage

logger = get_logger(__name__)
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


@dataclass
class StreamUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None


class GeminiProvider(AbstractProvider):
    name = "gemini"

    def __init__(self) -> None:
        self._settings = get_settings()

    def _headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise ProviderError("No Gemini API key provided", status_code=400)
        return {"x-goog-api-key": api_key, "Content-Type": "application/json"}

    def _url(self, model: str, *, stream: bool) -> str:
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{_BASE_URL}/models/{quote(model, safe='')}:{action}"
        return f"{url}?alt=sse" if stream else url

    def _payload(self, request: ChatRequest) -> dict:
        system_parts: list[str] = []
        contents: list[dict] = []
        for m in request.messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        payload: dict = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        gen: dict = {}
        if request.max_tokens is not None:
            gen["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            gen["temperature"] = request.temperature
        if gen:
            payload["generationConfig"] = gen
        return payload

    @staticmethod
    def _extract_text(candidate: dict) -> str:
        parts = candidate.get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    self._url(request.model, stream=False),
                    headers=self._headers(api_key),
                    json=self._payload(request),
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Gemini request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"Gemini error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(f"Gemini returned no candidates: {reason}", status_code=400)

        text = self._extract_text(candidates[0])
        um = data.get("usageMetadata", {})
        prompt = um.get("promptTokenCount", 0)
        completion = um.get("candidatesTokenCount", 0)
        total = um.get("totalTokenCount", prompt + completion)
        return ChatResponse(
            id=data.get("responseId", ""),
            provider=self.name,
            model=data.get("modelVersion", request.model),
            content=text,
            usage=Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total),
        )

    async def stream(
        self,
        request: ChatRequest,
        api_key: str,
        usage_out: StreamUsage | None = None,
    ) -> AsyncIterator[str]:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self._url(request.model, stream=True),
                headers=self._headers(api_key),
                json=self._payload(request),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"Gemini error {resp.status_code}: {body.decode(errors='replace')}",
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed Gemini stream chunk")
                        continue
                    candidates = chunk.get("candidates") or []
                    if candidates:
                        piece = self._extract_text(candidates[0])
                        if piece:
                            yield piece
                        fr = candidates[0].get("finishReason")
                        if fr and usage_out is not None:
                            usage_out.finish_reason = fr
                    um = chunk.get("usageMetadata")
                    if um and usage_out is not None:  # last-write-wins → final totals
                        usage_out.prompt_tokens = um.get("promptTokenCount", 0)
                        usage_out.completion_tokens = um.get("candidatesTokenCount", 0)
                        usage_out.total_tokens = um.get("totalTokenCount", 0)
```

## 8. Test plan (httpx.MockTransport, no live calls)

1. **URL/auth build:** assert non-stream URL = `.../models/gemini-2.0-flash:generateContent`, stream URL ends `:streamGenerateContent?alt=sse`, and request has header `x-goog-api-key` (not `?key=`), no `Authorization`.
2. **Payload role mapping:** input messages `[system, user, assistant, user]` → body has `systemInstruction.parts[0].text` = joined system text, `contents` roles = `["user","model","user"]`, each with `parts[0].text`. Assert `system` never appears as a content role.
3. **generationConfig omission:** `max_tokens=None, temperature=None` → no `generationConfig` key. With values set → `maxOutputTokens`/`temperature` present.
4. **Non-stream parse:** MockTransport returns canned `candidates[0].content.parts` (two text parts) + `usageMetadata` → assert `content` is concatenation, `usage.prompt_tokens=promptTokenCount`, `completion_tokens=candidatesTokenCount`, `total_tokens=totalTokenCount`.
5. **Blocked prompt:** response with empty `candidates` + `promptFeedback.blockReason="SAFETY"` → raises `ProviderError(status_code=400)`.
6. **HTTP error:** 429 body → `ProviderError` with `status_code==429`, message includes body text.
7. **Streaming:** MockTransport yields 3 `data: {...}` lines (last carries `usageMetadata` + `finishReason=STOP`, no `[DONE]`). Collect yielded pieces → equals full text; passed `StreamUsage` has correct token counts and `finish_reason="STOP"`.
8. **Stream malformed chunk:** inject one non-JSON `data:` line → skipped (logged), stream still completes.
9. **Registry:** `get_provider_for_model("gemini-1.5-pro").name == "gemini"`; unknown model still raises 400.
10. **Pricing:** `compute_cost("gemini-2.0-flash", 1000, 1000)` matches table; unknown `gemini-x` → 0 with warning.
11. **Backward compat:** calling `stream(request, api_key)` *without* `usage_out` works unchanged (router's existing zero-recording path still valid).

## Key files to create / touch
- **Create:** `C:\Sreekar\New folder\app\providers\gemini.py`
- **Edit:** `C:\Sreekar\New folder\app\providers\registry.py` (add `_gemini()` + `gemini-` branch)
- **Edit:** `C:\Sreekar\New folder\app\services\credentials.py` (`SUPPORTED_PROVIDERS += ("gemini",)`)
- **Edit:** `C:\Sreekar\New folder\app\services\pricing.py` (add `gemini-*` rows)
- **Optional (task #3):** `C:\Sreekar\New folder\app\routers\chat.py` (pass `StreamUsage` into `_stream` to record real streamed tokens — also fixes the existing OpenAI/Anthropic zero-token streaming gap)
- **Tests:** `tests/providers/test_gemini.py`

Sources: [Generating content — Gemini API](https://ai.google.dev/api/generate-content), [GenerateContentResponse — Vertex AI REST](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse), [Streaming REST cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Streaming_REST.ipynb)