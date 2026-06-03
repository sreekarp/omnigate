I have everything I need. The wire details are confirmed: endpoint `POST https://{endpoint}/openai/deployments/{deployment-id}/chat/completions?api-version=2024-10-21`, header `api-key`, response/stream shape identical to OpenAI chat completions, latest GA api-version is `2024-10-21`, and `stream_options.include_usage` yields a final chunk with `choices: []` and populated `usage`. Here is the spec.

---

# Spec: Azure OpenAI Provider Adapter (BYOK)

## 1. Summary

Add an `AzureOpenAIProvider` that talks to Azure OpenAI's chat-completions API. Wire format is byte-identical to OpenAI's `/v1/chat/completions` (request body, response JSON, SSE stream) — the only differences are: (a) URL contains the deployment and an `api-version` query param, (b) auth header is `api-key: {key}` not `Authorization: Bearer`, (c) the `model` field in the body is ignored (the URL deployment selects the model). Per-project BYOK already stores an encrypted key; Azure additionally needs three non-secret config values (endpoint, deployment, api-version), stored in a new nullable JSON `meta` column on `provider_credentials`.

Provider `name = "azure"`. Routing: models prefixed `azure/...` route here. Pricing and DB rows stay backward-compatible.

## 2. Wire details (verified against Microsoft Learn, GA api-version `2024-10-21`)

**Endpoint**
```
POST https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={api_version}
```
- `{resource}.openai.azure.com` is the host portion of the stored `endpoint`. We store the full base URL (`https://myres.openai.azure.com`) and append the path, so custom domains / sovereign clouds (`*.openai.azure.us`, `*.cognitiveservices.azure.com`) also work. Strip a trailing `/` from the stored endpoint before concatenating.
- `{deployment}` = the Azure deployment name (URL-encode it; deployment names allow `-` and `_`, but encode defensively).
- `{api_version}` default `2024-10-21` (latest GA). Override per-credential via `meta.api_version`.

**Headers**
```
api-key: {byok_key}
Content-Type: application/json
```
No `Authorization` header. (Azure also supports AAD bearer tokens, but BYOK = API key, so api-key only.)

**Request body** — identical to OpenAI. Reuse OpenAI's `_payload`:
```json
{
  "model": "<ignored by Azure, send anyway for harmless compat>",
  "messages": [{"role": "...", "content": "..."}],
  "stream": false,
  "max_tokens": 256,        // optional
  "temperature": 0.7        // optional
}
```
- Keep sending `model` (Azure ignores it; harmless and keeps `_payload` shared).
- For streaming, add `"stream_options": {"include_usage": true}` so the final chunk carries usage (see §5).

**Non-streaming response** — identical to OpenAI:
```json
{
  "id": "chatcmpl-...",
  "model": "gpt-4o-2024-08-06",
  "choices": [{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
  "usage": {"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}
}
```
Parse exactly as `OpenAIProvider.chat` does: `data["choices"][0]["message"]["content"]`, `data["usage"]`.

**Streaming response** — SSE, identical framing to OpenAI:
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hel"},"finish_reason":null}],"usage":null}
data: {"choices":[{"delta":{"content":"lo"}}]}
data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}   ← only when include_usage=true
data: [DONE]
```
Confirmed shapes from MS Learn:
- Every content chunk: `choices[0].delta.content` (may be absent on role-only / finish chunks).
- With `include_usage:true`: a **final chunk before `[DONE]` has `choices: []` and a populated `usage`**; all other chunks carry `usage: null`. If the stream is interrupted you may not receive it — handle absence gracefully.

## 3. DB change: `meta` JSON column on `provider_credentials`

Add a nullable JSON column to hold non-secret Azure config. JSON (not JSONB) keeps it generic; values are tiny. Use SQLAlchemy's portable `JSON` type so SQLite-based tests work too.

```python
# app/models/db.py  (inside ProviderCredential, after encrypted_key)
from sqlalchemy import JSON
...
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
```

`meta` shape for `provider == "azure"`:
```json
{"endpoint": "https://myres.openai.azure.com", "deployment": "gpt-4o-prod", "api_version": "2024-10-21"}
```
- `endpoint`, `deployment` required; `api_version` optional (defaults to `2024-10-21`).
- For `openai`/`anthropic`, `meta` stays `NULL` → fully backward compatible.

**Alembic migration** (new revision):
```python
def upgrade() -> None:
    op.add_column("provider_credentials",
        sa.Column("meta", sa.JSON(), nullable=True))
def downgrade() -> None:
    op.drop_column("provider_credentials", "meta")
```
Gotcha: existing rows get `meta = NULL` — adapters for openai/anthropic never read it, so no backfill needed.

## 4. Credentials service changes (`app/services/credentials.py`)

- Add `"azure"` to `SUPPORTED_PROVIDERS`.
- `set_provider_key(...)` gains `meta: dict | None = None`; persist on insert and update.
- Add `get_provider_credential(session, *, project_id, provider) -> tuple[str | None, dict | None]` returning `(decrypted_key, meta)` in one query (the chat router needs both for Azure). Keep the existing `get_provider_key` for openai/anthropic callers (backward compat).
- Validation in the key-management router (`POST /v1/keys`): when `provider == "azure"`, require `meta.endpoint` and `meta.deployment`; reject with 400 otherwise. Validate `endpoint` starts with `https://`.

Gotcha: `meta` is non-secret — safe to surface (endpoint/deployment/version) in `list_configured_providers`/dashboard, but never the key.

## 5. Deployment mapping (incoming model → Azure deployment)

Azure selects the model via the **deployment in the URL**, so the gateway must resolve an incoming model name to a deployment. Recommended layering, simplest-first:

1. **Single-deployment default (MVP):** the credential's `meta.deployment` is the deployment used for any `azure/*` model. Incoming `model="azure/gpt-4o"` → strip the `azure/` prefix for pricing/logging, but always hit `meta.deployment`. This matches "one BYOK Azure credential per project."
2. **Optional model→deployment map:** allow `meta.deployments` (a dict) for projects with multiple deployments under one resource:
   ```json
   {"endpoint":"...","api_version":"2024-10-21",
    "deployment":"gpt-4o-prod",
    "deployments":{"gpt-4o":"gpt-4o-prod","gpt-4o-mini":"gpt-4o-mini-prod"}}
   ```
   Resolution: strip `azure/` → look up in `meta.deployments` → fall back to `meta.deployment`. The adapter takes the resolved deployment as input; routing/credential lookup happens in the router.

**Routing** (`app/providers/registry.py`): add before the final raise:
```python
if name.startswith("azure/") or name.startswith("azure-"):
    return _azure()
```
Add an `@lru_cache _azure()` factory. Recommend canonical prefix `azure/` (slash) since deployment names use `-`.

**Pricing** (`app/services/pricing.py`): the deployed Azure model is an OpenAI model, so map the logical model to existing OpenAI prices. Add entries keyed by the canonical incoming name, e.g. `"azure/gpt-4o": (Decimal("0.0025"), Decimal("0.010"))`, reusing OpenAI rates. Unknown → cost 0 with warning (existing behavior, backward compatible). Note: Azure list prices can differ from OpenAI's — document that operators should tune `_PRICING` for Azure keys.

## 6. Chat router wiring (`app/routers/chat.py`)

Minimal, backward-compatible changes:
- After resolving `provider`, when `provider.name == "azure"`, fetch `(api_key, meta)` via `get_provider_credential` and validate `meta` has `endpoint` + `deployment`; resolve the deployment from `request.model` per §5. Construct the Azure provider call with that context.
- Keep the existing `get_provider_key` path for openai/anthropic.

Cleanest approach that avoids changing the `AbstractProvider.chat/stream` signatures: have the Azure adapter read its per-request context from an `AzureContext` passed at construction is awkward (providers are process-cached singletons). **Recommended:** make the Azure adapter *stateless* and pass the resolved endpoint/deployment/api_version by **packing them into the `api_key` is wrong** — instead, give the adapter dedicated methods and have the router build a small per-request `AzureTarget`. Two clean options:

- **Option A (recommended, no signature change):** Azure adapter is instantiated per-request with its target (do NOT lru_cache it). `registry.get_provider_for_model` returns the singleton type, but for azure the router builds `AzureOpenAIProvider(endpoint=..., deployment=..., api_version=...)` directly after reading `meta`. Keep `chat(request, api_key)` / `stream(request, api_key)` signatures intact. This preserves the `AbstractProvider` interface and the router's existing `provider.chat(request, api_key)` call.
- **Option B:** widen the interface with an optional `**context`. Heavier; avoid.

Use Option A. The router branch:
```python
provider = get_provider_for_model(request.model)  # may return the azure type/singleton
if provider.name == "azure":
    api_key, meta = await get_provider_credential(session, project_id=project.id, provider="azure")
    if not api_key or not meta or "endpoint" not in meta or "deployment" not in meta:
        raise HTTPException(400, "Azure credential incomplete (endpoint/deployment).")
    deployment = _resolve_deployment(request.model, meta)
    provider = AzureOpenAIProvider(
        endpoint=meta["endpoint"],
        deployment=deployment,
        api_version=meta.get("api_version", "2024-10-21"),
    )
else:
    api_key = await get_provider_key(session, project_id=project.id, provider=provider.name)
```
Everything downstream (latency, `record_usage`, streaming) is unchanged. Usage records store `provider="azure"`, `model=request.model` (the canonical `azure/...` name) — backward compatible with the existing schema.

## 7. Adapter sketch (`app/providers/azure_openai.py`)

```python
"""Azure OpenAI chat-completions adapter (async, httpx). BYOK api-key auth."""
import json
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import AbstractProvider, ProviderError
from app.schemas.chat import ChatRequest, ChatResponse, Usage

logger = get_logger(__name__)

_DEFAULT_API_VERSION = "2024-10-21"  # latest GA


class AzureOpenAIProvider(AbstractProvider):
    name = "azure"

    def __init__(self, *, endpoint: str, deployment: str,
                 api_version: str = _DEFAULT_API_VERSION) -> None:
        self._settings = get_settings()
        self._endpoint = endpoint.rstrip("/")
        self._deployment = deployment
        self._api_version = api_version

    def _url(self) -> str:
        return (
            f"{self._endpoint}/openai/deployments/{self._deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        if not api_key:
            raise ProviderError("No Azure OpenAI API key provided", status_code=400)
        return {"api-key": api_key, "Content-Type": "application/json"}

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": request.model,  # ignored by Azure; harmless
            "messages": [m.model_dump() for m in request.messages],
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(self, request: ChatRequest, api_key: str) -> ChatResponse:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    self._url(),
                    headers=self._headers(api_key),
                    json=self._payload(request, stream=False),
                )
            except httpx.HTTPError as exc:
                raise ProviderError(f"Azure OpenAI request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"Azure OpenAI error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ChatResponse(
            id=data.get("id", ""),
            provider=self.name,
            model=request.model,  # keep canonical azure/... name for logging/pricing
            content=choice["message"].get("content") or "",
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    async def stream(self, request: ChatRequest, api_key: str) -> AsyncIterator[str]:
        timeout = self._settings.request_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", self._url(),
                headers=self._headers(api_key),
                json=self._payload(request, stream=True),
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    raise ProviderError(
                        f"Azure OpenAI error {resp.status_code}: "
                        f"{body.decode(errors='replace')}",
                        status_code=resp.status_code,
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk_raw = line[len("data:"):].strip()
                    if chunk_raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_raw)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed Azure stream chunk")
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue  # final usage-only chunk (choices == [])
                    piece = choices[0].get("delta", {}).get("content")
                    if piece:
                        yield piece
```

Notes:
- Mirrors `OpenAIProvider` almost exactly — same parse logic — minimizing divergence. The `_payload` and parse code could be factored into a shared helper later, but duplicating keeps backward-compat risk near zero.
- `choices` guarded with `or []` so the include_usage final chunk (`choices: []`) and Azure's first-chunk content filter quirk don't crash.
- Streaming text path stays plain-text chunks as the router/`StreamingResponse` expects. (Like the existing code, the streamed usage chunk is observed but the router records zeros for streamed requests — leave that behavior unless §- a separate streaming-usage subsystem changes it.)

## 8. Gotchas

- **Azure ignores body `model`** — never rely on it to pick the model; the deployment in the URL wins. Sending it is harmless.
- **Header is `api-key`, lowercase, no Bearer.** Sending `Authorization` causes 401.
- **api-version is required** in the query string; omitting it returns 404/400.
- **First streaming chunk** from Azure often has `choices: []` (Azure prompt content-filter annotation chunk) before content begins — the `if not choices: continue` guard handles it. Don't assume `choices[0]` exists on every chunk.
- **`content` can be `None`** on role-only/finish/tool chunks — guard with `.get("content")` and truthiness (already done).
- **Endpoint normalization**: store and `rstrip("/")`; never let callers pass the full path. Custom-domain/sovereign endpoints work because we keep the full base URL.
- **Pricing drift**: Azure prices may differ from OpenAI; unknown models silently cost 0 (existing behavior) — operators should add `azure/...` entries.
- **Interrupted stream** → no final usage chunk; code must not require it.
- **`meta` JSON type**: use SQLAlchemy `JSON` (portable) not `JSONB`, so `httpx.MockTransport`/SQLite test fixtures work without Postgres.

## 9. Test plan (httpx.MockTransport, no network, no new deps)

Build an `httpx.MockTransport` handler that inspects the request and returns canned responses; inject by monkeypatching `httpx.AsyncClient` to use `transport=`. Tests:

1. **URL/auth construction**: assert request URL == `https://res.openai.azure.com/openai/deployments/dep1/chat/completions?api-version=2024-10-21`, header `api-key` present, no `Authorization`.
2. **Endpoint trailing-slash**: `endpoint="https://res.openai.azure.com/"` still produces a correct single-slash path.
3. **Non-stream parse**: mock OpenAI-shaped JSON → `ChatResponse.content`, `usage.{prompt,completion,total}_tokens`, `provider=="azure"`, `model==` canonical `azure/...`.
4. **Body `model` ignored / still sent**: assert payload includes `model` and `messages`, `stream=false`, optional `max_tokens`/`temperature` only when set.
5. **Stream happy path**: SSE bytes with role chunk + content deltas + final `choices:[]`+usage + `[DONE]`; assert concatenated yielded text == expected, and that the `choices:[]` chunk yields nothing and doesn't raise.
6. **Stream sets stream_options**: assert request body contains `stream_options.include_usage == true`.
7. **First-chunk `choices:[]` (content filter)**: leading empty-choices chunk is skipped without error.
8. **Error mapping**: 401 (bad key) and 404 (bad deployment/api-version) → `ProviderError` with matching `status_code`; missing key → 400.
9. **Routing**: `get_provider_for_model("azure/gpt-4o")` returns the azure provider; unknown stays a 400.
10. **Deployment resolution**: `_resolve_deployment` returns `meta.deployments[model]` when present, else `meta.deployment`.
11. **Credentials round-trip**: `set_provider_key(..., provider="azure", meta={...})` then `get_provider_credential` returns decrypted key + identical `meta`; openai/anthropic rows return `meta is None`.
12. **Migration smoke**: upgrade adds nullable `meta`; existing rows readable with `meta == None` (backward compat).
13. **Router integration**: incomplete Azure `meta` (missing `deployment`) → 400; complete meta → 200 and a `UsageRecord` row with `provider="azure"`, canonical model name, computed cost.

## 10. Files touched

- New: `C:\Sreekar\New folder\app\providers\azure_openai.py`
- Edit: `C:\Sreekar\New folder\app\providers\registry.py` (add `azure/` routing + `_azure()` factory; note: use Option A — router constructs the per-request instance, registry only needs to recognize the prefix)
- Edit: `C:\Sreekar\New folder\app\models\db.py` (`meta` JSON column on `ProviderCredential`)
- Edit: `C:\Sreekar\New folder\app\services\credentials.py` (`"azure"` supported, `meta` on upsert, new `get_provider_credential`)
- Edit: `C:\Sreekar\New folder\app\services\pricing.py` (`azure/...` price entries)
- Edit: `C:\Sreekar\New folder\app\routers\chat.py` (azure branch: fetch key+meta, resolve deployment, build adapter)
- Edit: key-management router (`POST /v1/keys`) — validate azure `meta`
- New: Alembic migration adding `provider_credentials.meta`
- New tests: `tests/providers/test_azure_openai.py` (+ credentials/routing cases)

Sources: [Azure OpenAI REST API reference (chat completions, GA 2024-10-21)](https://learn.microsoft.com/en-us/azure/foundry/openai/reference), [API version lifecycle (latest GA = 2024-10-21)](https://learn.microsoft.com/azure/ai-foundry/openai/api-version-lifecycle), [stream_options include_usage semantics](https://learn.microsoft.com/azure/ai-foundry/openai/reference-preview-latest)