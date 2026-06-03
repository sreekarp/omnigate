"""Standalone end-to-end smoke test against the live docker Postgres+Redis.

Reads .env (real docker DSNs), monkeypatches the OpenAI adapter so no network
is needed, and drives the full ASGI app. Run: python scripts/smoke_e2e.py
"""

import asyncio
import sys
from pathlib import Path

# Allow running directly: ensure the repo root (containing 'app') is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.main import app
from app.providers import openai as openai_mod
from app.schemas.chat import ChatResponse, StreamChunk, Usage

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, "-" if not detail else detail)


async def fake_chat(self, request, api_key):  # noqa: ANN001
    return ChatResponse(
        id="cmpl-smoke", provider=self.name, model=request.model,
        content="hello world", finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )


async def fake_stream(self, request, api_key):  # noqa: ANN001
    for piece in ["he", "llo ", "world"]:
        yield StreamChunk(text=piece)
    yield StreamChunk(usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12), finish_reason="stop")


async def main() -> None:
    openai_mod.OpenAIProvider.chat = fake_chat
    openai_mod.OpenAIProvider.stream = fake_stream

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/health/ready")
        check("health/ready", r.status_code == 200 and r.json()["status"] == "ready", str(r.json()))

        r = await c.post("/v1/signup", json={"email": "smoke@test.dev", "project_name": "Smoke"})
        check("signup", r.status_code == 201, str(r.status_code))
        api_key = r.json()["api_key"]
        H = {"x-api-key": api_key}

        # chat before key -> 400
        r = await c.post("/v1/chat", headers=H, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
        check("chat without provider key -> 400", r.status_code == 400, str(r.status_code))

        r = await c.post("/v1/keys", headers=H, json={"provider": "openai", "api_key": "sk-fake-smoke-key"})
        check("store openai key -> 204", r.status_code == 204, str(r.status_code))

        r = await c.post("/v1/chat", headers={**H, "x-user-id": "alice"}, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
        body = r.json() if r.status_code == 200 else r.text
        check("chat -> 200 + cost", r.status_code == 200 and body.get("cost_usd", 0) > 0, str(body)[:160])
        check("chat x-llmgw headers", r.headers.get("x-llmgw-provider") == "openai" and "x-ratelimit-limit" in r.headers, dict(r.headers).get("x-llmgw-provider", "?"))

        r = await c.get("/v1/me", headers=H)
        check("me lists openai", r.status_code == 200 and "openai" in r.json()["configured_providers"], str(r.json()))

        r = await c.get("/v1/models", headers=H)
        data = r.json().get("data", [])
        check("models list", r.status_code == 200 and any(m["id"] == "gpt-4o-mini" for m in data), f"{len(data)} models")

        # OpenAI-compatible endpoint (Bearer auth)
        r = await c.post("/v1/chat/completions", headers={"authorization": f"Bearer {api_key}"},
                         json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
        j = r.json() if r.status_code == 200 else {}
        check("openai-compat completion", r.status_code == 200 and j.get("object") == "chat.completion" and j["choices"][0]["message"]["content"] == "hello world", str(r.text)[:160])

        # streaming /v1/chat
        async with c.stream("POST", "/v1/chat", headers=H, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True}) as sr:
            chunks = [chunk async for chunk in sr.aiter_text()]
        text = "".join(chunks)
        check("stream text", "hello world" in text, repr(text)[:120])

        # OpenAI-compat streaming (SSE)
        async with c.stream("POST", "/v1/chat/completions", headers=H, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": True, "stream_options": {"include_usage": True}}) as sr:
            sse = "".join([chunk async for chunk in sr.aiter_text()])
        check("openai-compat SSE", "chat.completion.chunk" in sse and "data: [DONE]" in sse, repr(sse[-80:]))

        # named api key
        r = await c.post("/v1/keys/api", headers=H, json={"name": "ci-key"})
        check("create api key", r.status_code == 201 and r.json()["api_key"].startswith("llmg_"), str(r.status_code))
        new_key = r.json()["api_key"]
        r = await c.post("/v1/chat", headers={"x-api-key": new_key}, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
        check("chat with named key", r.status_code == 200, str(r.status_code))
        r = await c.get("/v1/keys/api", headers=H)
        check("list api keys", r.status_code == 200 and len(r.json()) == 1, str(r.status_code))

        # metrics (give the streaming finally a moment to commit)
        await asyncio.sleep(0.3)
        r = await c.get("/v1/metrics?range=24h&group_by=model", headers=H)
        j = r.json() if r.status_code == 200 else {}
        check("metrics totals", r.status_code == 200 and j["totals"]["requests"] >= 4, str(j.get("totals"))[:200])
        check("metrics breakdown", any(b["key"] == "gpt-4o-mini" for b in j.get("breakdown", [])), str(j.get("breakdown"))[:160])

        # prometheus
        r = await c.get("/metrics")
        check("prometheus", r.status_code == 200 and "llmgw_requests_total" in r.text, str(r.status_code))

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== {passed}/{len(results)} checks passed ====")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
