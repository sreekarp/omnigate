"""In-process completion engine — the litellm-style heart of the SDK.

Public entry points :func:`completion` / :func:`acompletion` call providers
directly (no hosted gateway), composing the resilience features around each
call:

    cache lookup -> spend guard -> [per candidate: resolve -> breaker -> retry]
                 -> cost -> cache store -> callbacks

Fallback models are tried in order; each may resolve to a different provider.
The circuit breaker counts only transient failures (5xx/timeout); client errors
(4xx) mark the provider alive and skip to the next candidate. Streaming uses the
primary model only and is not retried once bytes flow.

Sync and async share the pure provider specs and all helpers below; only the two
executors (``httpx.Client`` vs ``httpx.AsyncClient``) and the retry driver
differ. State that is intentionally process-global (the circuit breaker and the
spend accumulator) can be cleared with :func:`reset_state` (used by tests).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import Any, AsyncIterator, Iterator, Optional, Union

import httpx

from .cache import cache_key, get_cache, should_cache
from .callbacks import CallbackEvent, fire_failure, fire_success
from .callbacks import register as _register_callback
from .circuit_breaker import InMemoryCircuitBreaker
from .config import EngineConfig, get_config, set_config
from .exceptions import APIError, AuthError, BudgetExceededError, ProviderError, RateLimitError
from .keys import resolve_key, resolve_target, set_azure, set_override
from .models import ChatRequest, ChatResponse, StreamChunk, coerce_messages
from .pricing import compute_cost
from .providers.openai import parse_retry_after
from .providers.registry import provider_name_for_model, spec_for_model
from .resilience import RetryPolicy, is_retryable, retry_async, retry_sync

logger = logging.getLogger("omnigate")

_MAX_CANDIDATES = 6  # primary + up to 5 fallbacks


# ---------------------------------------------------------------------------
# process-global state (breaker + spend accumulator)
# ---------------------------------------------------------------------------

_breaker: Optional[InMemoryCircuitBreaker] = None
_breaker_sig: Optional[tuple[int, float]] = None
_spend_total: float = 0.0


def reset_state() -> None:
    """Clear the process-global circuit breaker and spend accumulator (tests)."""
    global _breaker, _breaker_sig, _spend_total
    _breaker = None
    _breaker_sig = None
    _spend_total = 0.0


def _get_breaker(cfg: EngineConfig) -> InMemoryCircuitBreaker:
    global _breaker, _breaker_sig
    sig = (cfg.circuit_breaker_fail_threshold, cfg.circuit_breaker_cooldown)
    if _breaker is None or _breaker_sig != sig:
        _breaker = InMemoryCircuitBreaker(
            fail_threshold=sig[0], cooldown_seconds=sig[1]
        )
        _breaker_sig = sig
    return _breaker


def _check_spend(cfg: EngineConfig) -> None:
    """Raise :class:`BudgetExceededError` if the local spend cap is reached."""
    if cfg.max_spend_usd is not None and _spend_total >= cfg.max_spend_usd:
        raise BudgetExceededError(
            f"Local spend cap of ${cfg.max_spend_usd:.6f} reached "
            f"(spent ${_spend_total:.6f}).",
            status_code=402,
        )


def _add_spend(cost: float) -> None:
    global _spend_total
    _spend_total += cost


# ---------------------------------------------------------------------------
# request building / config
# ---------------------------------------------------------------------------

_OPTIONAL_FIELDS = (
    "temperature", "max_tokens", "top_p", "stop",
    "presence_penalty", "frequency_penalty", "seed", "cache",
)


def _prepare(model: str, messages: Any, *, stream: bool, fallbacks, **optional) -> ChatRequest:
    data: dict[str, Any] = {
        "model": model,
        "messages": coerce_messages(messages),
        "stream": stream,
    }
    for name in _OPTIONAL_FIELDS:
        value = optional.get(name)
        if value is not None:
            data[name] = value
    if fallbacks:
        data["fallback_models"] = list(fallbacks)
    return ChatRequest(**data)


def _effective_config(timeout: Optional[float], num_retries: Optional[int]) -> EngineConfig:
    cfg = get_config()
    if timeout is None and num_retries is None:
        return cfg
    overrides: dict[str, Any] = {}
    if timeout is not None:
        overrides["timeout"] = timeout
    if num_retries is not None:
        # litellm semantics: num_retries is RETRIES; total attempts = retries + 1.
        overrides["retry_max_attempts"] = max(1, num_retries + 1)
    return dataclasses.replace(cfg, **overrides)


def _candidates(request: ChatRequest) -> list[str]:
    out: list[str] = []
    for model in [request.model, *request.fallback_models]:
        if model and model not in out:
            out.append(model)
    return out[:_MAX_CANDIDATES]


def _resolve(model: str, *, api_key, api_base, api_version):
    spec, provider_name = spec_for_model(model)
    key = resolve_key(provider_name, api_key)
    target = resolve_target(provider_name, model, api_base=api_base, api_version=api_version)
    return spec, provider_name, key, target


# ---------------------------------------------------------------------------
# error classification + SSE framing
# ---------------------------------------------------------------------------

def _classify(provider: str, status: int, body: str, retry_after_hdr: Optional[str]) -> APIError:
    retry_after = parse_retry_after(retry_after_hdr) if status in (429, 503) else None
    msg = f"{provider} error {status}: {body}"
    if status in (401, 403):
        return AuthError(msg, status_code=status, retry_after=retry_after)
    if status == 429:
        return RateLimitError(msg, retry_after=retry_after, status_code=status)
    return ProviderError(msg, status_code=status, retry_after=retry_after)


def _sse_data(line: str) -> Optional[str]:
    """Extract the payload of a ``data:`` SSE line, or ``None`` for other lines."""
    if not line or not line.startswith("data:"):
        return None
    return line[len("data:"):].strip()


# ---------------------------------------------------------------------------
# executors (the only sync/async difference)
# ---------------------------------------------------------------------------

def _call_sync(client, spec, req, key, target) -> ChatResponse:
    try:
        resp = client.post(
            spec.url(req, target),
            headers=spec.headers(key),
            json=spec.build_payload(req, stream=False),
        )
    except httpx.HTTPError as exc:
        raise ProviderError(f"{spec.name} request failed: {exc}", status_code=502) from exc
    if resp.status_code >= 400:
        raise _classify(spec.name, resp.status_code, resp.text, resp.headers.get("Retry-After"))
    return spec.parse_response(resp.json(), req.model)


async def _call_async(client, spec, req, key, target) -> ChatResponse:
    try:
        resp = await client.post(
            spec.url(req, target),
            headers=spec.headers(key),
            json=spec.build_payload(req, stream=False),
        )
    except httpx.HTTPError as exc:
        raise ProviderError(f"{spec.name} request failed: {exc}", status_code=502) from exc
    if resp.status_code >= 400:
        raise _classify(spec.name, resp.status_code, resp.text, resp.headers.get("Retry-After"))
    return spec.parse_response(resp.json(), req.model)


def _stream_sync(client, spec, req, key, target) -> Iterator[StreamChunk]:
    with client.stream(
        "POST",
        spec.stream_url(req, target),
        headers=spec.headers(key),
        json=spec.build_payload(req, stream=True),
    ) as resp:
        if resp.status_code >= 400:
            body = resp.read().decode(errors="replace")
            raise _classify(spec.name, resp.status_code, body, resp.headers.get("Retry-After"))
        state = spec.stream_begin()
        for line in resp.iter_lines():
            data = _sse_data(line)
            if data is None or data == "":
                continue
            if data == "[DONE]":
                break
            try:
                raw = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("skipping malformed %s stream chunk", spec.name)
                continue
            for chunk in spec.stream_feed(state, raw):
                yield chunk
        terminal = spec.stream_end(state)
        if terminal is not None:
            yield terminal


async def _stream_async(client, spec, req, key, target) -> AsyncIterator[StreamChunk]:
    async with client.stream(
        "POST",
        spec.stream_url(req, target),
        headers=spec.headers(key),
        json=spec.build_payload(req, stream=True),
    ) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode(errors="replace")
            raise _classify(spec.name, resp.status_code, body, resp.headers.get("Retry-After"))
        state = spec.stream_begin()
        async for line in resp.aiter_lines():
            data = _sse_data(line)
            if data is None or data == "":
                continue
            if data == "[DONE]":
                break
            try:
                raw = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("skipping malformed %s stream chunk", spec.name)
                continue
            for chunk in spec.stream_feed(state, raw):
                yield chunk
        terminal = spec.stream_end(state)
        if terminal is not None:
            yield terminal


# ---------------------------------------------------------------------------
# per-success finalisation shared by sync + async
# ---------------------------------------------------------------------------

def _finalize(result: ChatResponse, model: str, is_fallback: bool, t0: float) -> None:
    cost = compute_cost(
        result.model or model, result.usage.prompt_tokens, result.usage.completion_tokens
    )
    result.cost_usd = float(cost)
    result.fallback_used = is_fallback
    result.latency_ms = int((time.perf_counter() - t0) * 1000)


def _account_and_cache(result, req, cfg, cache, ckey, cache_eligible, is_fallback) -> None:
    _add_spend(result.cost_usd)
    if cache_eligible and ckey is not None and not is_fallback:
        cache.set(ckey, result, cfg.cache_ttl)


def _emit_success(result: ChatResponse, provider: str, *, cached: bool, latency_ms: int) -> None:
    fire_success(CallbackEvent(
        model=result.model, provider=provider, usage=result.usage,
        cost_usd=result.cost_usd, latency_ms=latency_ms,
        cached=cached, fallback_used=result.fallback_used,
    ))


def _fail(last_error: Optional[APIError]) -> ChatResponse:
    err = last_error or ProviderError(
        "No provider candidate could handle the request", status_code=502
    )
    fire_failure(CallbackEvent(
        model="", provider="", usage=None, cost_usd=0.0, latency_ms=0,
        cached=False, fallback_used=False, exception=err,
    ))
    raise err


def _cache_lookup(req, cfg, cache):
    """Return (hit | None, ckey | None, eligible, primary_provider)."""
    eligible = should_cache(req, enabled=cfg.cache_enabled)
    if not eligible:
        return None, None, False, None
    try:
        provider = provider_name_for_model(req.model)
        ckey = cache_key(provider, req)
    except APIError:
        return None, None, False, None
    hit = cache.get(ckey)
    if hit is not None:
        hit.cost_usd = 0.0
    return hit, ckey, True, provider


# ---------------------------------------------------------------------------
# orchestrators
# ---------------------------------------------------------------------------

def _run(req, *, config, transport, api_key, api_base, api_version) -> ChatResponse:
    cache = get_cache()
    hit, ckey, cache_eligible, primary = _cache_lookup(req, config, cache)
    if hit is not None:
        _emit_success(hit, primary or hit.provider, cached=True, latency_ms=0)
        return hit

    _check_spend(config)
    breaker = _get_breaker(config)
    policy = RetryPolicy.from_config(config)
    last_error: Optional[APIError] = None

    client = httpx.Client(timeout=config.timeout, transport=transport)
    try:
        for model in _candidates(req):
            is_fallback = model != req.model
            try:
                spec, provider, key, target = _resolve(
                    model, api_key=api_key, api_base=api_base, api_version=api_version
                )
            except APIError as exc:
                last_error = exc
                continue
            if config.circuit_breaker_enabled and not breaker.allow(provider):
                last_error = ProviderError(f"{provider} circuit is open", status_code=503)
                continue
            req_for_model = req if not is_fallback else req.model_copy(update={"model": model})
            t0 = time.perf_counter()
            try:
                result = retry_sync(
                    lambda: _call_sync(client, spec, req_for_model, key, target),
                    policy=policy,
                )
            except APIError as exc:
                last_error = exc
                if config.circuit_breaker_enabled:
                    (breaker.record_failure if is_retryable(exc) else breaker.record_success)(provider)
                continue
            except Exception as exc:  # noqa: BLE001 - never leak raw provider errors
                last_error = ProviderError(f"{provider} call failed: {exc}", status_code=502)
                if config.circuit_breaker_enabled:
                    breaker.record_failure(provider)
                continue
            if config.circuit_breaker_enabled:
                breaker.record_success(provider)
            _finalize(result, model, is_fallback, t0)
            _account_and_cache(result, req, config, cache, ckey, cache_eligible, is_fallback)
            _emit_success(result, provider, cached=False, latency_ms=result.latency_ms)
            return result
    finally:
        client.close()
    return _fail(last_error)


async def _arun(req, *, config, transport, api_key, api_base, api_version) -> ChatResponse:
    cache = get_cache()
    hit, ckey, cache_eligible, primary = _cache_lookup(req, config, cache)
    if hit is not None:
        _emit_success(hit, primary or hit.provider, cached=True, latency_ms=0)
        return hit

    _check_spend(config)
    breaker = _get_breaker(config)
    policy = RetryPolicy.from_config(config)
    last_error: Optional[APIError] = None

    client = httpx.AsyncClient(timeout=config.timeout, transport=transport)
    try:
        for model in _candidates(req):
            is_fallback = model != req.model
            try:
                spec, provider, key, target = _resolve(
                    model, api_key=api_key, api_base=api_base, api_version=api_version
                )
            except APIError as exc:
                last_error = exc
                continue
            if config.circuit_breaker_enabled and not breaker.allow(provider):
                last_error = ProviderError(f"{provider} circuit is open", status_code=503)
                continue
            req_for_model = req if not is_fallback else req.model_copy(update={"model": model})
            t0 = time.perf_counter()

            async def _thunk(s=spec, r=req_for_model, k=key, tg=target):
                return await _call_async(client, s, r, k, tg)

            try:
                result = await retry_async(_thunk, policy=policy)
            except APIError as exc:
                last_error = exc
                if config.circuit_breaker_enabled:
                    (breaker.record_failure if is_retryable(exc) else breaker.record_success)(provider)
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = ProviderError(f"{provider} call failed: {exc}", status_code=502)
                if config.circuit_breaker_enabled:
                    breaker.record_failure(provider)
                continue
            if config.circuit_breaker_enabled:
                breaker.record_success(provider)
            _finalize(result, model, is_fallback, t0)
            _account_and_cache(result, req, config, cache, ckey, cache_eligible, is_fallback)
            _emit_success(result, provider, cached=False, latency_ms=result.latency_ms)
            return result
    finally:
        await client.aclose()
    return _fail(last_error)


def _stream_run(req, *, config, transport, api_key, api_base, api_version) -> Iterator[StreamChunk]:
    spec, _provider, key, target = _resolve(
        req.model, api_key=api_key, api_base=api_base, api_version=api_version
    )
    client = httpx.Client(timeout=config.timeout, transport=transport)
    try:
        yield from _stream_sync(client, spec, req, key, target)
    finally:
        client.close()


async def _astream_run(req, *, config, transport, api_key, api_base, api_version) -> AsyncIterator[StreamChunk]:
    spec, _provider, key, target = _resolve(
        req.model, api_key=api_key, api_base=api_base, api_version=api_version
    )
    client = httpx.AsyncClient(timeout=config.timeout, transport=transport)
    try:
        async for chunk in _stream_async(client, spec, req, key, target):
            yield chunk
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def completion(
    *,
    model: str,
    messages: Any,
    stream: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Optional[Union[str, list[str]]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    fallbacks: Optional[list[str]] = None,
    cache: Optional[bool] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    timeout: Optional[float] = None,
    num_retries: Optional[int] = None,
    transport: Optional[httpx.BaseTransport] = None,
    user: Optional[str] = None,  # accepted for litellm parity; not sent upstream
) -> Union[ChatResponse, Iterator[StreamChunk]]:
    """Synchronous in-process completion (calls the provider directly).

    Returns a :class:`ChatResponse`, or an ``Iterator[StreamChunk]`` when
    ``stream=True``. Keys come from ``api_key=`` or the provider env var.
    """
    req = _prepare(
        model, messages, stream=stream, fallbacks=fallbacks,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        seed=seed, cache=cache,
    )
    cfg = _effective_config(timeout, num_retries)
    if stream:
        return _stream_run(req, config=cfg, transport=transport, api_key=api_key,
                           api_base=api_base, api_version=api_version)
    return _run(req, config=cfg, transport=transport, api_key=api_key,
                api_base=api_base, api_version=api_version)


async def acompletion(
    *,
    model: str,
    messages: Any,
    stream: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stop: Optional[Union[str, list[str]]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    fallbacks: Optional[list[str]] = None,
    cache: Optional[bool] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    timeout: Optional[float] = None,
    num_retries: Optional[int] = None,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    user: Optional[str] = None,
) -> Union[ChatResponse, AsyncIterator[StreamChunk]]:
    """Asynchronous twin of :func:`completion`.

    ``await acompletion(...)`` returns a :class:`ChatResponse`; with
    ``stream=True`` it returns an ``AsyncIterator[StreamChunk]`` to iterate.
    """
    req = _prepare(
        model, messages, stream=stream, fallbacks=fallbacks,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop,
        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        seed=seed, cache=cache,
    )
    cfg = _effective_config(timeout, num_retries)
    if stream:
        return _astream_run(req, config=cfg, transport=transport, api_key=api_key,
                            api_base=api_base, api_version=api_version)
    return await _arun(req, config=cfg, transport=transport, api_key=api_key,
                       api_base=api_base, api_version=api_version)


_CONFIG_FIELDS = {f.name for f in dataclasses.fields(EngineConfig)}
_KEY_KWARGS = {
    "openai_api_key": "openai",
    "anthropic_api_key": "anthropic",
    "gemini_api_key": "gemini",
    "azure_api_key": "azure",
}


def configure(**kwargs: Any) -> None:
    """Set process-global engine defaults and/or provider keys.

    Accepts any :class:`EngineConfig` field (e.g. ``cache_enabled=True``,
    ``max_spend_usd=5``, ``timeout=30``), provider keys
    (``openai_api_key=``/``anthropic_api_key=``/``gemini_api_key=``/
    ``azure_api_key=``), and ``azure_endpoint=`` / ``azure_api_version=``.
    Unknown arguments raise ``ValueError``.
    """
    config_overrides: dict[str, Any] = {}
    for name in list(kwargs):
        if name in _CONFIG_FIELDS:
            config_overrides[name] = kwargs.pop(name)
    for kw, provider in _KEY_KWARGS.items():
        if kw in kwargs:
            set_override(provider, kwargs.pop(kw))
    if "azure_endpoint" in kwargs:
        set_azure(endpoint=kwargs.pop("azure_endpoint"))
    if "azure_api_version" in kwargs:
        set_azure(api_version=kwargs.pop("azure_api_version"))
    if kwargs:
        raise ValueError(f"Unknown configure() arguments: {sorted(kwargs)}")
    if config_overrides:
        set_config(dataclasses.replace(get_config(), **config_overrides))


def register_callback(*, on_success=None, on_failure=None) -> None:
    """Register success/failure callbacks (see :mod:`omnigate.callbacks`)."""
    _register_callback(on_success=on_success, on_failure=on_failure)
