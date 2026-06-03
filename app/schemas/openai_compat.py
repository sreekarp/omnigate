"""OpenAI Chat Completions wire DTOs + translation to/from internal schemas.

These models mirror OpenAI's ``/v1/chat/completions`` request/response shapes so
the official OpenAI SDK can talk to the gateway unchanged. They are pure DTOs:
:func:`to_internal_chat_request` translates an inbound request into the gateway's
own :class:`~app.schemas.chat.ChatRequest`, and the ``build_*`` helpers reshape a
:class:`~app.schemas.chat.ChatResponse` (or streamed text) back into the OpenAI
wire format as compact ``dict`` payloads (the router serialises them with
``json.dumps(..., separators=(",", ":"))``).

Design notes / known limitations (documented intentionally):
- Internal ``Role`` is string-only ``system|user|assistant``. ``developer`` is
  normalised to ``system``; ``tool``/``function`` roles raise ``ValueError``.
- Multimodal (list) message content is rejected with ``ValueError`` — the
  internal schema is text-only for v1.
- ``n > 1`` raises ``ValueError`` (only a single choice is produced).
- ``created`` is always a caller-supplied parameter; this module never reads the
  clock (the router owns time, keeping these helpers deterministic/testable).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat import ChatRequest, ChatResponse, Message

# ---------------------------------------------------------------------------
# Request DTOs
# ---------------------------------------------------------------------------


class OAIChatMessage(BaseModel):
    """An inbound OpenAI chat message.

    ``content`` tolerates ``None`` (coerced to ``""`` on translation) and a
    multimodal list (rejected on translation). ``name`` is accepted and ignored.
    """

    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | None | list[Any] = None
    name: str | None = None


class OAIChatCompletionRequest(BaseModel):
    """OpenAI ``/v1/chat/completions`` request body.

    Tolerant of unknown fields (``extra="ignore"``) so SDK-only knobs we don't
    map (``logit_bias``, ``logprobs``, ``response_format``, ...) don't 422.
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[OAIChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    seed: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    user: str | None = None
    n: int | None = Field(default=None, ge=1)

    def wants_stream_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))


# Mapping of inbound OpenAI roles onto the internal text-only role set.
_ROLE_MAP: dict[str, str] = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "developer": "system",
}
_UNSUPPORTED_ROLES: frozenset[str] = frozenset({"tool", "function"})


def to_internal_chat_request(req: OAIChatCompletionRequest) -> ChatRequest:
    """Translate an OpenAI request DTO into the internal :class:`ChatRequest`.

    Raises ``ValueError`` for unsupported features (tool/function roles,
    multimodal content, ``n > 1``). ``developer`` role is folded to ``system``;
    ``content=None`` becomes ``""``; ``max_tokens`` falls back to
    ``max_completion_tokens``.
    """
    if req.n is not None and req.n > 1:
        raise ValueError("n > 1 is not supported")

    messages: list[Message] = []
    for msg in req.messages:
        role = msg.role
        if role in _UNSUPPORTED_ROLES:
            raise ValueError(f"role {role!r} not supported")
        mapped = _ROLE_MAP.get(role)
        if mapped is None:
            raise ValueError(f"role {role!r} not supported")
        if isinstance(msg.content, list):
            raise ValueError("multimodal not supported")
        content = "" if msg.content is None else msg.content
        messages.append(Message(role=mapped, content=content))

    return ChatRequest(
        model=req.model,
        messages=messages,
        max_tokens=req.max_tokens or req.max_completion_tokens,
        temperature=req.temperature,
        stream=req.stream,
        top_p=req.top_p,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        seed=req.seed,
    )


# ---------------------------------------------------------------------------
# Response DTOs
# ---------------------------------------------------------------------------


class OAIResponseMessage(BaseModel):
    role: str = "assistant"
    content: str = ""


class OAIChoice(BaseModel):
    index: int = 0
    message: OAIResponseMessage
    finish_reason: str | None = "stop"


class OAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OAIChatCompletionResponse(BaseModel):
    """OpenAI non-streaming chat-completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OAIChoice]
    usage: OAIUsage


# ---------------------------------------------------------------------------
# Builders (compact dicts; router does the json.dumps)
# ---------------------------------------------------------------------------


def build_completion_response(resp: ChatResponse, created: int) -> dict[str, Any]:
    """Build the OpenAI non-streaming response dict from an internal response.

    ``created`` is supplied by the router (this module never reads the clock).
    The gateway-stable ``resp.id`` is used as the wire ``id``.
    """
    return {
        "id": resp.id,
        "object": "chat.completion",
        "created": created,
        "model": resp.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": resp.content},
                "finish_reason": resp.finish_reason or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }


def build_chunk(
    delta_text: str,
    *,
    id: str,
    created: int,
    model: str,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build one ``chat.completion.chunk`` dict for an SSE content frame.

    ``delta`` carries ``content`` only when ``delta_text`` is non-empty, so an
    empty-text terminal frame yields ``"delta": {}`` (matching OpenAI's final
    stop chunk).
    """
    delta: dict[str, Any] = {}
    if delta_text:
        delta["content"] = delta_text
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def build_role_chunk(*, id: str, created: int, model: str) -> dict[str, Any]:
    """Build the initial role-primer ``chat.completion.chunk`` (delta.role)."""
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }


def build_usage_chunk(
    *,
    id: str,
    created: int,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> dict[str, Any]:
    """Build the final usage-only chunk (``choices: []``, ``usage`` populated).

    Emitted only when the client requested ``stream_options.include_usage``.
    """
    return {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
